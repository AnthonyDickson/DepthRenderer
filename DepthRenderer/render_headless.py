import shlex
from subprocess import Popen
from typing import Optional

import numpy as np
from PIL import Image
from pyrr import Matrix44

from DepthRenderer.DepthRenderer.utils import FrameTimer, log, interweave_arrays, flatten_arrays, get_perspective_matrix

import moderngl


class Camera:
    """
    A camera used for viewing a 3D scene.
    """

    def __init__(self, window_size, fov_y=60, near=0.01, far=1000.0, is_debug_mode=False, zoom_speed=10):
        """
        :param window_size: The size of the window the camera is for.
        :param fov_y: The vertical field of view in degrees.
        :param near: The distance of the near plane from the camera.
        :param far: The distance of the far plane from the camera.
        :param is_debug_mode: Whether to print debug info.
        :param zoom_speed: The speed to zoom in/out at (the degrees to change the vertical field of view at).
        """
        # TODO: Update window size on window resize.
        self.mouse_rotation_speed = 0.001
        self.near_zoom_rate = 1.05
        self.window_size = window_size

        self.fov_y = fov_y
        self.original_fov_y = fov_y
        self.near = near
        self.far = far

        self.view = np.eye(4, dtype=np.float32)
        self.projection = get_perspective_matrix(self.fov_y, self.aspect_ratio, near=near, far=far)

        self.zoom_speed = zoom_speed
        self.prev_mouse_x = None
        self.prev_mouse_y = None
        self.is_scroll_wheel_down = False
        self.is_lmb_down = False

        self.is_debug_mode = is_debug_mode

    @property
    def aspect_ratio(self):
        """
        The ratio between the width and height of the window.
        """
        return self.window_width / self.window_height

    @property
    def window_width(self):
        """
        The width of the window in pixels.
        """
        return self.window_size[0]

    @property
    def window_height(self):
        """
        The height of the window in pixels.
        """
        return self.window_size[1]

    @property
    def shape(self):
        """
        The height and width of the window in pixels.
        """
        return self.window_height, self.window_width

    @property
    def view_projection_matrix(self):
        """
        The view-projection matrix (part of the model-view-projection matrix) for the camera.
        """
        return self.projection @ self.view

    def _set_zoom(self, fov_y):
        """
        Helper function for setting the zoom based on a vertical field of view.

        :param fov_y: The vertical field of view in degrees.
        """
        fov_y = max(0.0, fov_y)

        self.projection = np.array(
            [[fov_y / self.aspect_ratio, 0, 0, 0],
             [0, fov_y, 0, 0],
             [0, 0, (self.far + self.near) / (self.near - self.far),
              (2 * self.near * self.far) / (self.near - self.far)],
             [0, 0, -1, 0]],
            dtype=np.float32
        )

    def zoom_in(self):
        """
        Zoom the camera in.
        """
        if self.fov_y < self.zoom_speed:
            self.fov_y *= self.near_zoom_rate
        else:
            self.fov_y += self.zoom_speed

        self._set_zoom(self.fov_y)

    def zoom_out(self):
        """
        Zoom the camera out.
        """
        if self.fov_y <= self.zoom_speed:
            self.fov_y *= 0.9
        else:
            self.fov_y -= self.zoom_speed

        self._set_zoom(self.fov_y)

    def reset_zoom(self):
        """
        Reset the zoom to its original value.
        """
        self.fov_y = self.original_fov_y
        self._set_zoom(self.fov_y)

class Mesh:
    """
    A textured triangle mesh.
    """

    def __init__(self, texture: Image.Image, vertices, texture_coordinates, indices):
        """
        :param texture: The mesh's texture.
        :param vertices: The vertices of the mesh.
        :param texture_coordinates: The UV coordinates mapping the texture to the mesh.
        :param indices: The indices for the triangles of the mesh.
        """
        self.texture = texture
        self.vertices = vertices
        self.texture_coordinates = texture_coordinates
        self.indices = indices

        self.transform = np.eye(4, dtype=np.float32)

    @staticmethod
    def from_texture(texture, depth_map: Optional[np.ndarray] = None, density=0, debug=False):
        """
        Create a mesh from a texture and optionally a depth map.

        The initial mesh is a quad on the XY plane where each side is subdivided 2^density times.
        Each vertex has its z-coordinate set to a [0.0, 1.0] normalised depth value from the closest corresponding pixel
        in the depth map.

        :param texture: The colour texture for the mesh.
        :param depth_map: (optional) The depth map (8-bit values) to create the mesh from.
        :param density: (optional) How fine the generated mesh should be. Increasing this value by one roughly quadruples the
            number of vertices.
        :param debug: (optional) Whether to print debug info.
        :return: The resulting mesh.
        """
        assert density % 1 == 0, f"Density must be a whole number, got {density}."
        assert density >= 0, f"Density must be a non-negative number, got {density}."

        if debug:
            log("Generating mesh...")

        timer = FrameTimer()

        height, width = depth_map.shape[:2]

        x, y = np.linspace(-1, 1, 2 ** density + 1, dtype=np.float32), \
               np.linspace(1, -1, 2 ** density + 1, dtype=np.float32)

        # Make the grid the same aspect ratio as the input depth map.
        y = (height / width) * y - 0.5 * (1.0 - height / width) * y

        x_texture, y_texture = np.linspace(0, 1, 2 ** density + 1, dtype=np.float32), \
                               np.linspace(0, 1, 2 ** density + 1, dtype=np.float32)

        num_cols = len(x)
        num_rows = len(y)

        col_i, row_i = np.meshgrid(np.arange(num_rows), np.arange(num_cols))
        u = (col_i / num_cols * width).astype(np.int)
        v = ((1 - row_i / num_rows) * height - 1).astype(np.int)
        x_coords = x[col_i]
        y_coords = y[row_i]

        if depth_map is not None:
            if len(depth_map.shape) == 3:
                z_coords = 1. - depth_map[v, u, 0] / 255.0
            else:
                z_coords = 1. - depth_map[v, u] / 255.0
        else:
            z_coords = np.ones_like(x_coords)

        u_coords = x_texture[col_i]
        v_coords = y_texture[row_i]

        # Generates the triangle indices a -> b
        #                                   /
        #                                  /
        #                                 /
        #                                c -> d
        # where the indices (a, b, c) and (c, b, d) each form a single triangle when using a clockwise winding order.
        a = row_i[:-1, :-1] * num_cols + col_i[:-1, :-1]
        b = (row_i[:-1, :-1] + 1) * num_cols + col_i[:-1, :-1]
        c = a + 1
        d = b + 1

        vertices = interweave_arrays(flatten_arrays([x_coords, y_coords, z_coords]))
        texture_coordinates = interweave_arrays(flatten_arrays([u_coords, v_coords]))
        indices = interweave_arrays(flatten_arrays([a, b, c, c, b, d]))

        vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        texture_coordinates = np.array(texture_coordinates, dtype=np.float32).reshape(-1, 2)
        indices = np.array(indices, dtype=np.uint32)

        if debug:
            log(f"Num. triangles: {len(indices) // 3:,d}")
            log(f"Num. vertices: {len(vertices):,d}")
            timer.update()
            log(f"Mesh Generation Took {1000 * timer.delta:.2f} ms "
                f"({1e9 * timer.delta / len(indices):.2f} ns per triangle)")

        return Mesh(texture, vertices, texture_coordinates, indices)

    @staticmethod
    def from_copy_with_new_depth(mesh, depth_map):
        height, width = depth_map.shape[:2]
        num_subdivisions = np.sqrt(len(mesh.vertices))

        col_i, row_i = np.meshgrid(np.arange(num_subdivisions), np.arange(num_subdivisions))
        u = (col_i / num_subdivisions * width).astype(np.int)
        v = ((1 - row_i / num_subdivisions) * height - 1).astype(np.int)
        z_coords = 1. - depth_map[v, u, 0] / 255.0

        texture = mesh.texture.copy()

        vertices = mesh.vertices.copy()
        vertices[:, 2] = z_coords.flatten()

        texture_coordinates = mesh.texture_coordinates.copy()
        indices = mesh.indices.copy()

        return Mesh(texture, vertices, texture_coordinates, indices)

class MeshRenderer:
    """
    Program for rendering a single mesh.
    """

    def __init__(self,
                 mesh: Mesh,
                 vertex_shader_path: str,
                 fragment_shader_path: str,
                 camera=Camera((512, 512))):
        """
        :param camera: The camera used for viewing the mesh.
        :param fps: The target frames per second to draw at.
        """
        self.camera = camera
        # Create headless OpenGL context
        self.ctx = moderngl.create_context(standalone=True)

        # Load shaders
        with open(vertex_shader_path, 'r') as f:
            vertex_shader_source = f.read()

        with open(fragment_shader_path, 'r') as f:
            fragment_shader_source = f.read()

        self.shader_program = self.ctx.program(vertex_shader=vertex_shader_source,
                                               fragment_shader=fragment_shader_source)

        self.mesh = mesh
        self.vbo = self.ctx.buffer(mesh.vertices.astype(np.float32))
        self.ibo = self.ctx.buffer(mesh.indices.astype(np.int32))
        self.uv_bo = self.ctx.buffer(mesh.texture_coordinates.astype(np.float32))
        vao_content = [(self.vbo, '3f', 'position'), (self.uv_bo, '2f', 'texcoord')]
        self.vao = self.ctx.vertex_array(program=self.shader_program, content=vao_content, index_buffer=self.ibo)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture(camera.shape, 4)])
        self.texture = self.ctx.texture(size=mesh.texture.size, components=3, data=mesh.texture.tobytes())

        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST)

        self.mvp = self.shader_program['mvp']

    def draw(self) -> Image.Image:
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)

        # TODO: Get working with perspective transform.
        mvp = self.camera.view @ self.mesh.transform
        # mvp = self.camera.view_projection_matrix @ self.mesh.transform
        self.mvp.write(Matrix44(mvp, dtype='f4'))

        self.texture.use()
        self.vao.render(mode=moderngl.TRIANGLES)

        data = self.fbo.read(components=3)
        image = Image.frombytes('RGB', self.fbo.size, data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        return image

    def cleanup(self):
        self.texture.release()
        self.fbo.release()
        self.vao.release()
        self.uv_bo.release()
        self.ibo.release()
        self.vbo.release()
        self.shader_program.release()
        self.ctx.release()

if __name__ == '__main__':
    # This code is needed for the Docker image to work without first entering the bash console.
    cmd = shlex.split("Xvfb :99 -screen 0 640x480x24")
    virtual_display_process = Popen(cmd)

    try:
        image = Image.open('data/nyu2_train/basement_0001a_out/1.jpg')
        depth = Image.open('data/nyu2_train/basement_0001a_out/1.png')
        mesh = Mesh.from_texture(image, np.asarray(depth), density=1)
        camera = Camera(window_size=(512, 512))

        renderer = MeshRenderer(mesh, camera=camera,
                                vertex_shader_path='DepthRenderer/DepthRenderer/shaders/shader.vert',
                                fragment_shader_path='DepthRenderer/DepthRenderer/shaders/shader.frag')

        # TODO: Test various transformations on the mesh.
        # initial_position = np.eye(4, dtype=np.float32)
        # initial_position[2, 3] = 1.0
        # mesh.transform = initial_position

        frame = renderer.draw()
        frame.save('frame.jpg')

        renderer.cleanup()
    finally:
        virtual_display_process.terminate()