import shlex
from subprocess import Popen
from typing import Optional

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from DepthRenderer.DepthRenderer.utils import FrameTimer, log, interweave_arrays, flatten_arrays

import moderngl


class Camera:
    """
    A camera used for viewing a 3D scene.
    """

    def __init__(self, window_size, fov_y=60, near=0.01, far=1000.0):
        """
        :param window_size: The size of the window the camera is for.
        :param fov_y: The vertical field of view in degrees.
        :param near: The distance of the near plane from the camera.
        :param far: The distance of the far plane from the camera.
        """
        self.window_size = window_size

        self.fov_y = fov_y
        self.original_fov_y = fov_y
        self.near = near
        self.far = far

        self.view = np.eye(4)
        self.projection = self.get_perspective_matrix(fov_y=fov_y, aspect_ratio=self.aspect_ratio, near=near, far=far)

    @classmethod
    def get_perspective_matrix(cls, fov_y, aspect_ratio, near=0.01, far=1000.0):
        """
        Get a 4x4 perspective matrix.

        :param fov_y: The field of view angle (degrees) visible along the y-axis.
        :param aspect_ratio: The ratio of the width/height of the viewport.
        :param near: The z-coordinate for the near plane.
        :param far: The z-coordinate for the far plane.
        :return: The perspective matrix.
        """
        s = 1 / np.tan(np.deg2rad(fov_y / 2))

        return np.array(
            [
                [s * (1 / aspect_ratio), 0, 0, 0],
                [0, s, 0, 0],
                [0, 0, -(near + far) / (near - far), -(2 * near * far) / (near - far)],
                [0, 0, -1, 0]
            ],
            dtype=np.float32
        )

    @property
    def aspect_ratio(self):
        """
        The ratio between the width and height of the window.
        """
        return self.width / self.height

    @property
    def width(self):
        """
        The width of the window in pixels.
        """
        return self.window_size[0]

    @property
    def height(self):
        """
        The height of the window in pixels.
        """
        return self.window_size[1]

    @property
    def size(self):
        """
        The width and height of the camera window in pixels.
        """
        return self.width, self.height

    @property
    def view_projection_matrix(self):
        """
        The view-projection matrix (part of the model-view-projection matrix) for the camera.
        """
        return self.projection @ self.view

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
    def from_texture(texture: Image.Image, depth_map: Optional[np.ndarray] = None, density=0, depth_scalar=1.0, debug=False):
        """
        Create a mesh from a texture and optionally a depth map.

        The initial mesh is a quad on the XY plane where each side is subdivided 2^density times.
        Each vertex has its z-coordinate set to a [0.0, 1.0] normalised depth value from the closest corresponding pixel
        in the depth map.

        :param texture: The colour texture for the mesh.
        :param depth_map: (optional) The depth map (8-bit values) to create the mesh from.
        :param density: (optional) How fine the generated mesh should be. Increasing this value by one roughly quadruples the
            number of vertices.
        :param depth_scalar: (optional) Scales the depth maps values.
        :param debug: (optional) Whether to print debug info.
        :return: The resulting mesh.
        """
        assert density % 1 == 0, f"Density must be a whole number, got {density}."
        assert density >= 0, f"Density must be a non-negative number, got {density}."

        if debug:
            log("Generating mesh...")

        timer = FrameTimer()

        width, height = texture.size

        x, y = np.linspace(-1, 1, 2 ** density + 1, dtype=np.float32), \
               np.linspace(1, -1, 2 ** density + 1, dtype=np.float32)

        # Make the grid the same aspect ratio as the input depth map.
        y = y / (width / height)

        x_texture, y_texture = np.linspace(0, 1, 2 ** density + 1, dtype=np.float32), \
                               np.linspace(0, 1, 2 ** density + 1, dtype=np.float32)

        num_cols = len(x)
        num_rows = len(y)

        col_i, row_i = np.meshgrid(np.arange(num_rows), np.arange(num_cols))
        u = (col_i / num_cols * width).astype(np.int)
        v = (row_i / num_rows * height).astype(np.int)
        x_coords = x[col_i]
        y_coords = y[row_i]

        if depth_map is not None:
            if depth_map.min() < 0 or depth_map.max() > 1.0:
                raise ValueError(f"Depth maps should be normalised into the range [0.0, 1.0].")

            if len(depth_map.shape) == 3:
                z_coords = depth_map[v, u, 0]
            else:
                z_coords = depth_map[v, u]
        else:
            z_coords = np.zeros_like(x_coords)

        z_coords *= depth_scalar

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

class MeshRenderer:
    """
    Program for rendering a single mesh.
    """

    def __init__(self,
                 mesh: Mesh,
                 vertex_shader: str,
                 fragment_shader: str,
                 camera=Camera((512, 512))):
        """
        :param mesh: The mesh to render.
        :param vertex_shader: The vertex shader source code as a string.
        :param fragment_shader: The vertex shader source code as a string.
        :param camera: The camera used for viewing the mesh.
        """
        self.camera = camera
        # Create headless OpenGL context
        self.ctx = moderngl.create_context(standalone=True)

        self.shader_program = self.ctx.program(vertex_shader=vertex_shader,
                                               fragment_shader=fragment_shader)
        self.mesh = mesh
        self.vbo = self.ctx.buffer(mesh.vertices.astype(np.float32))
        self.ibo = self.ctx.buffer(mesh.indices.astype(np.int32))
        self.uv_bo = self.ctx.buffer(mesh.texture_coordinates.astype(np.float32))
        vao_content = [(self.vbo, '3f', 'in_vert'), (self.uv_bo, '2f', 'in_texcoord')]
        self.vao = self.ctx.vertex_array(program=self.shader_program, content=vao_content, index_buffer=self.ibo)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture(size=camera.size, components=4)])
        self.texture = self.ctx.texture(size=mesh.texture.size, components=3, data=mesh.texture.tobytes())

        self.mvp = self.shader_program['mvp']

    def draw(self) -> Image.Image:
        self.ctx.clear()
        self.fbo.use()
        self.fbo.clear()

        # noinspection PyUnresolvedReferences
        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST)

        mvp = camera.view_projection_matrix @ mesh.transform
        self.mvp.write(mvp.astype(np.float32))
        self.texture.use()
        # noinspection PyTypeChecker
        self.vao.render(moderngl.TRIANGLES)

        data = self.fbo.read(components=3)
        render = Image.frombytes('RGB', self.fbo.size, data)
        render = render.transpose(Image.FLIP_TOP_BOTTOM)

        return render

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
        image = Image.open('data/nyu2_test/00000_colors.png')
        depth = Image.open('data/nyu2_test/00000_depth.png')

        # mesh = Mesh.from_texture(image, density=0)
        mesh = Mesh.from_texture(image, np.asarray(depth) / 10_000, density=10, depth_scalar=10.0)
        camera = Camera(window_size=(640, 480), fov_y=30)

        # Load shaders
        with open('DepthRenderer/DepthRenderer/shaders/shader.vert', 'r') as f:
            vertex_shader = f.read()

        with open('DepthRenderer/DepthRenderer/shaders/shader.frag', 'r') as f:
            fragment_shader = f.read()

        renderer = MeshRenderer(mesh, camera=camera, vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()

        translation = np.eye(4)

        # TODO: Fix bug where translating along the x-axis appears to orbit the scene. Is the projection matrix correct?
        translation[:3, 3] = [0.0, 0.0, 1.0]

        mesh.transform = rotation @ translation

        translation = np.eye(4)
        translation[:3, 3] = [0.0, 0.0, 0.0]

        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()

        camera.view = rotation @ translation

        frame = renderer.draw()
        frame.save('frame.png')

        renderer.cleanup()
    finally:
        virtual_display_process.terminate()