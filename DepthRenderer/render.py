import ctypes
from typing import Optional, Callable

import glfw
import numpy as np
from OpenGL import GL as gl
from OpenGL.GL.ARB import pixel_buffer_object
from PIL import Image

from .utils import get_perspective_matrix, get_translation_matrix, log, interweave_arrays, flatten_arrays, FrameTimer


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
            self.fov_y *= 1.1
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

    def mouse(self, window, button, action, mods):
        # TODO: Drag mouse to rotate view.
        if button == glfw.MOUSE_BUTTON_MIDDLE:
            is_scroll_wheel_down = action != glfw.RELEASE

            if self.is_scroll_wheel_down and not is_scroll_wheel_down:
                self.prev_mouse_x = None
                self.prev_mouse_y = None

            self.is_scroll_wheel_down = is_scroll_wheel_down
        elif self.is_debug_mode:
            print(f"mouse(window={window}, button={button}, action={action}, mods={mods})")

    def mouse_wheel(self, window, x_offset, y_offset):
        if y_offset > 0:
            self.zoom_in()
        elif y_offset < 0:
            self.zoom_out()
        elif self.is_debug_mode:
            print(f"mouse_wheel(window={window}, x_offset={x_offset}, y_offset={y_offset})")

    def mouse_movement(self, window, x, y):
        if self.prev_mouse_x is not None and self.prev_mouse_y is not None:
            dx = -(self.prev_mouse_x - x)
            dy = (self.prev_mouse_y - y)

            if self.is_scroll_wheel_down:
                t = get_translation_matrix(dx / self.window_width, dy / self.window_height)
                self.view = self.view @ t

        self.prev_mouse_x = x
        self.prev_mouse_y = y

        if self.is_debug_mode:
            print(f"mouse_movement(window={window}, x={x}, y={y})")

    def keyboard(self, window, key, scancode, action, mods):
        if mods == glfw.MOD_SHIFT and key == glfw.KEY_EQUAL and action == glfw.PRESS:
            self.zoom_in()
        elif mods == glfw.MOD_SHIFT and key == glfw.KEY_MINUS and action == glfw.PRESS:
            self.zoom_out()
        elif key == glfw.KEY_0 and action == glfw.PRESS:
            self.reset_zoom()
        elif self.is_debug_mode:
            print(f"keyboard(window={window}, key={key}, scancode={scancode}, action={action}, mods={mods})")


class ShaderProgram:
    """
    A OpenGL shader program.
    """

    def __init__(self, vertex_shader_path, fragment_shader_path):
        """
        :param vertex_shader_path: The path to the vertex shader source code.
        :param fragment_shader_path: The path to the fragment shader source code.
        """
        self.program = 0
        self.uniforms = dict()
        self.attributes = dict()

        self.vertex_shader_path = vertex_shader_path
        self.fragment_shader_path = fragment_shader_path

    def compile_and_link(self):
        """
        Compile and link the shader program.
        """
        with open(self.vertex_shader_path, 'r') as f:
            vertex_code = f.read()

        with open(self.fragment_shader_path, 'r') as f:
            fragment_code = f.read()

        self.program = gl.glCreateProgram()

        # Compile shaders
        def compile_and_attach(shader_type, shader_source):
            shader = gl.glCreateShader(shader_type)
            gl.glShaderSource(shader, shader_source)

            gl.glCompileShader(shader)
            if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                error = gl.glGetShaderInfoLog(shader).decode()
                print(error)
                raise RuntimeError("Shader compilation error")

            gl.glAttachShader(self.program, shader)

            return shader

        vertex = compile_and_attach(gl.GL_VERTEX_SHADER, vertex_code)
        fragment = compile_and_attach(gl.GL_FRAGMENT_SHADER, fragment_code)

        # Build program
        gl.glLinkProgram(self.program)
        if not gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS):
            print(gl.glGetProgramInfoLog(self.program))
            raise RuntimeError('Linking error')

        # Get rid of shaders (no more needed)
        gl.glDetachShader(self.program, vertex)
        gl.glDetachShader(self.program, fragment)

    def generate_uniform_location(self, uniform_name):
        """
        Register a uniform.

        :param uniform_name: The name of the uniform to register as it appears in the source code.
        """
        location = gl.glGetUniformLocation(self.program, uniform_name)

        if location == -1:
            raise RuntimeError(f"Could not find the uniform {uniform_name}.")

        self.uniforms[uniform_name] = location

    def generate_attribute_location(self, attribute_name):
        """
        Register an attribute.

        :param attribute_name: The name of the attribute to register as it appears in the source code.
        """
        location = gl.glGetAttribLocation(self.program, attribute_name)

        if location == -1:
            raise RuntimeError(f"Could not find the attribute {attribute_name}.")

        self.attributes[attribute_name] = location

    def get_uniform_location(self, uniform_name):
        """
        Get the location of a uniform variable.

        :param uniform_name: The name of the uniform to get as it appears in the source code.
        :return: The location of the uniform or -1 if it is not registered.
        """
        return self.uniforms.get(uniform_name, -1)

    def get_attribute_location(self, attribute_name):
        """
        Get the location of an attribute variable.

        :param attribute_name: The name of the uniform to get as it appears in the source code.
        :return: The location of the attribute or -1 if it is not registered.
        """
        return self.attributes.get(attribute_name, -1)

    def bind(self):
        # Make program the default program
        gl.glUseProgram(self.program)

    def unbind(self):
        gl.glUseProgram(0)

    def cleanup(self):
        gl.glDeleteProgram(self.program)


class OpenGLInterface(object):
    """
    A general interface for drawable OpenGL objects with deferred loading via `to_gpu(...)`.
    """

    def to_gpu(self, *args, **kwargs):
        """
        Load data to GPU and do any OpenGL-context-dependent setup.
        """
        pass

    def bind(self):
        """
        Do any necessary binding of buffers etc.
        """
        pass

    def draw(self):
        """
        Draw geometry.

        Should be called after `bind()` and before `unbind()`.
        """
        pass

    def unbind(self):
        """
        Undo any of the bind operations performed in `bind()`.
        """
        pass

    def cleanup(self):
        """
        Free any allocated resources/memory.
        """
        pass


class Texture(OpenGLInterface):
    """
    RGBA texture.
    """

    def __init__(self, image):
        assert isinstance(image, np.ndarray) and len(image.shape) == 3, \
            f"Image should be a image stored in a numpy array with exactly three dimensions (height, width and colour " \
            f"channels). Got an image of type {type(image)} with {len(image.shape)} dimensions."

        self.image = image

        self.texture_id = 0
        self.texture_sampler_id = 0

    def to_gpu(self):
        height, width, _ = self.image.shape

        self.texture_id = gl.glGenTextures(1)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

        # Texture parameters are part of the texture object, so you need to
        # specify them only once for a given texture object.
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, self.image)

    def bind(self):
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glUniform1i(self.texture_sampler_id, 0)

    def cleanup(self):
        gl.glDeleteTextures(1, [self.texture_id])

    def copy(self):
        return Texture(self.image.copy())


class Mesh(OpenGLInterface):
    """
    A textured triangle mesh.
    """

    def __init__(self, texture: Texture, vertices, texture_coordinates, indices):
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

        self.vao_id = 0
        self.vertex_buffer_id = 0
        self.uv_buffer_id = 0
        self.indices_buffer_id = 0

        self.position_attribute_location = 0
        self.texture_coordinate_attribute_location = 0

    def to_gpu(self, position_attribute_location, texture_coordinate_attribute_location):
        """
        Upload the vertex data and setup buffers.

        :param position_attribute_location: The location of the 'position' attribute in the shader program.
        :param texture_coordinate_attribute_location: The location of the 'texture_coordinate' attribute in the shader program.
        """
        self.position_attribute_location = position_attribute_location
        self.texture_coordinate_attribute_location = texture_coordinate_attribute_location

        self.vao_id = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao_id)

        self.vertex_buffer_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer_id)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)

        gl.glEnableVertexAttribArray(self.position_attribute_location)
        gl.glVertexAttribPointer(self.position_attribute_location, 3, gl.GL_FLOAT, False,
                                 self.vertices.strides[0], ctypes.c_void_p(0))

        self.uv_buffer_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.uv_buffer_id)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.texture_coordinates.nbytes, self.texture_coordinates,
                        gl.GL_STATIC_DRAW)

        gl.glEnableVertexAttribArray(self.texture_coordinate_attribute_location)
        gl.glVertexAttribPointer(self.texture_coordinate_attribute_location, 2, gl.GL_FLOAT, False,
                                 self.texture_coordinates.strides[0], ctypes.c_void_p(0))

        self.indices_buffer_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_buffer_id)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, gl.GL_STATIC_DRAW)

    def bind(self):
        gl.glBindVertexArray(self.vao_id)

        gl.glEnableVertexAttribArray(self.position_attribute_location)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer_id)

        gl.glEnableVertexAttribArray(self.texture_coordinate_attribute_location)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.uv_buffer_id)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_buffer_id)

    def draw(self):
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))

    def unbind(self):
        gl.glDisableVertexAttribArray(self.texture_coordinate_attribute_location)
        gl.glDisableVertexAttribArray(self.position_attribute_location)

        gl.glBindVertexArray(0)

    def cleanup(self):
        self.unbind()

        gl.glDeleteBuffers(1, [self.vertex_buffer_id])
        gl.glDeleteBuffers(1, [self.uv_buffer_id])
        gl.glDeleteBuffers(1, [self.indices_buffer_id])
        gl.glDeleteVertexArrays(1, [self.vao_id])

    @staticmethod
    def from_texture(texture, depth_map: Optional[np.ndarray] = None, density=0):
        """
        Create a mesh from a texture and optionally a depth map.

        The initial mesh is a quad on the XY plane where each side is subdivided 2^density times.
        Each vertex has its z-coordinate set to a [0.0, 1.0] normalised depth value from the closest corresponding pixel
        in the depth map.

        :param texture: The colour texture for the mesh.
        :param depth_map: (optional) The depth map (8-bit values) to create the mesh from.
        :param density: (optional) How fine the generated mesh should be. Increasing this value by one roughly quadruples the
            number of vertices.
        :return: The resulting mesh.
        """
        assert density % 1 == 0, f"Density must be a whole number, got {density}."
        assert density >= 0, f"Density must be a non-negative number, got {density}."

        log("Generating mesh...")

        timer = FrameTimer()

        height, width = depth_map.shape[:2]

        x, y = np.linspace(-1, 1, 2 ** density + 1, dtype=np.float32), \
               np.linspace(1, -1, 2 ** density + 1, dtype=np.float32)

        # Make the grid the same aspect ratio as the input depth map.
        y = (height / width) * y - 0.5 * (1.0 - height / width) * y

        x_texture, y_texture = np.linspace(0, 1, 2 ** density + 1, dtype=np.float32), \
                               np.linspace(1, 0, 2 ** density + 1, dtype=np.float32)

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
                 default_shader_program: ShaderProgram,
                 debug_shader_program: Optional[ShaderProgram] = None,
                 window_name='Hello world!',
                 camera=Camera((512, 512)),
                 fps=60,
                 fixed_time_step=True,
                 unlimited_frame_works=False):
        """
        :param default_shader_program: The main shader program to be used.
        :param debug_shader_program: (optional) The shader used for debugging.
        :param window_name: The name of the window to use for rendering.
        :param camera: The camera used for viewing the mesh.
        :param fps: The target frames per second to draw at.
        :param fixed_time_step: Whether the time step (delta) passed to the update callback should be a fixed time step
            of `1.0 / fps` (True) or vary based on the actual time taken to render the last frame (False).
            The expected result of setting this to `True` is that no matter how fast or slow the renderer is on screen,
            the video output will always be the same. Conversely, setting this to `Fast' means that the rendered video
            will reflect exactly what was shown on screen, and as such things like stuttering may lead to missing frames
            in the rendered video.
        :param unlimited_frame_works: Disable V-Sync and run the draw-update loop as fast as possible.
        """
        self.camera = camera

        if not glfw.init():
            raise RuntimeError("Could not initialise GLFW!")
        else:
            log("GLFW successfully initialised.")

        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        video_mode = glfw.get_video_mode(glfw.get_primary_monitor())
        screen_width = video_mode.size.width

        self.window_width = int(0.5 * screen_width)
        self.window_height = int(0.5 * screen_width // camera.aspect_ratio)
        self.window = glfw.create_window(width=self.window_width, height=self.window_height,
                                         title=window_name, monitor=None, share=None)

        if not self.window:
            raise RuntimeError("Could not create window!")
        else:
            log("GLFW window successfully created.")

        glfw.make_context_current(self.window)

        if unlimited_frame_works:  ## Need to set swap interval to zero to uncap frame rate.
            glfw.swap_interval(0)

        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_mouse_button_callback(self.window, self.mouse)
        glfw.set_scroll_callback(self.window, self.mouse_wheel)
        glfw.set_cursor_pos_callback(self.window, self.mouse_movement)

        log(f"GL_VERSION: {str(gl.glGetString(gl.GL_VERSION), 'utf-8')}")
        log(f"GL_RENDERER: {str(gl.glGetString(gl.GL_RENDERER), 'utf-8')}")
        log(f"GL_VENDOR: {str(gl.glGetString(gl.GL_VENDOR), 'utf-8')}")
        log(f"GLFW_API_VERSION: {str(glfw.get_version_string())}")

        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        if pixel_buffer_object.glInitPixelBufferObjectARB():
            log(f"Pixel buffer object supported.")
        else:
            # TODO: If pixel buffer object is not supported, run with slower, synchronous glReadPixels
            raise RuntimeError(f"Pixel buffer object not supported.")

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
        self.num_channels = 4
        self.data_size = self.window_width * self.window_height * self.num_channels

        self.pbo_ids = gl.glGenBuffers(2)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.pbo_ids[0])
        gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, self.data_size, None, gl.GL_STREAM_READ)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.pbo_ids[1])
        gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, self.data_size, None, gl.GL_STREAM_READ)

        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

        self.default_shader = default_shader_program
        self.debug_shader = debug_shader_program

        self.colour_sampler_uniform = "colourSampler"
        self.mvp_uniform = "mvp"

        self.position_attribute = "position"
        self.texcoord_attribute = "texcoord"

        def setup_shader_program(shader):
            shader.compile_and_link()
            shader.bind()

            shader.generate_uniform_location(self.colour_sampler_uniform)
            shader.generate_uniform_location(self.mvp_uniform)

            shader.generate_attribute_location(self.position_attribute)
            shader.generate_attribute_location(self.texcoord_attribute)

            gl.glUniform1i(shader.get_uniform_location(self.colour_sampler_uniform), 0)
            shader.unbind()

        setup_shader_program(self.default_shader)

        if self.debug_shader is not None:
            setup_shader_program(self.debug_shader)

        self.shader = self.default_shader
        self.shader.bind()

        self._mesh: Optional[Mesh] = None

        self.pbo_index = 0
        self.current_pixel_buffer_address = None

        self.fps: float = fps
        self.target_frame_time_secs = 1.0 / fps

        self._unlimited_frame_works = unlimited_frame_works
        self.fixed_time_step = fixed_time_step
        self.frame_timer = FrameTimer()

        self.is_paused = False
        self.wireframe_mode = False
        self.is_running = True

        self.on_update: Optional[Callable[[float], None]] = None
        self.on_exit: Optional[Callable[[], None]] = None

    @property
    def unlimited_frame_works(self):
        return self._unlimited_frame_works

    @unlimited_frame_works.setter
    def unlimited_frame_works(self, value):
        self._unlimited_frame_works = value

        if self._unlimited_frame_works:
            glfw.swap_interval(0)
        else:
            glfw.swap_interval(1)

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh):
        self._mesh = mesh
        self._mesh.to_gpu(self.shader.get_attribute_location(self.position_attribute),
                          self.shader.get_attribute_location(self.texcoord_attribute))
        self._mesh.texture.to_gpu()

    @property
    def frame_buffer_shape(self):
        """
        The shape (width, height) of the frame buffer.
        """
        return self.window_width, self.window_height

    def run(self):
        """
        Run the application.

        Blocks until execution is finished.
        """
        try:
            self.frame_timer.reset()

            while not glfw.window_should_close(self.window):
                self.frame_timer.update()

                if self.unlimited_frame_works or self.frame_timer.elapsed > self.target_frame_time_secs:
                    self.draw()

                    if self.on_update is not None and not self.is_paused:
                        if self.unlimited_frame_works or self.fixed_time_step:
                            delta = self.target_frame_time_secs
                        else:
                            delta = self.frame_timer.elapsed

                        self.on_update(delta)

                    self.frame_timer.elapsed = 0.0

                glfw.poll_events()

            if self.on_exit:
                self.on_exit()
        finally:
            glfw.terminate()

    def get_frame(self):
        """
        Copy the frame pixel data from pixel buffer to the CPU.

        :return: The currently buffered frame as a PIL.Image object.
        """
        if self.current_pixel_buffer_address:
            return Image.frombytes(mode='RGBA', size=self.frame_buffer_shape, data=self.current_pixel_buffer_address)

    def copy_frame_to_pixel_buffer(self):
        """
        Copy the frame buffer pixel data to the pixel buffer and update the current buffer address.
        """

        # Alternate through multiple PBOs so that while we read from one buffer, data can be written into another buffer
        # at the same time.
        num_pbo_buffers = len(self.pbo_ids)
        self.pbo_index = (self.pbo_index + 1) % num_pbo_buffers
        pbo_index_next = (self.pbo_index + 1) % num_pbo_buffers

        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.pbo_ids[self.pbo_index])
        gl.glReadPixels(0, 0, self.window_width, self.window_height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, 0)

        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.pbo_ids[pbo_index_next])
        src = gl.glMapBuffer(gl.GL_PIXEL_PACK_BUFFER, gl.GL_READ_ONLY)

        if src:
            self.current_pixel_buffer_address = (gl.GLfloat * self.data_size).from_address(src)

            gl.glUnmapBuffer(gl.GL_PIXEL_PACK_BUFFER)

        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

    def draw(self):
        if not self.is_running:
            return

        gl.glReadBuffer(gl.GL_FRONT)
        self.copy_frame_to_pixel_buffer()
        gl.glDrawBuffer(gl.GL_BACK)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if self.mesh is not None:
            self.shader.bind()
            mvp = self.camera.view_projection_matrix @ self.mesh.transform
            gl.glUniformMatrix4fv(self.shader.get_uniform_location(self.mvp_uniform), 1, gl.GL_TRUE, mvp)

            self.mesh.texture.bind()
            self.mesh.bind()
            self.mesh.draw()
            self.mesh.unbind()
            self.mesh.texture.unbind()

        self.shader.unbind()

        glfw.swap_buffers(self.window)

    def cleanup(self):
        gl.glDeleteBuffers(len(self.pbo_ids), self.pbo_ids)

    def close(self):
        glfw.set_window_should_close(self.window, glfw.TRUE)

    def mouse(self, window, button, action, mods):
        self.camera.mouse(window, button, action, mods)

    def mouse_wheel(self, window, x_offset, y_offset):
        self.camera.mouse_wheel(window, x_offset, y_offset)

    def mouse_movement(self, window, x, y):
        self.camera.mouse_movement(window, x, y)

    def keyboard(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.close()
        # TODO: Print key mappings to console on program launch.
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            self.is_paused = not self.is_paused
        elif key == glfw.KEY_1 and action == glfw.PRESS:
            self.debug_shader.unbind()
            self.default_shader.bind()
            self.shader = self.default_shader
        elif key == glfw.KEY_2 and action == glfw.PRESS:
            self.default_shader.unbind()
            self.debug_shader.bind()
            self.shader = self.debug_shader
        elif key == glfw.KEY_3 and action == glfw.PRESS:
            self.wireframe_mode = not self.wireframe_mode

            if self.wireframe_mode:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            else:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            self.camera.keyboard(window, key, scancode, action, mods)
