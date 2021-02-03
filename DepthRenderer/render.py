import ctypes
import datetime
import sys
from typing import Optional, Callable

import numpy as np
from OpenGL import GL as gl, GLUT as glut
from OpenGL.GL.ARB import pixel_buffer_object

from .utils import get_perspective_matrix, get_translation_matrix, log, interweave_arrays, flatten_arrays


class KeyByteCodes:
    ESCAPE = b'\x1b'
    ZERO = b'0'
    ONE = b'1'
    TWO = b'2'
    THREE = b'3'
    PLUS = b'+'
    MINUS = b'-'
    UNDERSCORE = b'_'
    SPACE = b' '


class Camera:
    def __init__(self, window_size, fov_y=60, near=0.01, far=1000.0, is_debug_mode=False):
        # TODO: Update window size on window resize.
        self.window_size = window_size

        self.fov_y = fov_y
        self.original_fov_y = fov_y
        self.near = near
        self.far = far

        self.view = np.eye(4, dtype=np.float32)
        self.projection = get_perspective_matrix(self.fov_y, self.aspect_ratio, near=near, far=far)

        self.prev_mouse_x = None
        self.prev_mouse_y = None
        self.is_scroll_wheel_down = False

        self.is_debug_mode = is_debug_mode

    @property
    def aspect_ratio(self):
        return self.window_width / self.window_height

    @property
    def window_width(self):
        return self.window_size[0]

    @property
    def window_height(self):
        return self.window_size[1]

    @property
    def view_projection_matrix(self):
        return self.projection @ self.view

    def reshape(self, width, height):
        gl.glViewport(0, 0, width, height)

    def set_zoom(self, fov_y):
        fov_y = max(0.0, fov_y)

        self.projection = np.array(
            [[fov_y / self.aspect_ratio, 0, 0, 0],
             [0, fov_y, 0, 0],
             [0, 0, (self.far + self.near) / (self.near - self.far),
              (2 * self.near * self.far) / (self.near - self.far)],
             [0, 0, -1, 0]],
            dtype=np.float32
        )

    def mouse(self, button, direction, x, y):
        if button == glut.GLUT_MIDDLE_BUTTON:
            is_scroll_wheel_down = direction == glut.GLUT_DOWN

            if self.is_scroll_wheel_down and not is_scroll_wheel_down:
                self.prev_mouse_x = None
                self.prev_mouse_y = None

            self.is_scroll_wheel_down = is_scroll_wheel_down
        elif self.is_debug_mode:
            print(f"mouse(button={button}, direction={direction}, x={x}, y={y})")

    def mouse_wheel(self, wheel, direction, x, y):
        if direction > 0:
            self.fov_y += 10
            self.set_zoom(self.fov_y)
        elif direction < 0:
            self.fov_y -= 10
            self.set_zoom(self.fov_y)
        elif self.is_debug_mode:
            print(f"mouse_wheel(wheel={wheel}, direction={direction}, x={x}, y={y})")

    def mouse_movement(self, x, y):
        if self.prev_mouse_x is not None and self.prev_mouse_y is not None:
            dx = -(self.prev_mouse_x - x)
            dy = (self.prev_mouse_y - y)

            if self.is_scroll_wheel_down:
                t = get_translation_matrix(dx / self.window_width, dy / self.window_height)
                self.view = self.view @ t

        self.prev_mouse_x = x
        self.prev_mouse_y = y

        if self.is_debug_mode:
            print(f"mouse_movement(x={x}, y={y})")

    def keyboard(self, key, x, y):
        is_shift_pressed = glut.glutGetModifiers() == glut.GLUT_ACTIVE_SHIFT

        if is_shift_pressed and key == KeyByteCodes.PLUS:
            self.fov_y += 10
            self.set_zoom(self.fov_y)
        elif is_shift_pressed and key == KeyByteCodes.UNDERSCORE:
            self.fov_y -= 10
            self.set_zoom(self.fov_y)
        elif key == KeyByteCodes.ZERO:
            self.fov_y = self.original_fov_y
            self.set_zoom(self.fov_y)
        elif self.is_debug_mode:
            print(f"keyboard(x={x}, y={y})")


class ShaderProgram:
    def __init__(self, vertex_shader_path, fragment_shader_path):
        self.program = 0
        self.uniforms = dict()
        self.attributes = dict()

        self.vertex_shader_path = vertex_shader_path
        self.fragment_shader_path = fragment_shader_path

    def compile_and_link(self):
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
        location = gl.glGetUniformLocation(self.program, uniform_name)

        if location == -1:
            raise RuntimeError(f"Could not find the uniform {uniform_name}.")

        self.uniforms[uniform_name] = location

    def generate_attribute_location(self, attribute_name):
        location = gl.glGetAttribLocation(self.program, attribute_name)

        if location == -1:
            raise RuntimeError(f"Could not find the attribute {attribute_name}.")

        self.attributes[attribute_name] = location

    def get_uniform_location(self, uniform_name):
        return self.uniforms.get(uniform_name, -1)

    def get_attribute_location(self, attribute_name):
        return self.attributes.get(attribute_name, -1)

    def bind(self):
        # Make program the default program
        gl.glUseProgram(self.program)

    def unbind(self):
        gl.glUseProgram(0)

    def cleanup(self):
        gl.glDeleteProgram(self.program)


class OpenGLInterface(object):
    def to_gpu(self):
        pass

    def bind(self):
        pass

    def draw(self):
        pass

    def unbind(self):
        pass

    def cleanup(self):
        pass


class Texture(OpenGLInterface):
    def __init__(self, image):
        assert isinstance(image, np.ndarray) and len(image.shape) == 3, \
            f"Image should be a image stored in a numpy array with exactly three dimensions (height, width and colour " \
            f"channels). Got an image of type {type(image)} with {len(image.shape)} dimensions."

        self.image = image

        self.texture_id = -1
        self.texture_sampler_id = -1

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
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, self.image)

    def bind(self):
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glUniform1i(self.texture_sampler_id, 0)

    def cleanup(self):
        gl.glDeleteTextures(1, self.texture_id)


class Mesh(OpenGLInterface):
    def __init__(self, texture: Texture, vertices, texture_coordinates, indices):
        self.texture = texture
        self.vertices = vertices
        self.texture_coordinates = texture_coordinates
        self.indices = indices

        self.transform = np.eye(4, dtype=np.float32)

        self.vao_id = -1
        self.vertex_buffer_id = -1
        self.uv_buffer_id = -1
        self.indices_buffer_id = -1

        self.position_attribute_location = -1
        self.texture_coordinate_attribute_location = -1

    def to_gpu(self, position_attribute_location, texture_coordinate_attribute_location):
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
        gl.glDeleteBuffers(1, self.vertex_buffer_id)
        gl.glDeleteBuffers(1, self.uv_buffer_id)
        gl.glDeleteBuffers(1, self.indices_buffer_id)
        gl.glDeleteVertexArrays(1, self.vao_id)

    @staticmethod
    def from_depth_map(texture, depth_map, density=0):
        assert density % 1 == 0, f"Density must be a whole number, got {density}."
        assert density >= 0, f"Density must be a non-negative number, got {density}."

        height, width = depth_map.shape[:2]

        x, y = np.linspace(-1, 1, 2 ** density + 1, dtype=np.float32), \
               np.linspace(1, -1, 2 ** density + 1, dtype=np.float32)

        # Make the grid the same aspect ratio as the input depth map.
        y = (height / width) * y + (1 - (height / width)) * y

        x_texture, y_texture = np.linspace(0, 1, 2 ** density + 1, dtype=np.float32), \
                               np.linspace(1, 0, 2 ** density + 1, dtype=np.float32)

        num_cols = len(x)
        num_rows = len(y)

        col_i, row_i = np.meshgrid(np.arange(num_rows), np.arange(num_cols))
        u = (col_i / num_cols * width).astype(np.int)
        v = ((1 - row_i / num_rows) * height - 1).astype(np.int)
        x_coords = x[col_i]
        y_coords = y[row_i]
        z_coords = 1. - depth_map[v, u, 0] / 255.0
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
        print(f"Num. tris: {len(indices) // 3}")

        return Mesh(texture, vertices, texture_coordinates, indices)


class QuadRenderer:
    def __init__(self, mesh,
                 default_shader_program: ShaderProgram,
                 debug_shader_program: Optional[ShaderProgram] = None,
                 window_name='Hello world!',
                 camera=Camera((512, 512)),
                 fps=60):
        """
        :param colour_image: The colour image (texture) to render on the quad.
        :param depth_map: The depth map to use for displacing the rendered mesh.
        :param window_name: The name of the window to use for rendering.
        :param window_size: The width and height of the window to use for rendering.
        :param fps: The target frames per second to draw at.
        """
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)

        self.initial_window_width = int(0.5 * glut.glutGet(glut.GLUT_SCREEN_WIDTH))
        self.initial_window_height = int(0.5 * glut.glutGet(glut.GLUT_SCREEN_WIDTH) // camera.aspect_ratio)
        glut.glutInitWindowSize(self.initial_window_width, self.initial_window_height)

        glut.glutCreateWindow(window_name)
        glut.glutReshapeFunc(self.reshape)
        # glut.glutIdleFunc(self.idle)
        glut.glutTimerFunc(int(1000 / fps), self.idle, 0)
        glut.glutDisplayFunc(self.display)
        glut.glutKeyboardFunc(self.keyboard)
        glut.glutMouseFunc(self.mouse)
        glut.glutMouseWheelFunc(self.mouse_wheel)
        glut.glutMotionFunc(self.mouse_movement)

        log(f"GL_VERSION: {str(gl.glGetString(gl.GL_VERSION), 'utf-8')}")
        log(f"GL_RENDERER: {str(gl.glGetString(gl.GL_RENDERER), 'utf-8')}")
        log(f"GL_VENDOR: {str(gl.glGetString(gl.GL_VENDOR), 'utf-8')}")
        log(f"GLUT_API_VERSION: {glut.GLUT_API_VERSION}")

        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glEnable(gl.GL_DEPTH_TEST)

        if pixel_buffer_object.glInitPixelBufferObjectARB():
            print(f"Pixel buffer object supported.")
        else:
            # TODO: If pixel buffer object is not supported, run with slower, synchronous glReadPixels
            raise RuntimeError(f"Pixel buffer object not supported.")

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
        self.num_channels = 4
        self.data_size = self.initial_window_width * self.initial_window_height * self.num_channels

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

        self.mesh = mesh
        self.mesh.to_gpu(self.shader.get_attribute_location(self.position_attribute),
                         self.shader.get_attribute_location(self.texcoord_attribute))
        self.mesh.texture.to_gpu()

        self.last_update_time: datetime.datetime = datetime.datetime.now()
        self.last_frame_time: datetime.datetime = datetime.datetime.now()
        self.fps: float = fps
        self.target_frame_time_ms = int(1000.0 // fps)
        self.start_time = datetime.datetime.now()
        self.update: Optional[Callable[[float], None]] = None
        self.paused = False
        self.wireframe_mode = False
        self.pbo_index = 0
        self.frame_buffer = None

        self.camera = camera
        self.width = camera.window_width
        self.height = camera.window_height

    @property
    def buffer_shape(self):
        return self.initial_window_width, self.initial_window_height

    @property
    def num_pbo_buffers(self):
        return len(self.pbo_ids)

    def run(self):
        """
        Run the application.

        Blocks until execution is finished.
        """
        glut.glutMainLoop()

    def idle(self, _):
        glut.glutPostRedisplay()
        now = datetime.datetime.now()
        delta = (now - self.last_update_time).total_seconds()

        if (now - self.last_frame_time).total_seconds() > 1.0:
            print(f"Frame Time: {1000 * delta:,.2f}ms")
            self.last_frame_time = now

        if self.update and not self.paused:
            self.update(delta)

        time_to_wait_ms = int(self.target_frame_time_ms + (self.target_frame_time_ms - 1000 * delta))
        time_to_wait_ms = min(self.target_frame_time_ms, max(time_to_wait_ms, 0))

        glut.glutTimerFunc(time_to_wait_ms, self.idle, 0)

        self.last_update_time = datetime.datetime.now()

    def read_frame(self):
        self.pbo_index = (self.pbo_index + 1) % self.num_pbo_buffers
        pbo_index_next = (self.pbo_index + 1) % self.num_pbo_buffers

        gl.glReadBuffer(gl.GL_FRONT)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.pbo_ids[self.pbo_index])
        gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, 0)

        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.pbo_ids[pbo_index_next])
        src = gl.glMapBuffer(gl.GL_PIXEL_PACK_BUFFER, gl.GL_READ_ONLY)

        if src:
            self.frame_buffer = (gl.GLfloat * self.data_size).from_address(src)

            gl.glUnmapBuffer(gl.GL_PIXEL_PACK_BUFFER)

        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glDrawBuffer(gl.GL_BACK)

    def display(self):
        self.read_frame()

        self.shader.bind()

        mvp = self.camera.view_projection_matrix @ self.mesh.transform
        gl.glUniformMatrix4fv(self.shader.get_uniform_location(self.mvp_uniform), 1, gl.GL_TRUE, mvp)

        self.mesh.texture.bind()
        self.mesh.bind()
        self.mesh.draw()
        self.mesh.unbind()
        self.mesh.texture.unbind()

        self.shader.unbind()

        glut.glutSwapBuffers()

    def cleanup(self):
        gl.glDeleteBuffers(len(self.pbo_ids), self.pbo_ids)

    def reshape(self, width, height):
        self.width = int(min(glut.glutGet(glut.GLUT_SCREEN_WIDTH), self.camera.aspect_ratio * height))
        self.height = int(self.width / self.camera.aspect_ratio)

        gl.glViewport(0, 0, self.width, self.height)
        glut.glutReshapeWindow(self.width, self.height)

    def mouse(self, button, dir, x, y):
        self.camera.mouse(button, dir, x, y)

    def mouse_wheel(self, wheel, direction, x, y):
        self.camera.mouse_wheel(wheel, direction, x, y)

    def mouse_movement(self, x, y):
        self.camera.mouse_movement(x, y)

    def keyboard(self, key, x, y):
        if key == KeyByteCodes.ESCAPE:
            sys.exit()
        # TODO: Print key mappings to console on program launch.
        elif key == KeyByteCodes.SPACE:
            self.paused = not self.paused
        elif key == KeyByteCodes.ONE:
            self.debug_shader.unbind()
            self.default_shader.bind()
            self.shader = self.default_shader
        elif key == KeyByteCodes.TWO:
            self.default_shader.unbind()
            self.debug_shader.bind()
            self.shader = self.debug_shader
        elif key == KeyByteCodes.THREE:
            self.wireframe_mode = not self.wireframe_mode

            if self.wireframe_mode:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            else:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            self.camera.keyboard(key, x, y)
