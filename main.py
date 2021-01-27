import ctypes
import datetime
import enum
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np
import plac
from PIL import Image, ImageOps


class KeyByteCodes:
    ESCAPE = b'\x1b'
    ONE = b'1'
    TWO = b'2'


class Axis(enum.Enum):
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()


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

        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

        # Set shaders source
        gl.glShaderSource(vertex, vertex_code)
        gl.glShaderSource(fragment, fragment_code)

        # Compile shaders
        gl.glCompileShader(vertex)
        if not gl.glGetShaderiv(vertex, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(vertex).decode()
            print(error)
            raise RuntimeError("Shader compilation error")

        gl.glCompileShader(fragment)
        gl.glCompileShader(fragment)
        if not gl.glGetShaderiv(fragment, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(fragment).decode()
            print(error)
            raise RuntimeError("Shader compilation error")

        # Attach shader objects to the program
        gl.glAttachShader(self.program, vertex)
        gl.glAttachShader(self.program, fragment)

        # Build program
        gl.glLinkProgram(self.program)
        if not gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS):
            print(gl.glGetProgramInfoLog(self.program))
            raise RuntimeError('Linking error')

        # Get rid of shaders (no more needed)
        gl.glDetachShader(self.program, vertex)
        gl.glDetachShader(self.program, fragment)

    def generate_uniform_location(self, uniform_name):
        self.uniforms[uniform_name] = gl.glGetUniformLocation(self.program, uniform_name)

    def generate_attribute_location(self, attribute_name):
        self.attributes[attribute_name] = gl.glGetAttribLocation(self.program, attribute_name)

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


class QuadRenderer:
    def __init__(self, colour_image, depth_map,
                 default_shader_program: ShaderProgram,
                 debug_shader_program: Optional[ShaderProgram] = None,
                 displacement_factor=1.0,
                 window_name='Hello world!', window_size=(512, 512), fps=60,
                 mesh_density=0):
        """
        :param colour_image: The colour image (texture) to render on the quad.
        :param depth_map: The depth map to use for displacing the rendered mesh.
        :param window_name: The name of the window to use for rendering.
        :param window_size: The width and height of the window to use for rendering.
        :param fps: The target frames per second to draw at.
        """
        # TODO: Get vertex displacement in shaders working properly.
        # GLUT init
        # --------------------------------------
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
        glut.glutCreateWindow(window_name)
        glut.glutReshapeWindow(*window_size)
        glut.glutReshapeFunc(self.reshape)
        glut.glutIdleFunc(self.idle)
        glut.glutDisplayFunc(self.display)
        glut.glutKeyboardFunc(self.keyboard)

        self.log(f"GL_VERSION: {str(gl.glGetString(gl.GL_VERSION), 'utf-8')}")
        self.log(f"GL_RENDERER: {str(gl.glGetString(gl.GL_RENDERER), 'utf-8')}")
        self.log(f"GL_VENDOR: {str(gl.glGetString(gl.GL_VENDOR), 'utf-8')}")
        self.log(f"GLUT_API_VERSION: {glut.GLUT_API_VERSION}")

        gl.glEnable(gl.GL_CULL_FACE)
        # TODO: Get program working with depth test.
        # gl.glEnable(gl.GL_DEPTH_TEST)

        # Build & activate program
        # --------------------------------------
        # Request a program and shader slots from GPU
        self.displacement_factor = displacement_factor
        self.default_shader = default_shader_program
        self.debug_shader = debug_shader_program

        self.colour_sampler_uniform = "colourSampler"
        self.depth_sampler_uniform = "depthSampler"
        self.mvp_uniform = "mvp"
        self.displacement_uniform = "displacementFactor"

        self.position_attribute = "position"
        self.texcoord_attribute = "texcoord"

        def setup_shader_program(shader):
            shader.compile_and_link()
            shader.bind()

            shader.generate_uniform_location(self.colour_sampler_uniform)
            shader.generate_uniform_location(self.depth_sampler_uniform)
            shader.generate_uniform_location(self.mvp_uniform)
            shader.generate_uniform_location(self.displacement_uniform)

            shader.generate_attribute_location(self.position_attribute)
            shader.generate_attribute_location(self.texcoord_attribute)

            gl.glUniform1i(shader.get_uniform_location(self.colour_sampler_uniform), 0)
            gl.glUniform1i(shader.get_uniform_location(self.depth_sampler_uniform), 1)
            gl.glUniform1f(shader.get_uniform_location(self.displacement_uniform), self.displacement_factor)
            shader.unbind()

        setup_shader_program(self.default_shader)

        if self.debug_shader is not None:
            setup_shader_program(self.debug_shader)

        self.shader = self.default_shader
        self.shader.bind()

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        vertices, vertex_indices, texture_coords, num_triangles = \
            QuadRenderer.generate_vertices_and_texture_coordinates(mesh_density)
        self.num_vertices = len(vertices)
        self.num_indices = vertex_indices.size
        self.num_indices_per_strip = vertex_indices.shape[0]
        self.num_strips = vertex_indices.shape[1]
        self.num_triangles = num_triangles
        print(self.num_indices)

        self.vertex_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        gl.glEnableVertexAttribArray(self.shader.get_attribute_location(self.position_attribute))
        gl.glVertexAttribPointer(self.shader.get_attribute_location(self.position_attribute), 3, gl.GL_FLOAT, False, vertices.strides[0], ctypes.c_void_p(0))

        self.uv_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.uv_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, texture_coords.nbytes, texture_coords, gl.GL_STATIC_DRAW)

        gl.glEnableVertexAttribArray(self.shader.get_attribute_location(self.texcoord_attribute))
        gl.glVertexAttribPointer(self.shader.get_attribute_location(self.texcoord_attribute), 2, gl.GL_FLOAT, False, texture_coords.strides[0],
                                 ctypes.c_void_p(0))

        self.indices_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_buffer)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, vertex_indices.nbytes, vertex_indices.ravel(), gl.GL_STATIC_DRAW)

        self.colour_texture_id = self.load_texture(colour_image)
        self.depth_texture_id = self.load_texture(depth_map)

        self.view: np.ndarray = np.eye(4, dtype=np.float32)
        self.model: np.ndarray = np.eye(4, dtype=np.float32)
        self.projection: np.ndarray = np.eye(4, dtype=np.float32)

        self.last_frame_time: datetime.datetime = datetime.datetime.now()
        self.fps: float = fps
        self.update: Optional[Callable[[float], None]] = None

    @staticmethod
    def generate_vertices_and_texture_coordinates(density=0):
        # TODO: Adapt grid dimensions to image/depth dimensions.
        assert density % 1 == 0, f"Density must be a whole number, got {density}."
        assert density >= 0, f"Density must be a non-negative number, got {density}."

        x, y = np.linspace(-1, 1, 2 ** density + 1, dtype=np.float32), np.linspace(1, -1, 2 ** density + 1,
                                                                                   dtype=np.float32)
        x_texture, y_texture = np.linspace(0, 1, 2 ** density + 1, dtype=np.float32), np.linspace(1, 0,
                                                                                                  2 ** density + 1,
                                                                                                  dtype=np.float32)

        vertices = []
        texture_coordinates = []

        for row in range(len(y)):
            for col in range(len(x)):
                vertices.append((x[col], y[row], 0.0))
                texture_coordinates.append((x_texture[col], y_texture[row]))

        vertex_indices = []

        for i in range(len(x) * (len(x) - 1)):
            vertex_indices.append(i)
            vertex_indices.append(i + len(x))

        vertices = np.array(vertices, dtype=np.float32)
        texture_coordinates = np.array(texture_coordinates, dtype=np.float32)
        vertex_indices = np.array(vertex_indices, dtype=np.uint32)

        num_triangles = (len(y) - 1) * 2 * (len(x) - 1)
        vertex_indices = vertex_indices.reshape((2 * len(x), -1))

        return vertices, vertex_indices, texture_coordinates, num_triangles

    @staticmethod
    def load_texture(image):
        assert isinstance(image, np.ndarray) and len(image.shape) == 3, \
            f"Image should be a image stored in a numpy array with exactly three dimensions (height, width and colour " \
            f"channels). Got an image of type {type(image)} with {len(image.shape)} dimensions."

        height, width, _ = image.shape

        texture = gl.glGenTextures(1)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

        # Texture parameters are part of the texture object, so you need to
        # specify them only once for a given texture object.
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, image)

        return texture

    def run(self):
        """
        Run the application.

        Blocks until execution is finished.
        """
        glut.glutMainLoop()

    def idle(self):
        now = datetime.datetime.now()
        delta = (now - self.last_frame_time).total_seconds()

        if delta > 1.0 / self.fps:
            if self.update:
                self.update(delta)

            self.last_frame_time = now

    def display(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glBindVertexArray(self.vao)
        self.shader.bind()

        mvp = self.projection @ self.view @ self.model
        gl.glUniformMatrix4fv(self.shader.get_uniform_location(self.mvp_uniform), 1, gl.GL_TRUE, mvp)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.colour_texture_id)
        gl.glUniform1i(self.shader.get_uniform_location(self.colour_sampler_uniform), 0)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture_id)
        gl.glUniform1i(self.shader.get_uniform_location(self.depth_sampler_uniform), 1)

        gl.glEnableVertexAttribArray(self.shader.get_attribute_location(self.position_attribute))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer)

        gl.glEnableVertexAttribArray(self.shader.get_attribute_location(self.texcoord_attribute))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.uv_buffer)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_buffer)

        gl.glDrawElements(gl.GL_TRIANGLE_STRIP, self.num_indices, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))

        gl.glDisableVertexAttribArray(self.shader.get_attribute_location(self.texcoord_attribute))
        gl.glDisableVertexAttribArray(self.shader.get_attribute_location(self.position_attribute))

        gl.glBindVertexArray(0)

        self.shader.unbind()

        glut.glutSwapBuffers()
        glut.glutPostRedisplay()

    def free_buffers(self):
        gl.glDeleteBuffers(1, self.vertex_buffer)
        gl.glDeleteBuffers(1, self.uv_buffer)
        gl.glDeleteBuffers(1, self.indices_buffer)
        self.default_shader.cleanup()
        self.debug_shader.cleanup()
        gl.glDeleteTextures(2, [self.colour_texture_id, self.depth_texture_id])
        gl.glDeleteVertexArrays(1, self.vao)

    def reshape(self, width, height):
        gl.glViewport(0, 0, width, height)

    def keyboard(self, key, x, y):
        # TODO: Add ability to zoom in.
        if key == KeyByteCodes.ESCAPE:
            sys.exit()
        # TODO: Print key mappings to console on program launch.
        elif key == KeyByteCodes.ONE:
            self.shader.unbind()
            self.shader = self.default_shader
        elif key == KeyByteCodes.TWO:
            self.shader.unbind()
            self.shader = self.debug_shader

    @staticmethod
    def log(message):
        """
        Print a message to stdout with a timestamp.
        :param message: The message to print.
        """
        print(f"[{datetime.datetime.now()}] {message}")

    @staticmethod
    def get_rotation_matrix(angle, axis=Axis.X, dtype=np.float32, degrees=False):
        """
        Get the 4x4 rotation matrix for the given angle and rotation axis.

        :param angle: The angle to rotate by.
        :param axis: The axis to rotate about.
        :param dtype: The data type of the rotation matrix.
        :param degrees: Whether the angle is in degrees (=True) or radians (=False).
        :return: The rotation matrix.
        """
        rotation_matrix = np.eye(4, dtype=dtype)

        if degrees:
            angle = np.deg2rad(angle)

        if axis == Axis.X:
            rotation_matrix[1, 1] = np.cos(angle)
            rotation_matrix[1, 2] = -np.sin(angle)
            rotation_matrix[2, 1] = np.sin(angle)
            rotation_matrix[2, 2] = np.cos(angle)
        elif axis == Axis.Y:
            rotation_matrix[0, 0] = np.cos(angle)
            rotation_matrix[0, 2] = np.sin(angle)
            rotation_matrix[2, 0] = -np.sin(angle)
            rotation_matrix[2, 2] = np.cos(angle)
        elif axis == Axis.Z:
            rotation_matrix[0, 0] = np.cos(angle)
            rotation_matrix[0, 1] = -np.sin(angle)
            rotation_matrix[1, 0] = np.sin(angle)
            rotation_matrix[1, 1] = np.cos(angle)
        else:
            raise RuntimeError(f"Invalid value for argument 'axis'; Expected type {Axis}, got {type(axis)}.")

        return rotation_matrix

    @staticmethod
    def get_translation_matrix(dx: float = 0, dy: float = 0, dz: float = 0, dtype=np.float32):
        """
        Get the 4x4 translation matrix for the given displacement values across the x, y and z axes.

        :param dx: The translation on the x axis.
        :param dy: The translation on the y axis.
        :param dz: The translation on the z axis.
        :param dtype: The data type of the translation matrix.
        :return: The 4x4 translation matrix.
        """
        translation_matrix = np.eye(4, dtype=dtype)

        translation_matrix[0, 3] = dx
        translation_matrix[1, 3] = dy
        translation_matrix[2, 3] = dz

        return translation_matrix

    @staticmethod
    def get_scale_matrix(sx: float = 1, sy: Optional[float] = None, sz: Optional[float] = None, dtype=np.float32):
        """
        Get the 4x4 scale matrix for the given scales for the x, y and z axes.

        :param sx: The scale of the x axis. If either `sy` or `sz` are set to `None`, then this value will be used for all axes.
        :param sy: The scale of the y axis.
        :param sz: The scale of the z axis.
        :param dtype: The data type of the scale matrix.
        :return: The 4x4 scale matrix.
        """
        scale_matrix = np.eye(4, dtype=dtype)

        if sy is None or sz is None:
            sy = sx
            sz = sx

        scale_matrix[0, 0] = sx
        scale_matrix[1, 1] = sy
        scale_matrix[2, 2] = sz

        return scale_matrix


def load_image(fp):
    img = Image.open(fp)
    img_data = np.asarray(img)

    if img.mode == 'RGBA':
        img_data = np.delete(img_data, -1, -1)  # Drop the alpha channel (the last in the channels dimension)

    # Images need to flipped vertically to be displayed the right way up.
    img = np.flip(img_data, axis=0)

    return img


def load_depth(fp, scaling_factor=1.0):
    """
    Load a depth map from disk, using an optional scaling factor to scale the depth values.

    :param fp: The file pointer (string path or file object) for the depth map file.
    :param scaling_factor: (optional) The scaling factor to apply to the depth map. The depth values are divided by this
        value.
    :return: The loaded depth map as a numpy array.
    """
    depth_map = load_image(fp) / scaling_factor

    if len(depth_map.shape) == 2:
        depth_map = np.expand_dims(depth_map, axis=2)

    depth_map = np.concatenate(3 * [depth_map], axis=2)

    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (255 * depth_map).astype(np.uint8)

    return depth_map


@plac.annotations(
    image_path=plac.Annotation(
        help='The path to the colour image.',
        kind='positional',
        type=Path
    ),
    depth_path=plac.Annotation(
        help='The path to depth map corresponding to the colour image.',
        kind='positional',
        type=Path
    ),
    depth_scaling_factor=plac.Annotation(
        help='The scaling factor to apply to the depth maps (depth values are divided by this value).',
        kind='option',
        type=float
    ),
    displacement_factor=plac.Annotation(
        help='A multiplicative scaling factor to scale the normalised depth value when displacing the mesh vertices. ',
        kind='option',
        type=float
    ),
    fps=plac.Annotation(
        help='The target frames per second at which to render.',
        kind='option',
        type=float
    ),
    window_width=plac.Annotation(
        help='The width and height of the window to display the rendered images in.',
        kind='option',
        type=int
    ),
    window_height=plac.Annotation(
        help='The height of the window to display the rendered images in.',
        kind='option',
        type=int
    ),
    mesh_density=plac.Annotation(
        help='How fine the generated mesh should be. Increasing this value by one roughly quadruples the number of vertices.',
        kind='option',
        type=int
    ),
    output_path=plac.Annotation(
        help='The path to save any output frames to.',
        kind='option',
        type=Path
    )
)
def main(image_path="brick_wall.jpg", depth_path="depth.png", depth_scaling_factor=1.0, displacement_factor=1.0,
         fps=60, window_width=512, window_height=512, mesh_density=8, output_path='frames'):
    """
    Render a colour/depth image pair on a grid mesh in OpenGL using the depth map to displace vertices on the mesh.

    :param image_path: The path to the colour image.
    :param depth_path: The path to depth map corresponding to the colour image.
    :param depth_scaling_factor: The scaling factor to apply to the depth maps (depth values are divided by this value).
    :param displacement_factor: A multiplicative scaling factor to scale the normalised depth value when displacing the mesh vertices.
    :param fps: The target frames per second at which to render.
    :param window_width: The width of the window to display the rendered images in.
    :param window_height: The height of the window to display the rendered images in.
    :param mesh_density: How fine the generated mesh should be. Increasing this value by one roughly quadruples the
        number of vertices.
    :param output_path: The path to save any output frames to.
    """
    colour = load_image(image_path)
    depth = load_depth(depth_path, depth_scaling_factor)

    # TODO: Set window width/height according to image dimensions.
    fov_y = 60
    aspect_ratio = window_width / window_height
    near = 0.0
    far = 1000.0

    # TODO: Make shader source paths configurable.
    default_shader = ShaderProgram(vertex_shader_path='shader.vert', fragment_shader_path='shader.frag')
    debug_shader = ShaderProgram(vertex_shader_path='shader.vert', fragment_shader_path='depth_debug.frag')

    renderer = QuadRenderer(colour, depth,
                            default_shader_program=default_shader, debug_shader_program=debug_shader,
                            displacement_factor=displacement_factor,
                            fps=fps, window_size=(window_width, window_height), mesh_density=mesh_density)
    renderer.model = QuadRenderer.get_scale_matrix(1.0) @ renderer.model
    QuadRenderer.log(f"Model: \n{renderer.model}")

    renderer.view = QuadRenderer.get_rotation_matrix(angle=30, axis=Axis.X, degrees=True) @ renderer.view
    QuadRenderer.log(f"View: \n{renderer.view}")
    renderer.view = QuadRenderer.get_translation_matrix(dz=-70) @ renderer.view
    QuadRenderer.log(f"View: \n{renderer.view}")

    renderer.projection = np.array(
        [[fov_y / aspect_ratio, 0, 0, 0],
         [0, fov_y, 0, 0],
         [0, 0, (far + near) / (near - far), (2 * near * far) / (near - far)],
         [0, 0, -1, 0]],
        dtype=np.float32
    )
    QuadRenderer.log(f"Projection: \n{renderer.projection}")

    class Task:
        def __init__(self, task):
            self.task = task

        def __call__(self, *args, **kwargs):
            self.has_done = True
            return self.task(*args, **kwargs)

    class OneTimeTask(Task):
        def __init__(self, task):
            super().__init__(task)

            self.has_done = False

        def __call__(self, *args, **kwargs):
            if not self.has_done:
                self.has_done = True
                return super().__call__(*args, **kwargs)

    class RecurringTask(Task):
        def __init__(self, task, frequency=1):
            super().__init__(task)
            self.frequency = frequency
            self.call_count = 0

        def __call__(self, *args, **kwargs):
            self.call_count += 1

            if self.call_count % self.frequency == 0:
                return super().__call__(*args, **kwargs)

    def render_to_image(output_path, file_format='PNG'):
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        # TODO: Get image width/height from window size.
        data = gl.glReadPixels(0, 0, 512, 512, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (512, 512), data)
        image = ImageOps.flip(image)  # in my case image is flipped top-bottom for some reason
        image.save(output_path, file_format)

    render_image = OneTimeTask(render_to_image)
    render_frames = RecurringTask(render_to_image, frequency=fps)

    os.makedirs(output_path, exist_ok=True)

    def update_func(delta):
        t = QuadRenderer.get_rotation_matrix(6 * delta, axis=Axis.Y, degrees=True)

        renderer.model = t @ renderer.model

        render_image(os.path.join(output_path, 'sample_frame.png'))
        render_frames(os.path.join(output_path, f"frame_{render_frames.call_count:03d}.png"))

    renderer.update = update_func

    try:
        renderer.run()
    finally:
        renderer.free_buffers()


if __name__ == '__main__':
    plac.call(main)
