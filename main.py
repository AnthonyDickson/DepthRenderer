# -----------------------------------------------------------------------------
# Python & OpenGL for Scientific Visualization
# www.labri.fr/perso/nrougier/python+opengl
# Copyright (c) 2017, Nicolas P. Rougier
# Distributed under the 2-Clause BSD License.
# -----------------------------------------------------------------------------
import datetime
import enum
import sys
import ctypes
from typing import Callable, Optional

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut

from PIL import Image


class KeyByteCodes:
    ESCAPE = b'\x1b'


class Axis(enum.Enum):
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()


class QuadRenderer:
    def __init__(self, texture_path, depth_path, window_name='Hello world!', window_size=(512, 512), fps=60, mesh_density=0):
        """
        :param texture_path: The path to the texture file to render on the quad.
        :param window_name: The name of the window to use for rendering.
        :param window_size: The width and height of the window to use for rendering.
        :param fps: The target frames per second to draw at.
        """
        # TODO: Load shaders from disk.
        # TODO: Get vertex displacement in shaders working properly.
        vertex_code = """
            uniform mat4   mvp;           // The model, view and projection matrices as one matrix.
            attribute vec3 position;      // Vertex position
            attribute vec2 texcoord;   // Vertex texture coordinates
            varying vec2   v_texcoord;       // Interpolated fragment texture coordinates (out)
            
            uniform sampler2D colourSampler; // Texture
            uniform sampler2D depthSampler; // Depth Texture
    
            void main()
            {
              float displacement = tex2D(depthSampler, v_texcoord).r / 255.0;
              gl_Position = mvp * vec4(position.x, position.y, position.z + displacement, 1.0);
              v_texcoord = texcoord;
            }
        """

        fragment_code = """
            uniform sampler2D colourSampler; // Texture
            uniform sampler2D depthSampler; // Depth Texture  
            varying vec2 v_texcoord;   // Interpolated fragment texture coordinates (in)
    
            void main()
            {
              gl_FragColor = texture2D(colourSampler, v_texcoord);
            } 
        """

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

        # Make program the default program
        gl.glUseProgram(self.program)

        self.colour_sampler_uniform = gl.glGetUniformLocation(self.program, "colourSampler")
        self.depth_sampler_uniform = gl.glGetUniformLocation(self.program, "depthSampler")
        self.mvp_uniform = gl.glGetUniformLocation(self.program, "mvp")

        self.position_attrib = gl.glGetAttribLocation(self.program, "position")
        self.texcoord_attrib = gl.glGetAttribLocation(self.program, "texcoord")

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

        gl.glEnableVertexAttribArray(self.position_attrib)
        gl.glVertexAttribPointer(self.position_attrib, 3, gl.GL_FLOAT, False, vertices.strides[0], ctypes.c_void_p(0))

        self.uv_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.uv_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, texture_coords.nbytes, texture_coords, gl.GL_STATIC_DRAW)

        gl.glEnableVertexAttribArray(self.texcoord_attrib)
        gl.glVertexAttribPointer(self.texcoord_attrib, 2, gl.GL_FLOAT, False, texture_coords.strides[0],
                                 ctypes.c_void_p(0))

        self.indices_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_buffer)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, vertex_indices.nbytes, vertex_indices.ravel(), gl.GL_STATIC_DRAW)

        self.colour_texture_id = self.load_texture(texture_path)
        self.depth_texture_id = self.load_texture(depth_path)

        gl.glUniform1i(self.colour_sampler_uniform, 0)
        gl.glUniform1i(self.depth_sampler_uniform, 1)

        self.view: np.ndarray = np.eye(4, dtype=np.float32)
        self.model: np.ndarray = np.eye(4, dtype=np.float32)
        self.projection: np.ndarray = np.eye(4, dtype=np.float32)

        self.last_frame_time: datetime.datetime = datetime.datetime.now()
        self.fps: float = fps
        self.update: Optional[Callable[[float], None]] = None

    @staticmethod
    def generate_vertices_and_texture_coordinates(density=0):
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
    def load_texture(fp):
        img = Image.open(fp)
        img_data = np.asarray(img)

        if img.mode == 'RGBA':
            img_data = np.delete(img_data, -1, -1)  # Drop the alpha channel (the last in the channels dimension)

        # Images need to flipped vertically to be displayed the right way up.
        img_data = np.flip(img_data, axis=0)

        texture = gl.glGenTextures(1)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

        # Texture parameters are part of the texture object, so you need to
        # specify them only once for a given texture object.
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, img.size[0], img.size[1], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
                        img_data)

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
        gl.glUseProgram(self.program)

        mvp = self.projection @ self.view @ self.model
        gl.glUniformMatrix4fv(self.mvp_uniform, 1, gl.GL_TRUE, mvp)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.colour_texture_id)
        gl.glUniform1i(self.colour_sampler_uniform, 0)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture_id)
        gl.glUniform1i(self.depth_sampler_uniform, 1)

        gl.glEnableVertexAttribArray(self.position_attrib)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer)

        gl.glEnableVertexAttribArray(self.texcoord_attrib)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.uv_buffer)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_buffer)

        gl.glDrawElements(gl.GL_TRIANGLE_STRIP, self.num_indices, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))

        gl.glDisableVertexAttribArray(self.texcoord_attrib)
        gl.glDisableVertexAttribArray(self.position_attrib)

        gl.glBindVertexArray(0)

        gl.glUseProgram(0)

        glut.glutSwapBuffers()
        glut.glutPostRedisplay()

    def free_buffers(self):
        gl.glDeleteBuffers(1, self.vertex_buffer)
        gl.glDeleteBuffers(1, self.uv_buffer)
        gl.glDeleteBuffers(1, self.indices_buffer)
        gl.glDeleteProgram(self.program)
        gl.glDeleteTextures(2, [self.colour_texture_id, self.depth_texture_id])
        gl.glDeleteVertexArrays(1, self.vao)

    def reshape(self, width, height):
        gl.glViewport(0, 0, width, height)

    def keyboard(self, key, x, y):
        # TODO: Add ability to zoom in.
        if key == KeyByteCodes.ESCAPE:
            sys.exit()

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


if __name__ == '__main__':
    texture_path = "brick_wall.jpg"
    depth_path = "depth.png"

    fps = 144
    window_width = 512
    window_height = 512

    fov_y = 60
    aspect_ratio = window_width / window_height
    near = 0.0
    far = 1000.0

    renderer = QuadRenderer(texture_path, depth_path, fps=fps, window_size=(window_width, window_height), mesh_density=8)
    renderer.model = QuadRenderer.get_scale_matrix(1.0) @ renderer.model
    QuadRenderer.log(f"Model: \n{renderer.model}")

    renderer.view = QuadRenderer.get_rotation_matrix(angle=30, axis=Axis.X, degrees=True) @ renderer.view
    QuadRenderer.log(f"View: \n{renderer.view}")
    renderer.view = QuadRenderer.get_translation_matrix(dz=-100) @ renderer.view
    QuadRenderer.log(f"View: \n{renderer.view}")

    renderer.projection = np.array(
        [[fov_y / aspect_ratio, 0, 0, 0],
         [0, fov_y, 0, 0],
         [0, 0, (far + near) / (near - far), (2 * near * far) / (near - far)],
         [0, 0, -1, 0]],
        dtype=np.float32
    )
    QuadRenderer.log(f"Projection: \n{renderer.projection}")

    # TODO: Record rendered colour and depth as RGBA frames.
    def update_func(delta):
        t = QuadRenderer.get_rotation_matrix(6 * delta, axis=Axis.Y, degrees=True)

        renderer.model = t @ renderer.model
        pass


    renderer.update = update_func

    try:
        renderer.run()
    finally:
        renderer.free_buffers()
