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


class KeyByteCodes:
    ESCAPE = b'\x1b'


class Axis(enum.Enum):
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()


class QuadRenderer:
    def __init__(self, window_name='Hello world!', window_size=(512, 512), fps=60):
        """
        :param window_name: The name of the window to use for rendering.
        :param window_size: The width and height of the window to use for rendering.
        :param fps: The target frames per second to draw at.
        """
        vertex_code = """
            uniform mat4   mvp;           // The model, view and projection matrices as one matrix.
            attribute vec4 color;         // Vertex color
            attribute vec3 position;      // Vertex position
            varying vec4   v_color;       // Interpolated fragment color (out)
    
            void main()
            {
              gl_Position = mvp * vec4(position, 1.0);
              v_color = color;
            }
        """

        fragment_code = """
            varying vec4 v_color;
    
            void main()
            {
              gl_FragColor = v_color;
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

        # Build data
        # --------------------------------------
        # Define a basic quad that covers the screen.
        data = np.zeros(4, [("position", np.float32, 3),
                            ("color", np.float32, 4)])
        data['position'] = (-1, +1, 0), (+1, +1, 0), (-1, -1, 0), (+1, -1, 0)
        data['color'] = (0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1)

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

        # Build buffer
        # --------------------------------------

        # Request a buffer slot from GPU
        buffer = gl.glGenBuffers(1)

        # Make this buffer the default one
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)

        # Upload data
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

        # Bind the position attribute
        # --------------------------------------
        stride = data.strides[0]
        offset = ctypes.c_void_p(0)
        loc = gl.glGetAttribLocation(self.program, "position")
        gl.glEnableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

        offset = ctypes.c_void_p(data.dtype["position"].itemsize)
        loc = gl.glGetAttribLocation(self.program, "color")
        gl.glEnableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, offset)

        self.view: np.ndarray = np.eye(4, dtype=np.float32)
        self.model: np.ndarray = np.eye(4, dtype=np.float32)
        self.projection: np.ndarray = np.eye(4, dtype=np.float32)

        self.last_frame_time: datetime.datetime = datetime.datetime.now()
        self.fps: float = fps
        self.update: Optional[Callable[[float], None]] = None

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
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        mvp = self.projection @ self.view @ self.model

        loc = gl.glGetUniformLocation(self.program, "mvp")
        gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, mvp)

        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        glut.glutSwapBuffers()
        glut.glutPostRedisplay()

    def reshape(self, width, height):
        gl.glViewport(0, 0, width, height)

    def keyboard(self, key, x, y):
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
    fps = 144
    window_width = 512
    window_height = 512

    fov_y = 60
    aspect_ratio = window_width / window_height
    near = 0.0
    far = 1000.0

    renderer = QuadRenderer(fps=fps, window_size=(window_width, window_height))

    renderer.model = QuadRenderer.get_scale_matrix(1.5) @ renderer.model
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


    def update_func(delta):
        t = QuadRenderer.get_rotation_matrix(6 * delta, axis=Axis.Y, degrees=True)

        renderer.model = t @ renderer.model
        pass

    renderer.update = update_func
    renderer.run()
