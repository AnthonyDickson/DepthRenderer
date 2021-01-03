# -----------------------------------------------------------------------------
# Python & OpenGL for Scientific Visualization
# www.labri.fr/perso/nrougier/python+opengl
# Copyright (c) 2017, Nicolas P. Rougier
# Distributed under the 2-Clause BSD License.
# -----------------------------------------------------------------------------
import sys
import ctypes
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut


class KeyByteCodes:
    ESCAPE = b'\x1b'


class QuadRenderer:
    def __init__(self, window_name='Hello world!', window_size=(512, 512)):
        """
        :param window_name: The name of the window to use for rendering.
        :param window_size: The width and height of the window to use for rendering.
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
        glut.glutDisplayFunc(self.display)
        glut.glutKeyboardFunc(self.keyboard)

        # Build data
        # --------------------------------------
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

        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

    def mainloop(self):
        # Enter the mainloop
        # --------------------------------------
        glut.glutMainLoop()

    def display(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        mvp = self.projection @ self.view @ self.model

        loc = gl.glGetUniformLocation(self.program, "mvp")
        gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, mvp)

        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        glut.glutSwapBuffers()

    def reshape(self, width, height):
        gl.glViewport(0, 0, width, height)

    def keyboard(self, key, x, y):
        if key == KeyByteCodes.ESCAPE:
            sys.exit()


if __name__ == '__main__':
    renderer = QuadRenderer()
    renderer.mainloop()
