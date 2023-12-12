"""This sample code shows how we can perform 3D rendering without a display in Docker."""

import shlex
from subprocess import Popen

import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44


if __name__ == '__main__':
    # This code is needed for the Docker image to work without first entering the bash console.
    cmd = shlex.split("Xvfb :99 -screen 0 640x480x24")
    virtual_display_process = Popen(cmd)

    if (return_code := virtual_display_process.poll()) is not None:
        raise RuntimeError(f"Code {return_code}, Could not start virtual display process.")

    # Code from https://moderngl.readthedocs.io/en/5.8.2/techniques/headless_ubuntu_18_server.html
    ctx = moderngl.create_context(standalone=True)

    prog = ctx.program(vertex_shader="""
        #version 330
        uniform mat4 model;
        in vec2 in_vert;
        in vec3 in_color;
        out vec3 color;
        void main() {
            gl_Position = model * vec4(in_vert, 0.0, 1.0);
            color = in_color;
        }
        """,
                       fragment_shader="""
        #version 330
        in vec3 color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color, 1.0);
        }
    """)

    vertices = np.array([
        -0.6, -0.6,
        1.0, 0.0, 0.0,
        0.6, -0.6,
        0.0, 1.0, 0.0,
        0.0, 0.6,
        0.0, 0.0, 1.0,
    ], dtype='f4')

    vbo = ctx.buffer(vertices)
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_color')
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((512, 512), 4)])

    fbo.use()
    ctx.clear()
    prog['model'].write(Matrix44.from_eulers((0.0, 0.1, 0.0), dtype='f4'))
    vao.render(moderngl.TRIANGLES)

    data = fbo.read(components=3)
    image = Image.frombytes('RGB', fbo.size, data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save('output.png')

    print(f"Saved render to 'output.png'.")

    virtual_display_process.terminate()