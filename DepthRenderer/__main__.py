import os
from pathlib import Path

import cv2
import numpy as np
import plac
from PIL import Image

from .animation import RotateXYBounce, Compose, Translate, RotateAxisBounce
from .render import Camera, ShaderProgram, MeshRenderer, Texture, Mesh
from .utils import log, get_translation_matrix, get_scale_matrix, \
    load_depth, DelayedTask, RecurringTask, AsyncImageWriter, AsyncVideoWriter, OneTimeTask, Axis, load_colour


def resize(image, size):
    height, width = size[:2]
    resized_image = Image.fromarray(image)
    resized_image = resized_image.resize((width, height), Image.ANTIALIAS)

    return np.asarray(resized_image)


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
    fps=plac.Annotation(
        help='The target frames per second at which to render.',
        kind='option',
        type=float
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
def main(image_path="samples/00000_colors.png", depth_path="samples/00000_depth.png",
         fps=60, mesh_density=8, output_path='frames'):
    """
    Render a colour/depth image pair on a grid mesh in OpenGL using the depth map to displace vertices on the mesh.

    :param image_path: The path to the colour image.
    :param depth_path: The path to depth map corresponding to the colour image.
    :param fps: The target frames per second at which to render.
    :param mesh_density: How fine the generated mesh should be. Increasing this value by one roughly quadruples the
        number of vertices.
    :param output_path: The path to save any output frames to.
    """
    colour = load_colour(image_path)
    depth = load_depth(depth_path)
    depth = resize(depth, colour.shape)

    texture = Texture(colour)
    mesh = Mesh.from_depth_map(texture, density=mesh_density, depth_map=depth)

    camera_position = get_translation_matrix(dz=-100)
    camera = Camera(window_size=tuple(reversed(colour.shape[:2])), fov_y=180)

    # TODO: Make shader source paths configurable.
    default_shader = ShaderProgram(vertex_shader_path='DepthRenderer/shaders/shader.vert',
                                   fragment_shader_path='DepthRenderer/shaders/shader.frag')
    debug_shader = ShaderProgram(vertex_shader_path='DepthRenderer/shaders/debug_shader.vert',
                                 fragment_shader_path='DepthRenderer/shaders/debug_shader.frag')

    renderer = MeshRenderer(mesh,
                            default_shader_program=default_shader,
                            debug_shader_program=debug_shader,
                            fps=fps, camera=camera)

    mesh.transform = get_scale_matrix(1.0) @ mesh.transform
    log(f"Model: \n{mesh.transform}")

    camera.view = camera_position @ camera.view
    log(f"View: \n{camera.view}")

    log(f"Projection: \n{camera.projection}")

    os.makedirs(output_path, exist_ok=True)

    animation_speed = 0.15

    anim = RotateXYBounce(np.deg2rad(2.5), offset=0.5, speed=-animation_speed)

    writer = AsyncImageWriter(renderer.frame_buffer_shape,
                              num_workers=1)

    render_one_frame = DelayedTask(OneTimeTask(writer.write), delay=10)

    video_writer = AsyncVideoWriter(os.path.join(output_path, f"{Path(image_path).name}.avi"),
                                    size=renderer.frame_buffer_shape,
                                    fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                                    fps=fps,
                                    num_workers=1)

    # Need to delay writing by one frame to ensure the window size is settled (#TODO: Find out why it's changing one after init)
    initial_delay = 3

    write_frame = DelayedTask(RecurringTask(video_writer.write), delay=initial_delay)

    def update_callback(delta):
        anim.update(delta)

        camera.view = camera_position @ anim.transform

        # TODO: Fix `[mpeg4 @ 000001a3c8fa7dc0] Invalid pts (1) <= last (1)` warnings/errors.
        # TODO: Fix intermittent SIGSEV and other errors.
        render_one_frame(renderer.frame_buffer, os.path.join(output_path, 'sample_frame.png'))
        write_frame(renderer.frame_buffer)

    def on_exit_callback():
        video_writer.cleanup()
        writer.cleanup()
        texture.cleanup()
        mesh.cleanup()
        default_shader.cleanup()
        debug_shader.cleanup()
        renderer.cleanup()

    renderer.on_update = update_callback
    renderer.on_exit = on_exit_callback

    renderer.run()


if __name__ == '__main__':
    plac.call(main)
