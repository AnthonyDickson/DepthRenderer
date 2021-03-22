import os
from pathlib import Path

import cv2
import numpy as np
import plac
from PIL import Image

from .animation import RotateXYBounce, Compose, Translate, RotateAxisBounce
from .render import Camera, ShaderProgram, MeshRenderer, Texture, Mesh
from .utils import log, get_translation_matrix, get_scale_matrix, \
    load_depth, DelayedTask, RecurringTask, AsyncImageWriter, AsyncVideoWriter, OneTimeTask, Axis, load_colour, perlin


def resize(image, size):
    height, width = size[:2]
    resized_image = Image.fromarray(image)
    resized_image = resized_image.resize((width, height), Image.ANTIALIAS)

    return np.asarray(resized_image)


def overlay_noise(image, **perlin_kwargs):
    height, width = image.shape[:2]

    noise = perlin(width, height, **perlin_kwargs)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = 255 * noise
    noise = np.expand_dims(noise, -1)

    new_image = image.astype(np.float) + noise
    new_image = new_image / new_image.max()
    new_image = (255 * new_image).astype(np.uint8)

    return new_image


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
    displacement_factor=plac.Annotation(
        help='A multiplicative constant applied to the normalised depth values.',
        kind='option',
        type=float
    ),
    output_path=plac.Annotation(
        help='The path to save any output frames to.',
        kind='option',
        type=Path
    )
)
def main(image_path="samples/00000_colors.png", depth_path="samples/00000_depth.png",
         fps=60, mesh_density=8, displacement_factor=4.0, output_path='frames'):
    """
    Render a colour/depth image pair on a grid mesh in OpenGL using the depth map to displace vertices on the mesh.

    :param image_path: The path to the colour image.
    :param depth_path: The path to depth map corresponding to the colour image.
    :param fps: The target frames per second at which to render.
    :param mesh_density: How fine the generated mesh should be. Increasing this value by one roughly quadruples the
        number of vertices.
    :param displacement_factor: A multiplicative constant applied to the normalised depth values.
        For example, with a displacement factor of 10.0, a normalised depth value of 0.2 becomes 10.0 * 0.2 = 2.0.
    :param output_path: The path to save any output frames to.
    """
    colour = load_colour(image_path)
    depth = load_depth(depth_path)
    depth = resize(depth, colour.shape)
    # depth = overlay_noise(overlay_noise(overlay_noise(depth, scale=32, seed=0), scale=16, seed=0), scale=8, seed=0)

    texture = Texture(colour)
    mesh = Mesh.from_depth_map(texture, density=mesh_density, depth_map=depth)
    mesh.vertices[:, 2] *= displacement_factor

    camera_position = get_translation_matrix(dz=-10)
    camera = Camera(window_size=tuple(reversed(colour.shape[:2])), fov_y=18)

    # TODO: Make shader source paths configurable.
    default_shader = ShaderProgram(vertex_shader_path='DepthRenderer/shaders/shader.vert',
                                   fragment_shader_path='DepthRenderer/shaders/shader.frag')
    debug_shader = ShaderProgram(vertex_shader_path='DepthRenderer/shaders/debug_shader.vert',
                                 fragment_shader_path='DepthRenderer/shaders/debug_shader.frag')

    renderer = MeshRenderer(default_shader_program=default_shader,
                            debug_shader_program=debug_shader,
                            fps=fps, camera=camera,
                            unlimited_frame_works=True
                            )

    renderer.mesh = mesh
    mesh.transform = get_scale_matrix(1.0) @ mesh.transform
    log(f"Model: \n{mesh.transform}")

    camera.view = camera_position @ camera.view
    log(f"View: \n{camera.view}")

    log(f"Projection: \n{camera.projection}")

    os.makedirs(output_path, exist_ok=True)

    animation_length_secs = 5
    animation_speed = 1.0 / animation_length_secs

    anim = Compose([
        RotateAxisBounce(np.deg2rad(2.5), axis=Axis.Y, offset=0.5, speed=-animation_speed),
        RotateAxisBounce(np.deg2rad(.5), axis=Axis.X, offset=0.5, speed=-animation_speed),
        Translate(distance=0.30, speed=animation_speed),
        Translate(distance=0.15, axis=Axis.Y, offset=0.25, speed=animation_speed)
    ])

    writer = AsyncImageWriter(num_workers=1)

    render_one_frame = DelayedTask(OneTimeTask(writer.write), delay=10)

    video_writer = AsyncVideoWriter(os.path.join(output_path, f"{Path(image_path).name}.avi"),
                                    size=renderer.frame_buffer_shape, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=fps)

    # Need to delay writing by one frame to ensure the window size is settled (#TODO: Find out why it's changing one after init)
    initial_delay = 3

    write_frame = DelayedTask(RecurringTask(video_writer.write), delay=initial_delay)

    close_window = DelayedTask(renderer.close, delay=3 * animation_length_secs * fps + initial_delay)

    def update_callback(delta):
        anim.update(delta)

        # TODO: Fix bug that ignores panning from mouse input. Need to store transforms to the view from mouse inputs
        #  separately instead of in Camera.view so that they are not overridden here.
        camera.view = camera_position @ anim.transform

        frame = renderer.get_frame()

        if frame:
            render_one_frame(frame, os.path.join(output_path, 'sample_frame.png'))
            write_frame(frame)

            close_window()

    def exit_callback():
        video_writer.cleanup()
        writer.cleanup()
        texture.cleanup()
        mesh.cleanup()
        default_shader.cleanup()
        debug_shader.cleanup()
        renderer.cleanup()

    renderer.on_update = update_callback
    renderer.on_exit = exit_callback

    log("Starting main loop...")
    renderer.run()
    log("Exited main loop.")


if __name__ == '__main__':
    plac.call(main)
