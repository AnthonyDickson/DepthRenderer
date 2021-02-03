import multiprocessing
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import plac
from PIL import Image

from animation import RotateXYBounce
from render import Camera, ShaderProgram, QuadRenderer, Texture, Mesh
from utils import log, get_translation_matrix, get_scale_matrix, \
    load_image, load_depth


class Task:
    def __init__(self, task):
        self.task = task
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        return self.task(*args, **kwargs)


class DelayedTask(Task):
    def __init__(self, task, delay=0):
        super().__init__(task)

        self.delay = delay

    def __call__(self, *args, **kwargs):
        self.call_count += 1

        if self.call_count > self.delay:
            return super().__call__(*args, **kwargs)


class OneTimeTask(Task):
    def __init__(self, task):
        super().__init__(task)

        self.is_done = False
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1

        if not self.is_done:
            self.is_done = True
            return super().__call__(*args, **kwargs)


class RecurringTask(Task):
    def __init__(self, task, frequency=1):
        super().__init__(task)
        self.frequency = frequency

    def __call__(self, *args, **kwargs):
        self.call_count += 1

        if self.call_count % self.frequency == 0:
            return super().__call__(*args, **kwargs)


class AsyncImageWriter:
    def __init__(self, num_workers=4):
        self.pool = multiprocessing.Pool(processes=num_workers)

    def join(self):
        self.pool.join()
        self.pool.close()

    def write(self, image, path, file_format='PNG'):
        self.pool.apply_async(self._worker, (image, path, file_format))

    @staticmethod
    def _worker(image, path, file_format):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image.save(path, file_format)


class VideoWriter:
    def __init__(self, path, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=24):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None
        self.shape: Optional[Tuple[int, int]] = None

    def write(self, frame):
        frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

        if self.writer is None:
            self.shape = tuple(reversed(frame.shape[:2]))
            self.writer = cv2.VideoWriter(self.path, self.fourcc, self.fps, self.shape)

        self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()


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
def main(image_path="brick_wall.jpg", depth_path="depth.png",
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
    colour = load_image(image_path)
    depth = load_depth(depth_path)

    texture = Texture(colour)
    mesh = Mesh.from_depth_map(texture, density=mesh_density, depth_map=depth)

    camera = Camera(window_size=tuple(reversed(colour.shape[:2])), fov_y=60)

    # TODO: Make shader source paths configurable.
    default_shader = ShaderProgram(vertex_shader_path='shader.vert', fragment_shader_path='shader.frag')
    debug_shader = ShaderProgram(vertex_shader_path='debug_shader.vert', fragment_shader_path='debug_shader.frag')

    renderer = QuadRenderer(mesh,
                            default_shader_program=default_shader,
                            debug_shader_program=debug_shader,
                            fps=fps, camera=camera)

    mesh.transform = get_scale_matrix(1.0) @ mesh.transform
    log(f"Model: \n{mesh.transform}")

    camera.view = get_translation_matrix(dz=-70) @ camera.view
    log(f"View: \n{camera.view}")

    log(f"Projection: \n{camera.projection}")

    os.makedirs(output_path, exist_ok=True)

    writer = AsyncImageWriter()

    render_image = DelayedTask(OneTimeTask(writer.write), delay=1)
    render_frames = RecurringTask(writer.write)

    # anim = RotateAxisBounce(np.deg2rad(10), axis=Axis.X)
    # anim = Translate()
    anim = RotateXYBounce(np.deg2rad(5))

    # anim = Compose([
    #     RotateAxisBounce(angle=np.deg2rad(15), speed=-1.0),
    #     Translate(distance=0.5),
    # ])

    video_writer = VideoWriter(os.path.join(output_path, f"{Path(image_path).name}.avi"),
                               cv2.VideoWriter_fourcc(*'DIVX'),
                               fps, )

    # Need to delay writing by one frame to ensure the window size is settled (#TODO: Find out why it's changing one after init)
    write_frame = DelayedTask(RecurringTask(video_writer.write), delay=1)

    def update_func(delta):
        anim.update(delta)

        mesh.transform = anim.transform

        # frame = renderer.get_frame()
        # write_frame(frame)
        # render_image(frame, os.path.join(output_path, 'sample_frame.png'))
        # render_frames(frame, os.path.join(output_path, f"frame_{render_frames.call_count:03d}.png"))

    renderer.update = update_func

    try:
        renderer.run()
    finally:
        video_writer.release()
        writer.join()
        texture.cleanup()
        mesh.cleanup()
        default_shader.cleanup()
        debug_shader.cleanup()
        renderer.cleanup()


if __name__ == '__main__':
    plac.call(main)
