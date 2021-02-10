import datetime
import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import plac
from PIL import Image

from DepthRenderer.animation import RotateXYBounce, Compose, Translate, RotateAxisBounce
from DepthRenderer.render import Camera, ShaderProgram, MeshRenderer, Texture, Mesh
from DepthRenderer.utils import log, get_translation_matrix, get_scale_matrix, \
    load_depth, DelayedTask, RecurringTask, AsyncImageWriter, AsyncVideoWriter, OneTimeTask, Axis, load_colour, \
    read_frame_buffer, process_frame_numpy, FrameTimer


def resize(image, size):
    height, width = size[:2]
    resized_image = Image.fromarray(image)
    resized_image = resized_image.resize((width, height), Image.ANTIALIAS)

    return np.asarray(resized_image)


def create_mosiac_cmd(output_path, video_sources, output_shape, num_rows, num_cols=None):
    if num_cols is None:
        num_cols = len(video_sources) // num_rows
    else:
        assert num_cols * num_rows == len(video_sources)

    input_files = ' '.join(map(lambda src_path: f"-i {src_path}", video_sources))

    output_width, output_height = output_shape
    height = output_height // num_rows
    width = output_width // num_cols

    filter_pieces = [f"nullsrc=size={output_width:d}x{output_height:d} [base]"]

    i = 0

    for row in range(num_rows):
        for col in range(num_cols):
            tag = f"{row}x{col}"
            filter_pieces.append(f"[{i}:v] setpts=PTS-STARTPTS, scale={width:d}x{height:d} [{tag}]")

            i += 1

    i = 1
    prev_tag = "base"

    for row in range(num_rows):
        for col in range(num_cols):
            tag = f"{row}x{col}"
            next_tag = f"tmp{i}"

            x_position = col * width
            y_position = row * height

            filter_piece = f"[{prev_tag}][{tag}] overlay=shortest=1:x={x_position:d}:y={y_position:d}"

            if i < len(video_sources):
                filter_piece = f"{filter_piece} [{next_tag}]"

            filter_pieces.append(
                filter_piece
            )

            prev_tag = next_tag

            i += 1

    complex_filter = f"-filter_complex \"{'; '.join(filter_pieces)}\""

    encoding_format = "-c:v libx264"

    return f"ffmpeg {input_files} {complex_filter} {encoding_format} {output_path}"


def create_mosaic_video(video_sources, output_path, name, source_shape, max_width=1920):
    os.makedirs(output_path, exist_ok=True)
    # TODO: Calculate num_rows and num_cols automatically based on the number of sources
    num_rows = 2
    num_cols = len(video_sources) // num_rows

    output_width, output_height = source_shape[1] * num_cols, source_shape[0] * num_rows
    aspect_ratio = output_width / output_height
    output_width = min(max_width, output_width)
    output_height = int(output_width / aspect_ratio)

    cmd = create_mosiac_cmd(os.path.join(output_path, f"{name}.avi"), video_sources,
                            output_shape=(output_width, output_height), num_rows=num_rows, num_cols=num_cols)

    cmd = f"{cmd} -y"
    log(cmd)
    subprocess.run(cmd, shell=True)


def create_concat_video(video_sources, output_path, name):
    os.makedirs(output_path, exist_ok=True)
    input_list = list(map(lambda path: f"file '{path}'\n", video_sources))

    tmp_file_path = os.path.join(output_path, 'tmp.txt')

    with open(tmp_file_path, 'w') as f:
        f.writelines(input_list)

    output_path = os.path.join(output_path, f"{name}.avi")
    cmd = f"ffmpeg -f concat -safe 0 -i {tmp_file_path} -c:v libx264 {output_path} -y"

    log(cmd)
    subprocess.run(cmd, shell=True)

    os.remove(tmp_file_path)


def create_paired_videos(video_sources, output_path, name, model_names):
    output_path = os.path.join(output_path, name)
    os.makedirs(output_path, exist_ok=True)
    # Copy since we are going to modify these lists.
    video_sources = video_sources.copy()
    model_names = model_names.copy()

    for i, model_name in enumerate(model_names):
        if model_name == "ground_truth":
            ground_truth_index = i
            break
    else:
        raise RuntimeError("Cannot create paired videos with ground truth file present. Make sure the path to the "
                           "ground truth video is included and starts with 'ground_truth'.")

    ground_truth_src = video_sources[i]

    assert Path(ground_truth_src).name.startswith('ground_truth')
    del video_sources[ground_truth_index]
    del model_names[ground_truth_index]

    assert len(video_sources) == len(model_names)

    for model_name, video_source in zip(model_names, video_sources):
        input_filename = Path(video_source).stem
        input_filename = input_filename.replace(f"{model_name}-", "")
        output_filename = f"ground_truth-{model_name}-{input_filename}.avi"
        paired_video_path = os.path.join(output_path, output_filename)
        cmd = f"ffmpeg -i {ground_truth_src} -i {video_source} -filter_complex hstack {paired_video_path} -y"

        log(cmd)
        subprocess.run(cmd, shell=True)


@plac.annotations(
    image_path=plac.Annotation(
        help='The path to the colour image.',
        kind='positional',
        type=Path
    ),
    depth_maps_path=plac.Annotation(
        help='The path to a folder of folders where each subfolder contains at least one depth map that shares '
             'the same file name as the colour image.',
        kind='positional',
        type=Path
    ),
    fps=plac.Annotation(
        help='The target frames per second at which to render.',
        kind='option',
        type=float
    ),
    mesh_density=plac.Annotation(
        help='How fine the generated mesh should be. Increasing this value by one roughly quadruples '
             'the number of vertices.',
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
def main(image_path, depth_maps_path, fps=60, mesh_density=8, displacement_factor=4.0, output_path='output'):
    """
    Render a colour/depth image pair on a grid mesh in OpenGL using the depth map to displace vertices on the mesh.

    :param image_path: The path to the colour image.
    :param depth_maps_path: The path to a folder of folders where each subfolder contains at least one depth map that
        shares the same file name as the colour image.
    :param fps: The target frames per second at which to render.
    :param mesh_density: How fine the generated mesh should be. Increasing this value by one roughly quadruples the
        number of vertices.
    :param displacement_factor: A multiplicative constant applied to the normalised depth values.
        For example, with a displacement factor of 10.0, a normalised depth value of 0.2 becomes 10.0 * 0.2 = 2.0.
    :param output_path: The path to save any output frames to.
    """
    model_names = list(
        sorted(
            filter(
                lambda path: os.path.isdir(os.path.join(depth_maps_path, path)),
                os.listdir(depth_maps_path)
            )
        )
    )

    image_filename = Path(image_path).name
    image_name = Path(image_path).stem

    video_output_path = os.path.join(output_path, 'single_videos', image_name)
    os.makedirs(video_output_path, exist_ok=True)

    colour = load_colour(image_path)

    camera_position = get_translation_matrix(dz=-10)
    camera = Camera(window_size=tuple(reversed(colour.shape[:2])), fov_y=18)

    default_shader = ShaderProgram(vertex_shader_path='DepthRenderer/shaders/shader.vert',
                                   fragment_shader_path='DepthRenderer/shaders/shader.frag')
    debug_shader = ShaderProgram(vertex_shader_path='DepthRenderer/shaders/debug_shader.vert',
                                 fragment_shader_path='DepthRenderer/shaders/debug_shader.frag')

    renderer = MeshRenderer(default_shader_program=default_shader,
                            debug_shader_program=debug_shader,
                            fps=fps, camera=camera,
                            unlimited_frame_works=True)

    class Context:
        def __init__(self, model_name, image_path, depth_map_path):
            self.model_name = model_name
            self.image_path = image_path
            self.depth_map_path = depth_map_path

            self.texture: Optional[Texture] = None
            self.mesh: Optional[Mesh] = None
            self.video_writer: Optional[AsyncVideoWriter] = None

        def load(self):
            colour = load_colour(self.image_path)
            depth = load_depth(self.depth_map_path)
            depth = resize(depth, colour.shape)

            self.texture = Texture(colour)
            self.mesh = Mesh.from_depth_map(self.texture, depth, density=mesh_density)
            self.mesh.vertices[:, 2] *= displacement_factor

            self.video_writer = AsyncVideoWriter(
                os.path.join(video_output_path, f"{self.model_name}.avi"),
                size=renderer.frame_buffer_shape, fps=fps)

        def cleanup(self):
            if self.video_writer is not None:
                # NOTE: This may be slow depending on how many frames are waiting to be written to disk.
                self.video_writer.cleanup()

            if self.texture is not None:
                self.texture.cleanup()

            if self.mesh is not None:
                self.mesh.cleanup()

    contexts = []
    video_sources = []

    for model_name in model_names:
        depth_map_path = os.path.join(depth_maps_path, model_name, image_filename)
        contexts.append(Context(model_name, image_path, depth_map_path))
        video_sources.append(os.path.join(video_output_path, f"{model_name}.avi"))

    class ContextSwitcher:
        def __init__(self, contexts):
            self.contexts = contexts

            self.current_context: Optional[Context] = None
            self.context_iterator = iter(contexts)

        def next_context(self):
            if self.current_context is not None:
                self.current_context.cleanup()

            context = next(self.context_iterator)
            context.load()

            self.current_context = context

            return self.current_context

        def cleanup(self):
            for context in self.contexts:
                context.cleanup()

    context_switcher = ContextSwitcher(contexts)

    os.makedirs(output_path, exist_ok=True)

    rotation_angle = 2.5
    loops_per_second = 0.5 / rotation_angle

    camera_animation = Compose([
        RotateAxisBounce(np.deg2rad(rotation_angle), axis=Axis.Y, offset=0.5, speed=-loops_per_second),
        RotateAxisBounce(np.deg2rad(rotation_angle / 5.0), axis=Axis.X, offset=0.5, speed=-loops_per_second),
        Translate(distance=0.30, speed=loops_per_second),
        Translate(distance=0.15, axis=Axis.Y, offset=0.25, speed=loops_per_second)
    ])

    initial_delay = 3
    animation_length_in_frames = fps / loops_per_second
    image_writer = AsyncImageWriter()

    def write_image_func(image, path, *args, **kwargs):
        os.makedirs(Path(path).parent, exist_ok=True)

        image_writer.write(image, path, *args, **kwargs)

    write_image = RecurringTask(write_image_func, frequency=fps)

    def write_frame_func():
        frame = renderer.get_frame()
        current_context = context_switcher.current_context

        if current_context is not None:
            image_output_path = os.path.join(output_path, 'frames', current_context.model_name,
                                             f"{write_image.call_count:06d}.png")

            write_image(frame, image_output_path)
            video_writer = current_context.video_writer

            if video_writer is not None:
                video_writer.write(frame)

    write_frame = DelayedTask(write_frame_func, delay=initial_delay)

    def next_mesh_func():
        current_context = context_switcher.next_context()
        current_context.image_output_path = os.path.join(output_path, 'frames', current_context.model_name)

        renderer.mesh = context_switcher.current_context.mesh
        camera_animation.reset()
        write_frame.reset()
        write_image.reset()

    next_mesh = RecurringTask(next_mesh_func, frequency=animation_length_in_frames + initial_delay + 1)

    class Flag:
        def __init__(self):
            self._is_set = False

        @property
        def is_set(self):
            return self._is_set

        @is_set.setter
        def is_set(self, value):
            self._is_set = bool(value)

        def set(self):
            self.is_set = True

        def unset(self):
            self.is_set = False

        def __bool__(self):
            return self.is_set

    reached_finished = Flag()

    def update_callback(delta):
        try:
            next_mesh()
        except StopIteration:
            reached_finished.set()
            renderer.close()

        camera_animation.update(delta)
        camera.view = camera_position @ camera_animation.transform
        write_frame()

    def on_exit_callback():
        context_switcher.cleanup()
        default_shader.cleanup()
        debug_shader.cleanup()
        renderer.cleanup()

        if reached_finished:
            create_mosaic_video(video_sources, os.path.join(output_path, 'mosaic'), image_name, colour.shape[:2])
            create_concat_video(video_sources, os.path.join(output_path, 'concat'), image_name)
            create_paired_videos(video_sources, os.path.join(output_path, 'paired'), image_name, model_names)

    renderer.on_update = update_callback
    renderer.on_exit = on_exit_callback

    renderer.run()


if __name__ == '__main__':
    plac.call(main)
