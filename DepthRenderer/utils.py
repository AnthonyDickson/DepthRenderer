import datetime
import enum
import time
from multiprocessing.pool import ThreadPool
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps


def log(message):
    """
    Print a message to stdout with a timestamp.
    :param message: The message to print.
    """
    print(f"[{datetime.datetime.now()}] {message}")


def get_perspective_matrix(fov_y, aspect_ratio, near=0.01, far=1000.0):
    """
    Get a 4x4 perspective matrix.

    :param fov_y: The field of view angle (degrees) visible along the y-axis.
    :param aspect_ratio: The ratio of the width/height of the viewport.
    :param near: The z-coordinate for the near plane.
    :param far: The z-coordinate for the far plane.
    :return: The perspective matrix.
    """
    return np.array(
        [[fov_y / aspect_ratio, 0, 0, 0],
         [0, fov_y, 0, 0],
         [0, 0, (far + near) / (near - far), (2 * near * far) / (near - far)],
         [0, 0, -1, 0]],
        dtype=np.float32
    )


class Axis(enum.Enum):
    """
    Enumeration of the axes in a 3-dimensional coordinate system (x, y, z).
    """
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()


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
    """
    Load an image from disk.

    The image is flipped vertically to ensure it is displayed the right way up in OpenGL.

    :param fp: The path to the image file or file object.
    :return: The loaded image.
    """
    img = Image.open(fp)
    img_data = np.asarray(img)

    # Images need to flipped vertically to be displayed the right way up.
    img_data = np.flip(img_data, axis=0)

    return img_data


def load_colour(fp, should_mask=False, mask_white=True):
    """
    Load a colour image.

    :param fp: The path to the image file or file object.
    :param should_mask: Whether certain regions of the image should be masked based on their colour.
    :param mask_white: Whether white pixels (mask_white=True) or black pixels (mask_white=False) should be masked.
    :return: The loaded image in RGBA format.
    """
    colour_image = load_image(fp)

    H, W, C = colour_image.shape

    if C == 3:
        colour_image = np.concatenate(
            (colour_image, colour_image.max() * np.ones(shape=(H, W, 1), dtype=colour_image.dtype)), axis=2)

    if should_mask:
        mask_colour = [255, 255, 255] if mask_white else [0, 0, 0]
        should_mask = np.all(colour_image[:, :, :3] == mask_colour, axis=2)
        colour_image[should_mask, 3] = 0

    return colour_image


def load_depth(fp):
    """
    Load a depth map from disk and normalises depth values.

    :param fp: The file pointer (string path or file object) for the depth map file.
    :return: The loaded depth map as a numpy array.
    """
    depth_map = load_image(fp)

    if len(depth_map.shape) == 2:
        depth_map = np.expand_dims(depth_map, axis=2)

    depth_map = np.concatenate(3 * [depth_map], axis=2)

    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (255 * depth_map).astype(np.uint8)

    return depth_map


def flatten_arrays(arrays):
    """
    Flatten multiple multidimensional arrays.

    :param arrays: The arrays to flatten.
    :return: A tuple of the flattened arrays.
    """
    return tuple(map(np.ravel, arrays))


def interweave_arrays(arrays):
    """
    Interweaves multiple arrays.

    >>> interweave_arrays([[1, 3, 5], [2, 4, 6]])
    [1, 2, 3, 4, 5, 6]
    """
    total_num_elements = sum(map(lambda array: array.size, arrays))
    dtype = arrays[0].dtype

    combined_array = np.empty(total_num_elements, dtype=dtype)

    for i, array in enumerate(arrays):
        combined_array[i::len(arrays)] = array

    return combined_array


class Task:
    """
    Encapsulates a callable object.

    >>> def say_hello():
    >>>     print("Hello, world!")
    >>>
    >>> task = Task(say_hello)
    >>> task()
    Hello, world!
    >>> task()
    Hello, world!
    """

    def __init__(self, task):
        self.task = task
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        return self.task(*args, **kwargs)

    def reset(self):
        """
        Clear the state of the task.
        """
        self.call_count = 0


class DelayedTask(Task):
    """
    Encapsulates a task that executes after a given delay.

    >>> def say_hello():
    >>>     print("Hello, world!")
    >>>
    >>> task = DelayedTask(say_hello, delay=1)
    >>> task()
    >>> task()
    Hello, world!
    """

    def __init__(self, task, delay=0):
        """
        :param task: The task to execute.
        :param delay: The number of calls to this object before `task` is executed.
        """
        super().__init__(task)

        self.delay = delay

    def __call__(self, *args, **kwargs):
        self.call_count += 1

        if self.call_count > self.delay:
            return super().__call__(*args, **kwargs)


class OneTimeTask(Task):
    """
    Encapsulates a task that should be done one time only.

    >>> def say_hello():
    >>>     print("Hello, world!")
    >>>
    >>> task = OneTimeTask(say_hello)
    >>> task()
    Hello, world!
    >>> task()
    """

    def __init__(self, task):
        super().__init__(task)

        self.is_done = False
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1

        if not self.is_done:
            self.is_done = True
            return super().__call__(*args, **kwargs)

    def reset(self):
        super().reset()

        self.is_done = False


class RecurringTask(Task):
    """
    Encapsulates a repeatable task.


    >>> def say_hello():
    >>>     print("Hello, world!")
    >>>
    >>> task = RecurringTask(say_hello, frequency=2)
    >>> task()
    Hello, world!
    >>> task()
    >>> task()
    Hello, world!
    >>> task()
    """

    def __init__(self, task, frequency=1):
        """
        :param task: The task to execute.
        :param frequency: How often the task should be executed where 1 = everytime, 2 =  every second call,
            3 = every third call etc...
        """
        super().__init__(task)

        assert frequency > 0, f"Frequency must be a positive integer, got {frequency}."

        self.frequency = frequency

    def __call__(self, *args, **kwargs):
        result = None

        if self.call_count % self.frequency == 0:
            result = super().__call__(*args, **kwargs)

        self.call_count += 1
        return result


def read_frame_buffer(frame_buffer, size, mode='RGBA'):
    """
    Read a frame buffer as a PIL.Image object.

    :param frame_buffer: The frame buffer to read.
    :param size: The width and height of the frame.
    :param mode: The mode (e.g. RGB, RGBA) of the frame buffer.

    :return: A PIL.Image object.
    """
    return Image.frombytes(mode, size, data=frame_buffer)


def process_frame_numpy(frame_from_buffer):
    """
    Process a PIL.Image formatted frame buffer as a numpy array.

    :param frame_from_buffer: The frame loaded with `read_frame_buffer(...)`.

    :return: The frame as a numpy array.
    """
    return np.flip(np.asarray(frame_from_buffer), axis=0)


def process_frame_pillow(frame_from_buffer):
    """
    Process a PIL.Image formatted frame buffer.

    :param frame_from_buffer: The frame loaded with `read_frame_buffer(...)`.

    :return: The frame as a PIL.Image object.
    """
    return ImageOps.flip(frame_from_buffer)


class ImageWriter:
    """
    Handles writing a given frame buffer to disk.
    """

    def write(self, frame, path, file_format='PNG'):
        """
        Write a frame to disk.

        :param frame: The frame to write.
        :param path: The path to save the image.
        :param file_format: The format to save the image in.
        """
        self._worker(frame, path, file_format)

    @staticmethod
    def _worker(frame, path, file_format):
        """
        Worker function used to write a frame to file.

        :param frame: The frame buffer copied from the GPU.
        :param path: The path to save the image.
        :param file_format: The format to save the image in.
        """
        image = process_frame_pillow(frame)

        image.save(path, file_format)


class AsyncImageWriter(ImageWriter):
    """
    Handles writing a given frame buffer to disk asynchronously on a separate thread.
    """

    def __init__(self, num_workers=4):
        """
        :param num_workers: The number of threads to use.
        """
        super().__init__()

        self.pool = ThreadPool(processes=num_workers)

    def write(self, frame, path, file_format='PNG'):
        """
        Write a frame to file.

        :param frame: The frame to write.
        :param path: The path to save the image.
        :param file_format: The format to save the image in.
        """
        self.pool.apply_async(self._worker, (frame, path, file_format))

    def cleanup(self):
        """
        Finish writing any queued frames and release any used resources.
        """
        self.pool.close()
        self.pool.join()


class VideoWriter:
    """
    Handles writing a series of frames to a video file.
    """

    def __init__(self, path, size, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=24):
        """
        :param path: The path to save the video to.
        :param size: The width and height of frames to be written.
        :param fourcc: The four character code of the video format to use.
        :param fps: The frame rate to encode the video at.
        """
        self.path = path
        self.size = size
        self.fourcc = fourcc
        self.fps = fps
        self.writer = cv2.VideoWriter(self.path, self.fourcc, self.fps, self.size)

    def write(self, frame):
        """
        Add a frame to the video.

        :param frame: The frame to add to the video.
        """
        self._worker(self.writer, frame)

    @staticmethod
    def _worker(writer, frame):
        """
        Worker function used to write a frame to a video.

        :param writer: The OpenCV VideoWriter object to use for writing the video.
        :param frame: The frame to write.
        """
        frame = process_frame_numpy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        writer.write(frame)

    def cleanup(self):
        """
        Finish writing the video and free any used resources.
        """
        if self.writer:
            self.writer.release()


class AsyncVideoWriter(VideoWriter):
    """
    Handles writing a series of frames to a video file asynchronously on a separate thread.
    """

    def __init__(self, path, size, fourcc=cv2.VideoWriter.fourcc(*'DIVX'), fps=24):
        """
        :param path: The path to save the video to.
        :param size: The width and height of frames to be written.
        :param fourcc: The four character code of the video format to use.
        :param fps: The frame rate to encode the video at.
        """
        super().__init__(path, size, fourcc, fps)

        # Have to use a thread pool instead of process pool since VideoWriter objects cannot be pickled.
        # Have to use thread pool size of one to avoid various errors.
        self.pool = ThreadPool(processes=1)

    def write(self, frame):
        """
        Add a frame to the video.

        :param frame: The frame to add to the video.
        """
        self.pool.apply_async(self._worker, (self.writer, frame))

    def cleanup(self):
        """
        Finish writing the video and free any used resources.
        """
        self.pool.close()
        self.pool.join()

        super().cleanup()


class FrameTimer:
    def __init__(self):
        self.last_frame_time = time.time()
        self.delta = 0.0
        self.elapsed = 0.0

    def reset(self):
        self.last_frame_time = time.time()
        self.delta = 0.0
        self.elapsed = 0.0

    def update(self):
        now = time.time()
        self.delta = now - self.last_frame_time
        self.elapsed += self.delta
        self.last_frame_time = now

def perlin(width, height, scale=5, seed=None):
    X = np.linspace(0, scale, width, endpoint=False)
    Y = np.linspace(0, scale, height, endpoint=False)
    x, y = np.meshgrid(X, Y)

    if seed is not None:
        np.random.seed(seed)

    # permutation table
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()

    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)

    # internal coordinates
    xf = x - xi
    yf = y - yi

    # fade factors
    def fade(t):
        "6t^5 - 15t^4 + 10t^3"
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    u = fade(xf)
    v = fade(yf)

    # noise components

    def gradient(h, x, y):
        "grad converts h to the right gradient vector and return the dot product with (x,y)"
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[:, :, 0] * x + g[:, :, 1] * y

    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)

    def lerp(a, b, x):
        "linear interpolation"
        return a + x * (b - a)

    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)

    return lerp(x1, x2, v)