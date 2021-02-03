import datetime
import enum
import multiprocessing
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
    img = Image.open(fp)
    img_data = np.asarray(img)

    if img.mode == 'RGBA':
        img_data = np.delete(img_data, -1, -1)  # Drop the alpha channel (the last in the channels dimension)

    # Images need to flipped vertically to be displayed the right way up.
    img = np.flip(img_data, axis=0)

    return img


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


def read_frame_buffer(frame_buffer, width, height, mode='RGBA'):
    return Image.frombuffer(mode, size=(width, height), data=frame_buffer)


def process_frame_numpy(frame_from_buffer):
    return np.flip(np.asarray(frame_from_buffer), axis=0)


def process_frame_pillow(frame_from_buffer):
    return ImageOps.flip(frame_from_buffer)


class AsyncImageWriter:
    def __init__(self, size, num_workers=4):
        self.pool = multiprocessing.Pool(processes=num_workers)

        self.size = size
        self.width, self.height = size

    def join(self):
        self.pool.join()
        self.pool.close()

    def write(self, frame_buffer, path, file_format='PNG'):
        self.pool.apply_async(self._worker, (frame_buffer, self.size, path, file_format))

    @staticmethod
    def _worker(frame_buffer, size, path, file_format):
        width, height = size
        image = process_frame_pillow(read_frame_buffer(frame_buffer, width, height))

        image.save(path, file_format)


class VideoWriter:
    def __init__(self, path, size, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=24):
        self.path = path
        self.size = size
        self.fourcc = fourcc
        self.fps = fps
        self.writer = cv2.VideoWriter(self.path, self.fourcc, self.fps, self.size)

    def write(self, frame_buffer):
        if frame_buffer is not None:
            frame = process_frame_numpy(read_frame_buffer(frame_buffer, *self.size))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()


class AsyncVideoWriter(VideoWriter):
    def __init__(self, path, size, fourcc=cv2.VideoWriter.fourcc(*'DIVX'), fps=24, num_workers=4):
        super().__init__(path, size, fourcc, fps)

        # Have to use a thread pool instead of process pool since VideoWriter objects cannot be pickled.
        self.pool = multiprocessing.pool.ThreadPool(processes=num_workers)

    def release(self):
        self.pool.join()
        self.pool.close()

        super().release()

    def write(self, frame_buffer):
        self.pool.apply_async(self._worker, (self.writer, frame_buffer, self.size))

    @staticmethod
    def _worker(writer, frame_buffer, size):
        if frame_buffer is not None:
            width, height = size

            frame = process_frame_numpy(read_frame_buffer(frame_buffer, width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            writer.write(frame)