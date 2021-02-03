import numpy as np

from .utils import get_rotation_matrix, get_translation_matrix, Axis


class Animation:
    def __init__(self):
        self.elapsed = 0.0
        self._transform = np.eye(4, dtype=np.float32)

    def update(self, delta):
        self.elapsed += delta

    def apply(self, other):
        return other @ self._transform

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value


class RotateAxisBounce(Animation):
    def __init__(self, angle=np.pi / 2, axis=Axis.Y, speed=1.0):
        super().__init__()

        self.angle = angle
        self.axis = axis
        self.speed = speed

    def update(self, delta):
        super().update(delta)

        new_angle = np.sin(self.speed * self.elapsed) * self.angle
        self.transform = get_rotation_matrix(new_angle, axis=self.axis)


class RotateXYBounce(Animation):
    def __init__(self, angle=np.pi / 2, speed=1.0):
        super().__init__()

        self.angle = angle
        self.speed = speed

    def update(self, delta):
        super().update(delta)

        y_axis_rotation = np.sin(self.speed * self.elapsed) * self.angle
        x_axis_rotation = np.cos(self.speed * self.elapsed) * self.angle

        self.transform = get_rotation_matrix(y_axis_rotation, axis=Axis.Y) @ get_rotation_matrix(x_axis_rotation,
                                                                                                 axis=Axis.X)


class Translate(Animation):
    def __init__(self, distance=1.0, axis=Axis.X, speed=1.0):
        super().__init__()

        self.distance = distance
        self.speed = speed
        self.axis = axis

    def update(self, delta):
        super().update(delta)

        t = np.sin(self.speed * self.elapsed) * self.distance

        dx = 0.0
        dy = 0.0
        dz = 0.0

        if self.axis == Axis.X:
            dx = t
        elif self.axis == Axis.Y:
            dy = t
        elif self.axis == Axis.Z:
            dz = t

        self.transform = get_translation_matrix(dx, dy, dz)


class Compose(Animation):
    def __init__(self, animations):
        super().__init__()

        assert len(animations) > 0

        self.animations = animations

    def update(self, delta):
        super().update(delta)

        for animation in self.animations:
            animation.update(delta)

    @property
    def transform(self):
        transform = np.eye(4, dtype=np.float32)

        for animation in self.animations:
            transform = transform @ animation.transform

        return transform

    @transform.setter
    def transform(self, value):
        raise RuntimeError(f"{self.__class__.__name__} does not support setter for transform.")
