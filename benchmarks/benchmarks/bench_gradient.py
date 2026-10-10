import numpy as np


class GradientStridedAxis:
    params = [
        [np.float32, np.float64],
        ["C", "F"],
        [
            ((1000, 1000), 0),
            ((1000, 1000), 1),
            ((100, 100, 100), 0),
            ((100, 100, 100), 1),
            ((100, 100, 100), 2),
        ],
    ]
    param_names = ["dtype", "order", "case"]

    def setup(self, dtype, order, case):
        shape, axis = case
        rng = np.random.default_rng(42)
        arr = rng.random(shape).astype(dtype)

        if order == "C":
            self.arr = np.ascontiguousarray(arr)
        elif order == "F":
            self.arr = np.asfortranarray(arr)

        self.dx = 0.01
        self.axis = axis

    def time_gradient_scalar_spacing_axis(self, dtype, order, case):
        np.gradient(self.arr, self.dx, axis=self.axis)
