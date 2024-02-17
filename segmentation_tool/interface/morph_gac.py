import skimage
from skimage.color import rgb2gray
import numpy as np
from skimage.segmentation import morphological_geodesic_active_contour

import segmentation_tool.interface.segmentation_if as segmentation_if


class MorphGAC(segmentation_if.SegmentationInterface):
    def _get_controls(self):
        return [
            segmentation_if.Control("Iterations", "number", "num_iter", [1, 1000]),
            segmentation_if.Control("Smoothing", "number", "smoothing", [1, 10]),
            segmentation_if.Control("Threshold", "number", "threshold", [0, 1], step=0.01),
            segmentation_if.Control("Balloon", "number", "balloon", [-1, 1], step=0.01),
        ]

    def _segment(self, image, kwargs):
        gray_image = rgb2gray(image)

        init_ls = np.zeros(gray_image.shape, dtype=np.int8)
        init_ls[10:-10, 10:-10] = 1

        kwargs["num_iter"] = int(kwargs["num_iter"])
        kwargs["smoothing"] = int(kwargs["smoothing"])

        # List with intermediate results for plotting the evolution
        segmented = morphological_geodesic_active_contour(
            gray_image,
            **kwargs
        )

        return skimage.measure.find_contours(segmented, 0.5)
