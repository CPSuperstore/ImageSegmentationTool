import skimage
from skimage.color import rgb2gray
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set

import segmentation_tool.interface.segmentation_if as segmentation_if


class MorphACWE(segmentation_if.SegmentationInterface):
    def _get_controls(self):
        return [
            segmentation_if.Control("Iterations", "number", "num_iter", [1, 10]),
            segmentation_if.Control("Square Size", "number", "square_size", [1, 10], default=6),
            segmentation_if.Control("Smoothing", "number", "smoothing", [1, 10], default=1),
            segmentation_if.Control("Lambda 1", "number", "lambda1", [0, 2], default=1, step=0.01),
            segmentation_if.Control("Lambda 2", "number", "lambda2", [0, 2], default=1, step=0.01),
        ]

    def _segment(self, image, mask, kwargs):
        gray_image = rgb2gray(image)
        init_ls = checkerboard_level_set(gray_image.shape, int(kwargs.pop("square_size")))

        kwargs["num_iter"] = int(kwargs["num_iter"])
        kwargs["smoothing"] = int(kwargs["smoothing"])
        kwargs["init_level_set"] = init_ls

        segmented = morphological_chan_vese(gray_image, **kwargs)
        segments = skimage.measure.find_contours(segmented, 0.5, mask=mask)

        return segments
