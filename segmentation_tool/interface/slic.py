from skimage.segmentation import slic, mark_boundaries

import segmentation_tool.interface.segmentation_if as segmentation_if


class SLIC(segmentation_if.SegmentationInterface):
    def _get_controls(self):
        return [
            segmentation_if.Control("Segments", "number", "n_segments", default=100, allowed_values=[100, 1000]),
            segmentation_if.Control("Compactness", "number", "compactness", default=1, allowed_values=[0, 100], step=0.01),
            segmentation_if.Control("Max Iterations", "number", "max_num_iter", default=1, allowed_values=[1, 100]),
            segmentation_if.Control("Sigma", "number", "sigma", default=1, allowed_values=[0, 100], step=0.01),
            segmentation_if.Control("Min Size Factor", "number", "min_size_factor", default=1, allowed_values=[1, 10], step=0.01),
            segmentation_if.Control("Max Size Factor", "number", "max_size_factor", default=3, allowed_values=[1, 10], step=0.01),
        ]

    def _segment(self, image, kwargs):
        kwargs["n_segments"] = int(kwargs["n_segments"])
        kwargs["max_num_iter"] = int(kwargs["max_num_iter"])

        if kwargs["compactness"] == 0:
            kwargs["compactness"] = 0.01

        segments = slic(image, **kwargs)
        return mark_boundaries(image, segments, color=(1, 0, 0)) * 255
