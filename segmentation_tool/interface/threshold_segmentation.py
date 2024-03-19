import numpy as np
import skimage

import segmentation_tool.interface.segmentation_if as segmentation_if


class ThresholdSegmentation(segmentation_if.SegmentationInterface):
    def _is_slow(self):
        return True

    def _get_controls(self):
        return [
            segmentation_if.Control("Foreground Color", "color", "fg"),
            segmentation_if.Control("Background Color", "color", "bg")
        ]

    def _segment(self, image, kwargs):
        fg_color = kwargs["fg"]
        bg_color = kwargs["bg"]

        fg_distance = np.linalg.norm(image - fg_color, axis=2)
        bg_distance = np.linalg.norm(image - bg_color, axis=2)

        mask = np.zeros((image.shape[0], image.shape[1]))

        for y, row in enumerate(image):
            for x, distance in enumerate(row):
                if fg_distance[y][x] < bg_distance[y][x]:
                    mask[y][x] = 1

        return skimage.measure.find_contours(mask, 0.5)
