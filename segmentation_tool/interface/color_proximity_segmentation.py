import numpy as np
import skimage

import segmentation_tool.interface.segmentation_if as segmentation_if


class ColorProximitySegmentation(segmentation_if.SegmentationInterface):
    def _get_controls(self):
        return [
            segmentation_if.Control("Color", "color", "color"),
            segmentation_if.Control("Proximity", "number", "proximity", [0, 255], default=50)
        ]

    def _segment(self, image, mask, kwargs):
        color = kwargs["color"]
        proximity = kwargs["proximity"]

        color_distance = np.linalg.norm(image - color, axis=2) <= proximity
        return skimage.measure.find_contours(color_distance, 0.5, mask=mask)
