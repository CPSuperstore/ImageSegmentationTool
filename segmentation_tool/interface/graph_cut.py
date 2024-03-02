# https://github.com/lwthatcher/Graph-Cut/tree/master
# https://www.csd.uwo.ca/~yboykov/Papers/pami04.pdf
# An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision
# Yuri Boykov and Vladimir Kolmogorov, 2004

import cv2
import maxflow
import skimage
from PIL import Image
from pylab import *
from skimage.color import rgb2gray

import segmentation_tool.interface.segmentation_if as segmentation_if


class GraphCut(segmentation_if.SegmentationInterface):
    def _is_slow(self):
        return True

    def _get_controls(self):
        return [
            segmentation_if.Control("Kappa", "number", "kappa", [0, 200], default=2),
            segmentation_if.Control("Sigma", "number", "sigma", [0, 200], default=100),
        ]

    def _segment(self, image, kwargs):
        image = np.uint8(rgb2gray(image) * 255)

        pil_image = Image.fromarray(image, mode='L')

        kappa = kwargs["kappa"]
        sigma = kwargs["sigma"]

        # TODO: Add way to control these bboxes
        foreground_bbox = (225, 142, 279, 185)
        background_bbox = (7, 120, 61, 163)

        foreground = pil_image.crop(foreground_bbox)
        background = pil_image.crop(background_bbox)

        foreground = array(foreground)
        background = array(background)

        foreground_mean = mean(cv2.calcHist([foreground], [0], None, [256], [0, 256]))
        background_mean = mean(cv2.calcHist([background], [0], None, [256], [0, 256]))

        # initialize the foreground/background probability vector
        foreground_probability_vector = ones(shape=image.shape)
        background_probability_vector = ones(shape=image.shape)

        # Converting the image array to a vector for ease.
        vector_image = image.reshape(-1, 1)

        m, n = image.shape[0], image.shape[1]

        graph = maxflow.Graph[int](m, n)
        pic = maxflow.Graph[int]()

        graph.add_nodes(m * n)
        node_ids = pic.add_grid_nodes(image.shape)

        pic.add_grid_edges(node_ids, 0)
        pic.add_grid_tedges(node_ids, image, 255 - image)

        pic.maxflow()

        segmented = pic.get_grid_segments(node_ids)

        # Define the Probability function
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):

                # Probability of a pixel being in the foreground
                foreground_probability_vector[i, j] = -log(
                    abs(image[i, j] - foreground_mean) /
                    (abs(image[i, j] - foreground_mean) + abs(image[i, j] - background_mean))
                )

                # Probability of a pixel being in the background
                background_probability_vector[i, j] = -log(
                    abs(image[i, j] - background_mean) /
                    (abs(image[i, j] - background_mean) + abs(image[i, j] - foreground_mean))
                )

        # convert  to column vector for ease
        foreground_probability_vector = foreground_probability_vector.reshape(-1, 1)
        background_probability_vector = background_probability_vector.reshape(-1, 1)

        # normalize the input image vector
        for i in range(vector_image.shape[0]):
            vector_image[i] = vector_image[i] / linalg.norm(vector_image[i])

        # checking the 4-neighborhood pixels
        for i in range(m * n):
            source_weight = (
                    foreground_probability_vector[i] /
                    (foreground_probability_vector[i] + background_probability_vector[i])
            )
            sink_weight = (
                    background_probability_vector[i] /
                    (foreground_probability_vector[i] + background_probability_vector[i])
            )

            # edges between pixels and terminal
            graph.add_tedge(i, source_weight[0], sink_weight)

            # find the cost function for two pixels
            if i % n != 0:  # for left pixels
                w = kappa * exp(-(abs(vector_image[i] - vector_image[i - 1]) ** 2) / sigma)
                graph.add_edge(i, i - 1, w[0], kappa - w[0])  # edges between two pixels

            if (i + 1) % n != 0:  # for right pixels
                w = kappa * exp(-(abs(vector_image[i] - vector_image[i + 1]) ** 2) / sigma)
                graph.add_edge(i, i + 1, w[0], kappa - w[0])  # edges between two pixels

            if i // n != 0:  # for top pixels
                w = kappa * exp(-(abs(vector_image[i] - vector_image[i - n]) ** 2) / sigma)
                graph.add_edge(i, i - n, w[0], kappa - w[0])  # edges between two pixels

            if i // n != m - 1:  # for bottom pixels
                w = kappa * exp(-(abs(vector_image[i] - vector_image[i + n]) ** 2) / sigma)
                graph.add_edge(i, i + n, w[0], kappa - w[0])  # edges between two pixels

        return skimage.measure.find_contours(segmented, 0.5)
