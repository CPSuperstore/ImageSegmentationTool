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
    def _get_controls(self):
        return [
            segmentation_if.Control("Kappa", "number", "kappa", [0, 200], default=2),
            segmentation_if.Control("Sigma", "number", "sigma", [0, 200], default=100),
        ]

    def _segment(self, image, kwargs):
        # TODO: Clean up code
        image = Image.fromarray(np.uint8(rgb2gray(image) * 255), mode='L')

        image.save("tmp.png")

        kappa = kwargs["kappa"]
        sigma = kwargs["sigma"]

        fore = (225, 142, 279, 185)
        back = (7, 120, 61, 163)

        foreground = image.crop(fore)
        background = image.crop(back)

        image = array(image)
        foreground = array(foreground)
        background = array(background)

        foreground_mean = mean(cv2.calcHist([foreground], [0], None, [256], [0, 256]))
        background_mean = mean(cv2.calcHist([background], [0], None, [256], [0, 256]))

        F, B = ones(shape=image.shape), ones(shape=image.shape)  # initalizing the foreground/background probability vector
        Im = image.reshape(-1, 1)  # Coverting the image array to a vector for ease.
        m, n = image.shape[0], image.shape[1]  # copy the size
        g, pic = maxflow.Graph[int](m, n), maxflow.Graph[int]()  # define the graph
        structure = np.array([[inf, 0, 0],
                              [inf, 0, 0],
                              [inf, 0, 0]
                              ])  # initializing the structure....
        source, sink, J = m * n, m * n + 1, image  # Defining the Source and Sink (terminal)nodes.
        nodes, nodeids = g.add_nodes(m * n), pic.add_grid_nodes(J.shape)  # Adding non-nodes
        pic.add_grid_edges(nodeids, 0), pic.add_grid_tedges(nodeids, J, 255 - J)
        gr = pic.maxflow()
        IOut = pic.get_grid_segments(nodeids)
        for i in range(image.shape[0]):  # Defining the Probability function....
            for j in range(image.shape[1]):
                F[i, j] = -log(abs(image[i, j] - foreground_mean) / (
                        abs(image[i, j] - foreground_mean) + abs(image[i, j] - background_mean)))  # Probability of a pixel being foreground
                B[i, j] = -log(abs(image[i, j] - background_mean) / (
                        abs(image[i, j] - background_mean) + abs(image[i, j] - foreground_mean)))  # Probability of a pixel being background
        F, B = F.reshape(-1, 1), B.reshape(-1, 1)  # convertingb  to column vector for ease
        for i in range(Im.shape[0]):
            Im[i] = Im[i] / linalg.norm(Im[i])  # normalizing the input image vector
        w = structure  # defining the weight
        for i in range(m * n):  # checking the 4-neighborhood pixels
            ws = (F[i] / (F[i] + B[i]))  # source weight
            wt = (B[i] / (F[i] + B[i]))  # sink weight
            g.add_tedge(i, ws[0], wt)  # edges between pixels and terminal
            if i % n != 0:  # for left pixels
                w = kappa * exp(-(abs(Im[i] - Im[i - 1]) ** 2) / sigma)  # the cost function for two pixels
                g.add_edge(i, i - 1, w[0], kappa - w[0])  # edges between two pixels
                '''Explaination of the likelihood function: * used Bayes’ theorem for conditional probabilities
                * The function is constructed by multiplying the individual conditional probabilities of a pixel being either 
                foreground or background in order to get the total probability. Then the class with highest probability is selected.
                * for a pixel i in the image:
                                   * weight from sink to i:
                                   probabilty of i being background/sum of probabilities
                                   * weight from source to i:
                                   probabilty of i being foreground/sum of probabilities
                                   * weight from i to a 4-neighbourhood pixel:
                                    K * e−|Ii−Ij |2 / sigma
                                     where kappa and sigma are parameters that determine hwo close the neighboring pixels are how fast the values
                                     decay towards zero with increasing dissimilarity
                '''
            if (i + 1) % n != 0:  # for right pixels
                w = kappa * exp(-(abs(Im[i] - Im[i + 1]) ** 2) / sigma)
                g.add_edge(i, i + 1, w[0], kappa - w[0])  # edges between two pixels
            if i // n != 0:  # for top pixels
                w = kappa * exp(-(abs(Im[i] - Im[i - n]) ** 2) / sigma)
                g.add_edge(i, i - n, w[0], kappa - w[0])  # edges between two pixels
            if i // n != m - 1:  # for bottom pixels
                w = kappa * exp(-(abs(Im[i] - Im[i + n]) ** 2) / sigma)
                g.add_edge(i, i + n, w[0], kappa - w[0])  # edges between two pixels

        return skimage.measure.find_contours(IOut, 0.5)

