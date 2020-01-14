import maxflow
from numpy import (
    full,
    inf,
    zeros,
    log,
    mgrid,
    array,
    int_,
    logical_not,
)
from random import choice

from utils import (
    neighbor_exists,
    lookup_table,
    get_neighbor_coordinate,
)

import matplotlib.pyplot as plt


class MaxFlowGraph():
    def __init__(self, L, S, image):
        self.L = L
        self.S = S
        self.image = image
        self.labeling = image
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.nodes = zeros((image.shape[0], image.shape[1], 2))
        self.edges = zeros((image.shape[0], image.shape[1], 4, 2, 2))
        self.alpha = -1
        self.list_of_labels = [k for k in range(256)]

    def get_random_alpha(self):
        """Gets random label for alpha-expansion iteration
        Removes chosen alpha from list not to chose it any more during the iteration
        """
        self.alpha = choice(self.list_of_labels)
        self.list_of_labels.remove(self.alpha)

    def calculate_weights_of_two_label_graph(self):
        """Calculate weights of nodes and edges for two-label graph
        """
        self.nodes = zeros((self.height, self.width, 2))
        self.edges = zeros((self.height, self.width, 4, 2, 2))
        for i in range(self.height):
            for j in range(self.width):
                k = self.labeling[i, j]
                self.nodes[i, j, 0] = self.node_weight(self.image[i, j], k)
                self.nodes[i, j, 1] = self.node_weight(self.image[i, j], self.alpha)
                for n in range(4):
                    if neighbor_exists(i, j, n, self.height, self.width):
                        i_n, j_n = get_neighbor_coordinate(i, j, n)
                        k_n = self.labeling[i_n, j_n]
                        self.edges[i_n, j_n, n, 0, 1] = self.edge_weight(k, self.alpha)
                        self.edges[i_n, j_n, n, 1, 0] = self.edge_weight(self.alpha, k_n)
                        self.edges[i_n, j_n, n, 0, 0] = self.edge_weight(k, k_n)
                        self.edges[i_n, j_n, n, 1, 1] = self.edge_weight(self.alpha, self.alpha)

    def update_weights_of_two_label_graph(self):
        """Update weights of nodes and edges so that parallel edges have zero costs
        """
        for i in range(self.height):
            for j in range(self.width):
                k = self.labeling[i, j]
                for n in range(4):
                    if neighbor_exists(i, j, n, self.height, self.width):
                        i_n, j_n = get_neighbor_coordinate(i, j, n)
                        k_n = self.labeling[i_n, j_n]
                        a = self.edge_weight(k, k_n)
                        b = self.edge_weight(self.alpha, k_n)
                        c = self.edge_weight(k, self.alpha)
                        d = self.edge_weight(self.alpha, self.alpha)
                        self.nodes[i, j, 1] = d - c
                        self.nodes[i_n, j_n, 0] = a
                        self.nodes[i_n, j_n, 1] = c
                        self.edges[i, j, n, 0, 0] = 0
                        self.edges[i, j, n, 0, 1] = 0
                        self.edges[i, j, n, 1, 1] = 0
                        self.edges[i, j, n, 1, 0] = b + c - a - d

    def update_labeling(self, segments):
        """Update image after maxflow

        Parameters
        ----------
        matrix of binary values of image size
            Values correspond to sink or source segment of maxflow graph
        """
        for i in range(self.height):
            for j in range(self.width):
                if int_(logical_not(segments[i, j])) == 1:
                    self.labeling[i, j] = self.alpha

    def alpha_expansion_step(self):
        """Solve maxflow problem for 2-labeled graph with chosen alpha
        """
        self.calculate_weights_of_two_label_graph()
        self.update_weights_of_two_label_graph()
        # Create the graph
        g = maxflow.Graph[float]()
        # Add the nodes. nodeids has the identifiers of the nodes in the grid
        nodeids = g.add_grid_nodes((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                for n in range(4):
                    if neighbor_exists(i, j, n, self.height, self.width):
                        i_n, j_n = get_neighbor_coordinate(i, j, n)
                        # Add non-terminal edges
                        g.add_edge(
                            nodeids[i, j], nodeids[i_n, j_n],
                            self.edges[i, j, n, 0, 1], self.edges[i, j, n, 1, 0]
                        )
                # Add terminal edges
                g.add_tedge(nodeids[i, j],
                            self.nodes[i, j, 0],
                            self.nodes[i, j, 1])
        # Find the maximum flow
        g.maxflow()
        segments = g.get_grid_segments(nodeids)
        self.update_labeling(segments)

    def alpha_expansion_iteration(self):
        """Solve maxflow problem for all alphas
        """
        self.list_of_labels = [k for k in range(256)]
        while len(self.list_of_labels) > 0:
            self.get_random_alpha()
            print("Current alpha ", self.alpha, ".", len(self.list_of_labels), "alphas left")
            self.alpha_expansion_step()

    def alpha_expansion(self, number_of_iterations):
        """Perform iterations of alpha-expansion

        Parameters
        ----------
        number_of_iterations: unsigned integer
            Number of iterations
        """
        for i in range(number_of_iterations):
            print("Iteration", i + 1, "of", number_of_iterations)
            self.alpha_expansion_iteration()

    def edge_weight(self, label1, label2):
        """Computing of edge weight between two labels for initial problem

        Parameters
        ----------
        label1: int
            Intensity of one object (pixel)
        label2: int
            Intensity of other object

        Returns
        -------
        float
            Edge weight
        """
        return self.L * log(
            1 + lookup_table[label1, label2] / (2 * self.S ** 2))

    def node_weight(self, label1, label2):
        """Computing of node weight for the given label in object

        Parameters
        ----------
        label1: int
            Intensity of the pixel in image
        label2: int
            Intensity that corresponds to some label in graph

        Returns
        -------
        int
            Node weight
        """
        return lookup_table[label1, label2]
