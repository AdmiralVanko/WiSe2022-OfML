#!/usr/bin/env python3
# This is the file where should insert your own code.
#
# Author: Your Name <your@email.com>
import itertools
import numpy as np

# For exercise 1.2
def evaluate_energy(nodes, edges, assignment):
    cost = 0

    for (node, a) in zip(nodes, assignment):
        cost += node.costs[a]
    for edge in edges:
        cost += edge.costs[assignment[edge.left], assignment[edge.right]]
    return cost


# For exercise 1.3
def bruteforce(nodes, edges):

    energy_combinations = list(
        itertools.product(*[[i for i in range(len(node.costs))] for node in nodes])
    )

    min_energy = 999
    min_assignment = None

    for assignment in energy_combinations:
        energy = evaluate_energy(nodes, edges, assignment)
        if energy < min_energy:
            min_energy = energy
            min_assignment = assignment

    return (min_assignment, min_energy)


# Extra for exercise 1.4 and 1.6
def edge_cost(nodes, edges):
    # This function calculates the cost of the edges adding the cost of the left node
    for edge in edges:
        for (i, j) in edge.costs.keys():
            edge.costs[i, j] += nodes[edge.left].costs[i]

    return edges


# For exercise 1.4
def dynamic_programming(nodes, edges):

    # Assuming that the nodes are sorted from left to right and neighbors are fully connected as in the example
    # for convenience so we can use a 2D array to store the values
    # Calculate the minimum cost for connecting the second node to the first node (i.e. the first edge)
    F = np.zeros((len(nodes), len(nodes[0].costs)))
    ptr = np.zeros((len(nodes), len(nodes[0].costs)))

    edges = edge_cost(nodes, edges)

    for i in range(0, len(nodes) - 1):

        lefts = {left for (left, _) in edges[i].costs.keys()}
        rights = {right for (_, right) in edges[i].costs.keys()}
        temp = np.zeros(len(lefts))
        if i == 0:
            for right in rights:
                for left in lefts:
                    temp[left] = edges[i].costs[left, right]
                F[i][right] = np.min(temp)
                ptr[i][right] = np.argmin(temp)

        else:
            for right in rights:
                for left in lefts:
                    temp[left] = F[i - 1][left] + edges[i].costs[left, right]

                F[i][right] = np.min(temp)
                ptr[i][right] = np.argmin(temp)

    # Add the cost of the last node
    F[-1] += nodes[-1].costs
    ptr[-1] = np.argmin(F[-1])

    return F, ptr


def backtrack(nodes, edges, F, ptr):
    assignment = np.zeros(len(nodes))

    for i in range(len(nodes) - 1, -1, -1):
        if i == len(nodes) - 1:
            assignment[i] = np.argmin(F[i])
        else:
            print(assignment[i + 1])
            assignment[i] = ptr[i][int(assignment[i + 1])]

    return assignment


# For exercise 1.5
def compute_min_marginals(nodes, edges):
    m = [[0 for l in n] for n in nodes]
    return m


# For execise 1.6
def dynamic_programming_tree(nodes, edges):

    edges = edge_cost(nodes, edges)

    # A leaf node is one that is only on the left side of an edge (in this case)
    # Find a leaf node
    leaf_idx = None
    F = np.zeros((len(nodes), len(nodes[0].costs)))
    ptr = np.zeros((len(nodes), len(nodes[0].costs)))

    edges = edges.sort(key=lambda x: x.left)  # Sort the edges by the left node
    no_leaf = []

    for i in range(len(nodes)):
        # If the node is not on the right side of any edge then it is a leaf node (in this case)
        if all([edge.right != i for edge in edges]):
            _, right = (
                nodes.pop(i),
                [edge for edge in edges if edge.left == i],
            )  # Remove the leaf node from the list of nodes
            no_leaf += right # This keeps track of the edges that are not leaf nodes so there is a entry for F and ptr
            break
        # Calculate the minimum cost for connecting the parent to the leaf
        # This is the same as the dynamic programming algorithm
        lefts = {left for (left, _) in edges[i].costs.keys()}
        rights = {right for (_, right) in edges[i].costs.keys()}
        temp = np.zeros(len(lefts))
        if not i in no_leaf:
            for right in rights:
                for left in lefts:
                    temp[left] = edges[i].costs[left, right]
                F[i][right] = np.min(temp)
                ptr[i][right] = np.argmin(temp)
        else:
            for right in rights:
                for left in lefts:
                    temp[left] = F[right][left] + edges[i].costs[left, right]

                F[i][right] = np.min(temp)
                ptr[i][right] = np.argmin(temp)


def backtrack_tree(nodes, edges, F, ptr):
    assignment = [0] * len(nodes)
    return assignment
