#!/usr/bin/env python3
# This is the file where should insert your own code.
#
# Author: Your Name <your@email.com>


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
    assignment = [0] * len(nodes)

    # print(edges)
    min_energy = 999
    min_assignment = None

    energy = evaluate_energy(nodes, edges, assignment)
    if energy < min_energy:
        min_energy = energy
        min_assignment = assignment.copy()

    if not min_assignment:
        return None
    return (min_assignment, min_energy)


# For exercise 1.4
def dynamic_programming(nodes, edges):
    F, ptr = None, None
    return F, ptr

def backtrack(nodes, edges, F, ptr):
    assignment = [0] * len(nodes)
    return assignment


# For exercise 1.5
def compute_min_marginals(nodes, edges):
    m = [[0 for l in n] for n in nodes]
    return m


# For execrise 1.6
def dynamic_programming_tree(nodes, edges):
    F, ptr = None, None
    return F, ptr

def backtrack_tree(nodes, edges, F, ptr):
    assignment = [0] * len(nodes)
    return assignment
