############
# This is the file where should insert your own code.
#
# Author: Björn Bulkens <bjoern.bulkens@stud.uni-heidelberg.de>
#
# IMPORTANT: You should use the following function definitions without applying
# any changes to the interface, because the function will be used for automated
# checks. It is fine to declare additional helper functions.

###########
import math
from collections import namedtuple
import grid as grid
import itertools
import numpy as np
from copy import deepcopy


Node = namedtuple("Node", "costs")
Edge = namedtuple("Edge", "left right costs")


def evaluate_energy(nodes, edges, assignment):
    energy = 0.0

    # ind_node_idxs = findAllNodes(edges)
    # TEMPNodes = [nodes[i] for i in ind_node_idxs]

    for i in range(0, len(assignment)):
        energy += nodes[i].costs[assignment[i]]

    for i in range(0, len(edges)):
        energy += edges[i].costs[assignment[edges[i].left], assignment[edges[i].right]]

    return energy


def to_image(labeling, dimensions):
    width, height = dimensions
    data = np.asarray(labeling).reshape(height, width) / 15
    return Image.fromarray(np.uint8(data * 255))


def findAllNodes(subproblem):

    theNodes = []
    for edge in subproblem:
        theNodes.append(edge.left)
        theNodes.append(edge.right)
    return list(set(sorted(theNodes)))


# For exercise 1.3
def bruteforce(nodes, edges):
    assignment = [0] * len(nodes)

    tempList = []
    temp = []
    for i in range(0, len(nodes)):
        for i in range(0, len(nodes[i].costs)):
            temp.append(i)
        tempList.append(temp)
        temp = []

    permutations = list(itertools.product(*tempList))

    minEnergy = 1000000000000
    minPerm = [0] * len(nodes)
    for perm in permutations:
        if evaluate_energy(nodes, edges, list(perm)) <= minEnergy:
            minEnergy = evaluate_energy(nodes, edges, list(perm))
            minPerm = perm

    return (minPerm, minEnergy)


def argmin(d):

    try:
        if not d:
            return None
        min_val = min(d.values())
        return [k for k in d if d[k] == min_val][0]
    except:
        print(d)


###################################################################################################################################################################################

# Task 1
# Iterated Conditional Modes (ICM)
#


def icm_update_step(nodes, edges, grid, assignment, u):
    # `assignment` is the current labeling (list that assigns a label to
    # each node).
    # `u` is the index of the current node for which an update should be performed.
    # Task: Update the assignemnt for node `u`.
    # Return: Nothing.
    # Here we have to change the assignment

    prev_assignment = assignment.copy()
    bestLabel = 0
    bestCosts = math.inf

    # print("New Node: ", u, "")
    # print("Before:", evaluate_energy(nodes, edges, prev_assignment))
    for label in range(0, len(nodes[u].costs)):
        prev_assignment[u] = label
        costs = evaluate_energy(nodes, edges, prev_assignment)
        if costs < bestCosts:
            bestCosts = costs
            bestLabel = label

    assignment[u] = bestLabel
    # print("Assignment: ", assignment, "\n\t", bestCosts, " ", bestLabel)


def icm_single_iteration(nodes, edges, grid, assignment):
    # `assignment` is the current labeling (list that assigns a label to
    # each node).
    # Task: Perform a full iteration of all ICM update steps.
    # Return: Nothing.

    for i in range(0, len(nodes)):
        icm_update_step(nodes, edges, grid, assignment, i)


def icm_method(nodes, edges):
    # Task: Run ICM algorithm until convergence.
    # Return: Assignment after converging.

    Grid = grid.determine_grid(nodes, edges)
    assignment = [0 for i in range(0, len(nodes))]
    energy_iter = []
    energy_iter.append(evaluate_energy(nodes, edges, assignment))

    condition = True
    while condition:
        prev_assignment = assignment.copy()
        icm_single_iteration(nodes, edges, Grid, assignment)
        energy_iter.append(evaluate_energy(nodes, edges, assignment))

        if assignment != prev_assignment:
            pass
        else:
            return assignment, energy_iter


#
# Block ICM
#


def dynamic_programming(node_indxes, nodes, edges):
    # Based on the submission 1.4
    F = []
    ptr = []
    for indx, node in enumerate(nodes):
        # The F_0 is simply zero for all labels
        if indx == 0:
            F.append([0 for _ in range(len(node.costs))])
        else:
            # A list of edges that connect the current node to the previous node
            current_edge = [
                edge
                for edge in edges
                if (
                    edge.left == node_indxes[int(indx - 1)]
                    and edge.right == node_indxes[indx]
                )
                or (
                    edge.right == node_indxes[int(indx - 1)]
                    and edge.left == node_indxes[indx]
                )
            ][0]

            # The current node and the previous node
            current_node = nodes[indx]
            prev_node = nodes[int(indx - 1)]
            # Temporary list of F values
            temp_f = []
            temp_ptr = {}
            for label_indx in range(len(current_node.costs)):
                # The next entry is the minimal cost of the previous node plus the cost of the current edge
                next_entry = [
                    F[int(indx - 1)][i]
                    + prev_node.costs[i]
                    + current_edge.costs.get((i, label_indx))
                    for i in range(len(prev_node.costs))
                ]
                # The minimal cost of the next entry
                minimal_edge_arg = min(
                    range(len(next_entry)), key=lambda i: next_entry[i]
                )
                # this would translate to {right node arg: left node arg} acting as a pointer for the minimal path
                temp_ptr[label_indx] = minimal_edge_arg
                temp_f.append(next_entry[minimal_edge_arg])

            F.append(temp_f)
            ptr.append(temp_ptr)
    return F, ptr


def backtrack(nodes, edges, F, ptr):
    assignment = [0] * len(nodes)
    for i in range(len(nodes[-1].costs)):
        F[-1][i] += nodes[-1].costs[i]
    assignment[-1] = min(range(len(F[-1])), key=lambda i: F[-1][i])
    for indx, r in reversed(list(enumerate(ptr))):
        assignment[indx] = r.get(assignment[indx + 1])
    return assignment


def block_icm_update_step(nodes, edges, GRID, assignment, subproblem):
    # `grid` is the helper structure for the grid graph representation
    # of `nodes` and `edges`. See grid example for details.
    # `assignment` is the current labeling (list that assigns a label to
    # each node).
    # `subproblem` is the current chain-structured subproblem.
    # Task: Update the assignemnt for the current suproblem.
    # Return: Nothing.

    # The indices of nodes of our induced subgraph V' = V \ V^f
    ind_node_idxs = findAllNodes(subproblem)
    # The corresponding nodes V'
    ind_nodes = deepcopy([nodes[i] for i in ind_node_idxs])
    # The corresponding edges E'
    ind_edges = subproblem.copy()

    # Use the current assignment to add the cost of col-edges to the unary costs of the subproblem

    #####################################################################
    #####################################################################
    # Question to Tutor: If we need to take the fixed costs of the col-edges into account,
    # we need to add them to the unary costs of the nodes, right?
    #####################################################################
    #####################################################################

    for edge in edges:
        if edge not in ind_edges:
            if edge.left in ind_node_idxs:
                temp = [
                    i
                    for i in range(len(ind_nodes[ind_node_idxs.index(edge.left)].costs))
                ]
                # print(
                #     "Induced Nodes before:", ind_nodes[ind_node_idxs.index(edge.left)]
                # )
                for i in temp:
                    ind_nodes[ind_node_idxs.index(edge.left)].costs[
                        i
                    ] += edge.costs.get((i, assignment[edge.right]))

                # print("Edge costs:", edge.costs)
                # print("Assignment:", assignment[edge.left], assignment[edge.right])
                # print("Induced Nodes after:", ind_nodes[ind_node_idxs.index(edge.left)])
            elif edge.right in ind_node_idxs:
                temp = [
                    i
                    for i in range(
                        len(ind_nodes[ind_node_idxs.index(edge.right)].costs)
                    )
                ]

                for i in temp:
                    ind_nodes[ind_node_idxs.index(edge.right)].costs[
                        i
                    ] += edge.costs.get((assignment[edge.left], i))

            else:
                pass

    # Solve this with dynamic programming for chain-structured graphs
    F, pointer = dynamic_programming(ind_node_idxs, ind_nodes, ind_edges)
    # Backtrack to find the optimal assignment
    assignment = backtrack(ind_nodes, ind_edges, F, pointer)


def block_icm_single_iteration(nodes, edges, GRID, assignment):
    # Similar to ICM but you should iterate over all subproblems in the row
    # column decomposition.

    # For each row/block locally optimize the costs
    rows = grid.row_column_decomposition(GRID)
    for row in rows:
        block_icm_update_step(nodes, edges, GRID, assignment, row)


def block_icm_method(nodes, edges):

    energies = []
    myGrid = grid.determine_grid(nodes, edges)
    assignment = [0 for i in range(0, len(nodes))]

    energies.append(evaluate_energy(nodes, edges, assignment))

    condition = True
    while condition:

        prev_assignment = assignment.copy()
        block_icm_single_iteration(nodes, edges, myGrid, assignment)

        if assignment == prev_assignment:
            condition = False
        else:
            energies.append(evaluate_energy(nodes, edges, assignment))
    return assignment, energies


#########################################################################################################################################

# Task 2
# Subgradient
#


def subgradient_compute_single_subgradient(nodes, edges, grid, edge_idx):
    # Task: Compute the subgradient for given edge of index `edge_idx`.
    # Let n be the number of left labels and m be the number of right
    # labels, the subgradient has n + m elements.
    # Return: A tuple where the first term is a list of the n left
    # subgradient values and the second term is a list of the m right
    # subgradient values.

    # Array of zeros for the left/right subgradient
    LEFTS = np.zeros_like(nodes[edges[edge_idx].left].costs)
    RIGHTS = np.zeros_like(nodes[edges[edge_idx].right].costs)

    # Find the minimum label for the left/right node
    minLabelLeft = np.argmin(nodes[edges[edge_idx].left].costs)
    minLabelRight = np.argmin(nodes[edges[edge_idx].right].costs)
    minEdge = argmin(edges[edge_idx].costs)

    # Find the minimum edge
    for i in range(0, len(nodes[edges[edge_idx].left].costs)):
        if i == minEdge[0] and minLabelLeft == i:
            LEFTS[i] = 0
        if i == minEdge[0] and minLabelLeft != i:
            LEFTS[i] = 1
        if i != minEdge[0] and minLabelLeft == i:
            LEFTS[i] = -1
        if i != minEdge[0] and minLabelLeft != i:
            LEFTS[i] = 0

    for i in range(0, len(nodes[edges[edge_idx].right].costs)):
        if i == minEdge[1] and minLabelRight == i:
            RIGHTS[i] = 0
        if i == minEdge[1] and minLabelRight != i:
            RIGHTS[i] = 1
        if i != minEdge[1] and minLabelRight == i:
            RIGHTS[i] = -1
        if i != minEdge[1] and minLabelRight != i:
            RIGHTS[i] = 0

    return LEFTS, RIGHTS


def subgradient_compute_full_subgradient(nodes, edges, grid):
    # Task: Compute the full subgradient for the full problem.
    # Return dictionary with (u, v) => list of subgradient values.
    # Note: subgradient[u, v] is not identical to subgradient[v, u] (see book
    # where \phi_{u,v} is also not identical to \phi_{v,u}).

    subgradDict = {}
    # Iterate over all edges
    for i in range(0, len(edges)):
        # Compute the subgradient for each edge
        left, right = subgradient_compute_single_subgradient(nodes, edges, grid, i)
        # Add the subgradient to the dictionary
        subgradDict.update({(edges[i].left, edges[i].right): left})
        subgradDict.update({(edges[i].right, edges[i].left): right})

    return subgradDict


## "Reparametrization incorrectly computed."
def subgradient_apply_update(nodes, edges, grid, subgradient, stepsize):
    # Task: Reparametrize the model by modifying the costs (in direction of
    # `subgradient` multiplied by `stepsize`).

    for i in range(0, len(nodes)):
        for j in range(0, len(nodes)):
            for k in range(0, len(edges)):
                if edges[k].left == i and edges[k].right == j:
                    for l in range(0, len(nodes[i].costs)):
                        nodes[i].costs[l] -= subgradient[(i, j)][l] * stepsize

    #  for k, v in subgradient.items():
    #      for i in range(0, len(nodes[k[0]].costs)):
    #          nodes[k[0]].costs[i] -= v[i]*stepsize

    for i in range(0, len(edges)):
        for index, (key, value) in enumerate(edges[i].costs.items()):
            edges[i].costs[key] += (
                subgradient[edges[i].left, edges[i].right][key[0]] * stepsize
            )
            edges[i].costs[key] += (
                subgradient[edges[i].right, edges[i].left][key[0]] * stepsize
            )


def subgradient_update_step(nodes, edges, grid, stepsize):
    # Task: Compute subgradient for the problem and reparametrize the the
    # whole problem.

    subgradientsDict = subgradient_compute_full_subgradient(nodes, edges, grid)
    subgradient_apply_update(nodes, edges, grid, subgradientsDict, stepsize)


### DONE
def subgradient_round_primal(nodes, edges, grid):
    # Task: Implement primal rounding as discussed in the lecture.
    # Return: Assignment (list that assigns each node one label).

    solution = []
    for i in range(0, len(nodes)):
        solution.append(np.argmin(nodes[i].costs))
    return solution


### DONE
def subgradient_method(nodes, edges, iterations):

    # Task: Run subgradient method for given number of iterations.
    # Return: Assignment.
    GRID = grid.determine_grid(nodes, edges)

    beta = 0.2
    alpha = -0.2

    for i in range(0, iterations):

        adaptiveStep = beta * (1 + i) ** alpha
        subgradient_update_step(nodes, edges, GRID, adaptiveStep)

    assign = subgradient_round_primal(nodes, edges, GRID)

    return assign


#########################################################################################################################################

##
# Min-Sum Diffusion
#


def min_sum_diffusion_accumulation(nodes, edges, grid, u):
    # Task: Implement the reparametrization for the accumulation phase of N:
    # print(edges[7].costs[7,0])

    for i in range(0, len(edges)):
        if edges[i].left == u:
            for j in range(0, len(nodes[u].costs)):
                mini = math.inf
                miniLabel = 0
                for k in range(0, len(nodes[edges[i].right].costs)):
                    if mini > edges[i].costs[j, k]:
                        mini = edges[i].costs[j, k]
                        miniLabel = k

                minimum = edges[i].costs[j, miniLabel]
                nodes[u].costs[j] = nodes[u].costs[j] + mini
                for (a, b), c in edges[i].costs.items():
                    if a == j:
                        edges[i].costs[(a, b)] = edges[i].costs[(a, b)] - mini

        if edges[i].right == u:
            for j in range(0, len(nodes[u].costs)):
                mini = math.inf
                miniLabel = 0
                for k in range(0, len(nodes[edges[i].left].costs)):
                    if mini > edges[i].costs[k, j]:
                        mini = edges[i].costs[k, j]
                        miniLabel = k

                minimum = edges[i].costs[miniLabel, j]
                nodes[u].costs[j] = nodes[u].costs[j] + mini
                for (a, b), c in edges[i].costs.items():
                    if a == j:
                        edges[i].costs[(a, b)] = edges[i].costs[(a, b)] - mini


def min_sum_diffusion_distribution(nodes, edges, grid, u):
    # See accumulation, but for distribution.

    count = 0
    # Counte neighbors
    for x in range(0, len(edges)):
        if edges[x].left == u or edges[x].right == u:
            count += 1

    for r in range(0, len(nodes[u].costs)):
        for edge in edges:
            for (a, b), c in edge.costs.items():
                if edge.left == u and a == r:
                    edge.costs[(a, b)] += nodes[u].costs[a] / count
                if edge.right == u and b == r:
                    edge.costs[(a, b)] += nodes[u].costs[b] / count

    for k in range(0, len(nodes[u].costs)):
        nodes[u].costs[k] = 0


def min_sum_diffusion_round_primal(nodes, edges, grid, u):
    # Implement the rounding technique as discussed in the lecture for the
    # given node `u`.
    solution = []
    for o in range(0, len(nodes)):
        solution.append(np.argmin(nodes[o].costs))
    return solution


def min_sum_diffusion_update_step(nodes, edges, grid, u):
    # Implement a single update step for the given node `u` (accumulation,
    # rounding, distribution).
    # Return the assignment/label for node `u`.

    min_sum_diffusion_accumulation(nodes, edges, grid, u)

    min_sum_diffusion_distribution(nodes, edges, grid, u)

    neighbors = []
    for j in range(0, len(edges)):
        if edges[j].left == u:
            label = argmin(edges[j].costs)[0]
            break
        if edges[j].right == u:
            label = argmin(edges[j].costs)[1]
            break
    return label


def min_sum_diffusion_single_iteration(nodes, edges, grid):
    # Implement a single iteration for the Min-Sum diffusion method.
    # Iterate over all nodes and perform the update step on them.
    # Return the assignment/labeling for the full model.
    ASS = []
    for w in range(0, len(nodes)):
        s = min_sum_diffusion_update_step(nodes, edges, grid, w)
        ASS.append(s)
    return ASS


def min_sum_diffusion_method(nodes, edges, grid):
    # Implement the Min-Sum diffusion method (run multiple iterations).
    # Return the assignment/labeling of the full model.

    for u in range(0, 10):

        assign = min_sum_diffusion_single_iteration(nodes, edges, grid)

    return assign


#########################################################################################################################################

#
# Anisotropic Min-Sum Diffusion (TRW-S / SRMP)
#


def trws_accumulation(nodes, edges, grid, forward, u):
    # See Min-Sum diffusion, but implement anisotropic version.
    # `forward` is a boolean that specifies the current direction.
    pass


def trws_distribution(nodes, edges, grid, forward, u):
    # See Min-Sum diffusion, but implement anisotropic version.
    # `forward` is a boolean that specifies the current direction.
    pass


def trws_round_primal(nodes, edges, grid, assignment, forward, u):
    # See Min-Sum diffusion, but implement anisotropic version.
    # `assignment` is a partial assignment/labeling for the full
    # problem (all nodes up to the current one already have a label assigned,
    # all other labels are set to `None`).
    # `forward` is a boolean that specifies the current direction.
    pass


def trws_update_step(nodes, edges, grid, assignment, forward, u):
    # See Min-Sum diffusion, but implement anisotropic version.
    # `forward` is a boolean that specifies the current direction.
    # `assignment` is a partial assignment.
    pass


def trws_single_iteration(nodes, edges, grid, forward):
    # See Min-Sum diffusion, but implement anisotropic version.
    # `forward` is a boolean that specifies the current direction.
    pass


def trws_method(nodes, edges, grid):
    # See Min-Sum diffusion, but implement anisotropic version.
    # `forward` is a boolean that specifies the current direction.
    pass
