# This is the file where should insert your own code.
#
# Author: Your Name <your@email.com>

import pulp
import numpy as np


def convert_to_lp(nodes, edges):
    lp = pulp.LpProblem("GM")
    print("HIER!")
    # populate LP
    return lp


def lp_to_labeling(nodes, edges, lp):
    labeling = []
    # compute labeling
    return labeling


def convert_to_ilp(nodes, edges):
    ilp = pulp.LpProblem("GM")
    # populate ILP

    # build a numpy array of nodes and costs
    nodes = np.array([node.costs for node in nodes])

    edges = np.array([edge.costs for edge in edges])

    # create a pulp variable for each node
    variables = [
        [
            pulp.LpVariable(
                "n_{}_{}".format(i, j), lowBound=0, upBound=1, cat=pulp.LpInteger
            )
            for j in range(nodes.shape[1])
        ]
        for i in range(nodes.shape[0])
    ]

    assert np.shape(np.array(variables)) == np.shape(nodes)

    # Create objective function (<x, c>)
    for i in range(len(variables)):
        obj = pulp.LpAffineExpression(
            [(variables[i][j], nodes[i][j]) for j in range(len(variables[i]))]
        )

        ilp += obj

    # Create a single simplex constraint (sum of variables = 1) for each list in variables
    for i in range(len(variables)):
        lhs = pulp.LpAffineExpression(
            [(variables[i][j], 1) for j in range(len(variables[i]))]
        )
        constr = pulp.LpConstraint(lhs, pulp.LpConstraintEQ, "simplex constraint", 1)
        ilp += constr

    # Right now we could solve optimal labeling without edges
    # Create ilp variables for each edge
    edge_variables = [
        [
            pulp.LpVariable(
                "e_{}_{}".format(i, j), lowBound=0, upBound=1, cat=pulp.LpInteger
            )
            for j in range(len(edges[i].costs))
        ]
        for i in range(len(edges))
    ]

    # Create objective function (<x, c>)
    for i in range(len(edge_variables)):
        obj = pulp.LpAffineExpression(
            [(edge_variables[i][j], edges[i][j]) for j in range(len(edge_variables[i]))]
        )
        ilp += obj

    # Create a single simplex constraint (sum of variables = 1) for each list in edge_variables
    for i in range(len(edge_variables)):
        lhs = pulp.LpAffineExpression(
            [(edge_variables[i][j], 1) for j in range(len(edge_variables[i]))]
        )
        constr = pulp.LpConstraint(lhs, pulp.LpConstraintEQ, "simplex constraint", 1)
        ilp += constr

    return ilp

    # Add constraint that each edge variable connects two active nodes


def ilp_to_labeling(nodes, edges, ilp):
    labeling = []
    # compute labeling
    return labeling
