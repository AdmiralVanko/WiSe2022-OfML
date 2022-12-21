#!/usr/bin/env python3
import time
from pulp import *
import time
import numpy as np
from model_2_4 import *
from PIL import Image
import numpy as np
from collections import namedtuple
import os
from PIL import Image

####################################################################################################################################################
def evaluate_energy(nodes, edges, assignment):
    energy = 0.0

    for i in range(0, len(assignment)):
        energy += nodes[i].costs[assignment[i]]

    for i in range(0, len(edges)):
        energy += edges[i].costs[assignment[edges[i].left], assignment[edges[i].right]]

    return energy


#################################################################################################################################################### EXERCISE 2.1


# Method that converts an graphical model into an linear program
def convert_to_ilp(nodes, edges):

    node_var = [
        [LpVariable(f"Node{i}{j}", cat="Binary") for j in range(len(nodes[i].costs))]
        for i in range(0, len(nodes))
    ]

    # edge_var for the edges
    edge_var = []

    for i in range(0, len(edges)):
        temp = []
        for (a, b), _ in edges[i].costs.items():

            temp.append(
                LpVariable(
                    f"Edge{edges[i].left}{edges[i].right}{a}{b}",
                    cat="Binary",
                )
            )
        edge_var.append(temp)

    # Create optimization problem
    problem = LpProblem("myProblem", LpMinimize)

    # Constraint 1
    for i in range(0, len(node_var)):
        problem += lpSum(node_var[i]) == 1

    # Constraint 2
    for i in range(0, len(edge_var)):
        problem += lpSum(edge_var[i]) == 1

    # Constraint 3 Left
    for i in range(0, len(edge_var)):
        test = True
        count = 0
        splitted = []
        while test:
            splitted.append(edge_var[i][count : count + len(node_var[edges[i].right])])
            if count + len(node_var[edges[i].right]) >= len(edge_var[i]):
                test = False
            else:
                count += len(node_var[edges[i].right])
        splitted = np.array(splitted)
        val = 0
        for m in range(0, len(splitted)):
            temp3 = 0
            for k in range(0, len(splitted[m])):
                temp3 += splitted[m][k]
            if val <= 4:
                problem += temp3 == node_var[edges[i].left][val]
                val += 1

    # Constraint 3 Right
    for i in range(0, len(edge_var)):
        test = True
        count = 0
        splitted = []
        while test:
            splitted.append(edge_var[i][count : count + len(node_var[edges[i].right])])
            if count + len(node_var[edges[i].right]) >= len(edge_var[i]):
                test = False
            else:
                count += len(node_var[edges[i].right])
        splitted = np.array(splitted).transpose()
        val = 0
        for m in range(0, len(splitted)):
            temp3 = 0

            for k in range(0, len(splitted[m])):
                temp3 += splitted[m][k]
            if val <= 4:
                problem += temp3 == node_var[edges[i].right][val]
                val += 1

    # What we want to minimize
    summe2 = 0
    summe1 = 0
    for i in range(0, len(edge_var)):
        for index, key in enumerate(edges[i].costs):
            summe2 += edge_var[i][index] * edges[i].costs[key]
    for i in range(0, len(node_var)):
        for h in range(0, len(node_var[i])):
            summe1 += node_var[i][h] * nodes[i].costs[h]
    problem += summe1 + summe2

    solution = problem.solve()

    # In the terminal you can clearly see the correct solution but I dont know where to return it in the code as 'solution'
    return problem, problem.objective.value()


####################################################################################################################################################
# Method that returns the optimal labeling for a given solved ILP
def ilp_to_labeling(_, __, problem, ___):
    solution = []
    for v in problem.variables():
        if str(v.name)[0] == "N" and v.varValue == 1:
            solution.append(int(str(v.name)[-1]))
    return tuple(solution)


#################################################################################################################################################### EXERCISE 2.2
def convert_to_lp(nodes, edges):
    # Basically ilp but with continuous variables

    # node_var for the nodes
    node_var = [
        [
            LpVariable(f"Node{i}{j}", cat="Continuous")
            for j in range(len(nodes[i].costs))
        ]
        for i in range(0, len(nodes))
    ]

    count = []

    # edge_var for the edges
    edge_var = []
    for i in range(0, len(edges)):
        temp2 = []
        for (a, b), _ in edges[i].costs.items():

            temp2.append(
                LpVariable(
                    f"Edge{edges[i].left}{edges[i].right}{a}{b}",
                    lowBound=0,
                    upBound=1,
                    cat="Continuous",
                )
            )
        edge_var.append(temp2)

    # Create optimization problem
    problem = LpProblem("myProblem", LpMinimize)

    # Constraint 1
    for i in range(0, len(node_var)):
        problem += lpSum(node_var[i]) == 1

    # Constraint 2 ---> WORKS
    for i in range(0, len(edge_var)):
        problem += lpSum(edge_var[i]) == 1

    # Constraint 3 Left
    for i in range(0, len(edge_var)):
        test = True
        count = 0
        splitted = []
        while test:
            splitted.append(edge_var[i][count : count + len(node_var[edges[i].right])])
            if count + len(node_var[edges[i].right]) >= len(edge_var[i]):
                test = False
            else:
                count += len(node_var[edges[i].right])
        splitted = np.array(splitted)
        val = 0
        for m in range(0, len(splitted)):
            temp3 = 0
            for k in range(0, len(splitted[m])):
                temp3 += splitted[m][k]
            if val <= 4:
                problem += temp3 == node_var[edges[i].left][val]
                val += 1

    # Constraint 3 Right
    for i in range(0, len(edge_var)):
        test = True
        count = 0
        splitted = []
        while test:
            splitted.append(edge_var[i][count : count + len(node_var[edges[i].right])])
            if count + len(node_var[edges[i].right]) >= len(edge_var[i]):
                test = False
            else:
                count += len(node_var[edges[i].right])
        splitted = np.array(splitted).transpose()
        val = 0
        for m in range(0, len(splitted)):
            temp3 = 0

            for k in range(0, len(splitted[m])):
                temp3 += splitted[m][k]
            if val <= 4:
                problem += temp3 == node_var[edges[i].right][val]
                val += 1

    # What we want to minimize
    summe2 = 0
    summe1 = 0
    for i in range(0, len(edge_var)):
        for index, key in enumerate(edges[i].costs):
            summe2 += edge_var[i][index] * edges[i].costs[key]
    for i in range(0, len(node_var)):
        for h in range(0, len(node_var[i])):
            summe1 += node_var[i][h] * nodes[i].costs[h]
    problem += summe1 + summe2

    solution = problem.solve()

    return problem, problem.objective.value()


####################################################################################################################################################
def lp_to_labeling(nodes, edges, problem, optimalValue):

    NODES = []
    EDGES = []
    assignment = []

    for v in problem.variables():
        if str(v.name)[0] == "N":
            NODES.append((v.varValue))
        if str(v.name)[0] == "E":
            EDGES.append((v.varValue))

    index = 0
    for i in range(0, len(nodes)):
        k = len(nodes[i].costs)
        for _ in range(0, len(nodes[i])):
            slicedArray = NODES[index : index + k]
            assignment.append(slicedArray.index(max(slicedArray)))
            index = index + k

    LP = str(problem.objective.value())
    ILP = str(convert_to_ilp(nodes, edges)[1])
    rounded = str(evaluate_energy(nodes, edges, assignment))

    print()
    print("Optimal LP Energy: " + LP)
    print("Rounded LP Energy: " + rounded)
    print("Optimal ILP Energy: " + ILP)
    print(assignment)
    #### THE FIRST LABEL ONLY IS WRONG for 2.2
    return tuple(assignment), optimalValue


#################################################################################################################################################### EXERCISE 2.4
def test_models_2_4():

    solution_string = """\n\n# ACYCLIC #\n# MODELS #\n"""  # for the report

    for i, (nodes, edges) in enumerate(ACYCLIC_MODELS, start=1):
        ILP = str(convert_to_ilp(nodes, edges)[1])
        LP = str(convert_to_lp(nodes, edges)[1])
        solution_string += f"Model {i}\n ILP {ILP}\n LP {LP}\n\n"

    solution_string += """\n\n# CYCLIC #\n# MODELS #\n"""  # for the report

    for i, (nodes, edges) in enumerate(CYCLIC_MODELS):
        ILP = str(convert_to_ilp(nodes, edges)[1])
        LP = str(convert_to_lp(nodes, edges)[1])
        solution_string += f"Model {i}\n ILP {ILP}\n LP {LP}\n\n"

    print(solution_string)


######################################################################################################################################### EXERCISE 2.5

# Did not do exercise 2.5

########################################################################################################################################## EXERCISE 2.6


def overSegmentation():

    start_time = time.time()
    # Load all picture and keep them as binary arrays (1=White, 0=black)
    PICTURES = []
    directory = "segments"
    for filename in os.listdir(directory):
        picture = directory + "/" + filename
        img = Image.open(picture).convert("L")
        img = np.array(img)
        PICTURES.append(img)
    PICTURES = np.array(PICTURES)

    # Array of LPVariables
    toUse = []
    for i in range(0, len(PICTURES)):
        toUse.append(LpVariable("Segmentation " + str(i), cat="Binary"))

    # Array of LPVariables
    RESULT = []
    temp = []
    counter = 0
    for i in range(0, len(PICTURES[0])):
        for j in range(0, len(PICTURES[0][i])):
            counter += 1
            temp.append(
                LpVariable("RESULT" + str(i) + str(j) + str(counter), cat="Binary")
            )
        RESULT.append(temp)
        temp = []

    # Create optimization problem
    problem = LpProblem("myProblem", LpMaximize)

    # Optimization problem
    for j in range(0, len(PICTURES[0])):
        for k in range(0, len(PICTURES[0][j])):
            temp = 0
            for i in range(0, len(PICTURES)):
                temp += PICTURES[i][j][k] * toUse[i] * (1 / 255)
            RESULT[j][k] += temp
            problem += RESULT[j][k] <= 1
    problem += lpSum(RESULT)

    # Solve Optimization problem
    LpSolverDefault.msg = 1
    status = problem.solve()
    # Create Output image
    toCheck = {}
    for var in problem.variables():
        if str(var.name)[0] == "S":
            # print(var.name.split())
            toCheck[int(var.name.split("_")[1])] = var.value()

    print("Skipped Images:")
    toCheck = dict(sorted(toCheck.items()))
    summe = np.zeros_like(PICTURES[0])
    for key, value in toCheck.items():
        if value == 1.0:
            summe += PICTURES[key]

    # Create Image out of array
    img = Image.fromarray(np.array(np.array(summe)))
    img.show()
    img.save("RESULT.png")
    print("Time: " + str(round(time.time() - start_time, 0)) + " seconds")


if __name__ == "__main__":
    test_models_2_4()
    overSegmentation()
