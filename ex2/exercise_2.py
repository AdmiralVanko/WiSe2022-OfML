#!/usr/bin/env python3
import time
from pulp import *
import time
import math
import numpy as np
import model_2_4 as model
from PIL import Image
import numpy as np
from collections import namedtuple
import logging
import math
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

    # mu1 for the nodes
    mu1 = []
    temp1 = []
    counter = 0
    for i in range(0, len(nodes)):
        for j in range(0, len(nodes[i].costs)):
            counter += 1
            temp1.append(
                LpVariable("Node" + str(i) + str(j) + str(counter), cat="Binary")
            )
        mu1.append(temp1)
        temp1 = []

    # mu2 for the edges
    mu2 = []
    temp2 = []
    counter = 0
    for i in range(0, len(edges)):
        for (a, b), c in edges[i].costs.items():
            counter += 1
            temp2.append(
                LpVariable(
                    "Edge"
                    + str(edges[i].left)
                    + str(edges[i].right)
                    + str(a)
                    + str(b)
                    + str(counter),
                    cat="Binary",
                )
            )
        mu2.append(temp2)
        temp2 = []

    # Create optimization problem
    problem = LpProblem("myProblem", LpMinimize)

    # Constraint 1
    for i in range(0, len(mu1)):
        problem += lpSum(mu1[i]) == 1

    # Constraint 2 ---> WORKS
    for i in range(0, len(mu2)):
        problem += lpSum(mu2[i]) == 1

    # Constraint 3 Left
    for i in range(0, len(mu2)):
        test = True
        count = 0
        splitted = []
        while test:
            splitted.append(mu2[i][count : count + len(mu1[edges[i].right])])
            if count + len(mu1[edges[i].right]) >= len(mu2[i]):
                test = False
            else:
                count += len(mu1[edges[i].right])
        splitted = np.array(splitted)
        val = 0
        for m in range(0, len(splitted)):
            temp3 = 0
            for k in range(0, len(splitted[m])):
                temp3 += splitted[m][k]
            if val <= 4:
                problem += temp3 == mu1[edges[i].left][val]
                val += 1

    # Constraint 3 Right
    for i in range(0, len(mu2)):
        test = True
        count = 0
        splitted = []
        while test:
            splitted.append(mu2[i][count : count + len(mu1[edges[i].right])])
            if count + len(mu1[edges[i].right]) >= len(mu2[i]):
                test = False
            else:
                count += len(mu1[edges[i].right])
        splitted = np.array(splitted).transpose()
        val = 0
        for m in range(0, len(splitted)):
            temp3 = 0

            for k in range(0, len(splitted[m])):
                temp3 += splitted[m][k]
            if val <= 4:
                problem += temp3 == mu1[edges[i].right][val]
                val += 1

    # What we want to minimize
    summe2 = 0
    summe1 = 0
    for i in range(0, len(mu2)):
        for index, key in enumerate(edges[i].costs):
            summe2 += mu2[i][index] * edges[i].costs[key]
    for i in range(0, len(mu1)):
        for h in range(0, len(mu1[i])):
            summe1 += mu1[i][h] * nodes[i].costs[h]
    problem += summe1 + summe2

    solution = problem.solve()
    print(solution)

    return problem, solution


####################################################################################################################################################
# Method that returns the optimal labeling for a given solved ILP
def ilp_to_labeling(nodes, edges, problem, omtimalValue):
    solution = []
    for v in problem.variables():
        if str(v.name)[0] == "N" and v.varValue == 1:
            solution.append(int(str(v.name)[-1]))
    return tuple(solution), omtimalValue


#################################################################################################################################################### EXERCISE 2.2
def convert_to_lp(nodes, edges):

    # mu1 for the nodes
    mu1 = []
    temp1 = []

    count = []

    counter = 0
    for i in range(0, len(nodes)):
        for j in range(0, len(nodes[i].costs)):
            counter += 1
            temp1.append(
                LpVariable(
                    "Node" + str(i) + str(j) + str(counter),
                    lowBound=0,
                    upBound=1,
                    cat="Continuous",
                )
            )
            count.append("Node" + str(i) + str(j) + str(counter))
        mu1.append(temp1)
        temp1 = []

    # mu2 for the edges
    mu2 = []
    temp2 = []
    counter = 0
    for i in range(0, len(edges)):
        for (a, b), c in edges[i].costs.items():
            counter += 1
            temp2.append(
                LpVariable(
                    "Edge"
                    + str(edges[i].left)
                    + str(edges[i].right)
                    + str(a)
                    + str(b)
                    + str(counter),
                    lowBound=0,
                    upBound=1,
                    cat="Continuous",
                )
            )
        mu2.append(temp2)
        temp2 = []

    # Create optimization problem
    problem = LpProblem("myProblem", LpMinimize)

    # Constraint 1
    for i in range(0, len(mu1)):
        problem += lpSum(mu1[i]) == 1

    # Constraint 2 ---> WORKS
    for i in range(0, len(mu2)):
        problem += lpSum(mu2[i]) == 1

    # Constraint 3 Left
    for i in range(0, len(mu2)):
        test = True
        count = 0
        splitted = []
        while test:
            splitted.append(mu2[i][count : count + len(mu1[edges[i].right])])
            if count + len(mu1[edges[i].right]) >= len(mu2[i]):
                test = False
            else:
                count += len(mu1[edges[i].right])
        splitted = np.array(splitted)
        val = 0
        for m in range(0, len(splitted)):
            temp3 = 0
            for k in range(0, len(splitted[m])):
                temp3 += splitted[m][k]
            if val <= 4:
                problem += temp3 == mu1[edges[i].left][val]
                val += 1

    # Constraint 3 Right
    for i in range(0, len(mu2)):
        test = True
        count = 0
        splitted = []
        while test:
            splitted.append(mu2[i][count : count + len(mu1[edges[i].right])])
            if count + len(mu1[edges[i].right]) >= len(mu2[i]):
                test = False
            else:
                count += len(mu1[edges[i].right])
        splitted = np.array(splitted).transpose()
        val = 0
        for m in range(0, len(splitted)):
            temp3 = 0

            for k in range(0, len(splitted[m])):
                temp3 += splitted[m][k]
            if val <= 4:
                problem += temp3 == mu1[edges[i].right][val]
                val += 1

    # What we want to minimize
    summe2 = 0
    summe1 = 0
    for i in range(0, len(mu2)):
        for index, key in enumerate(edges[i].costs):
            summe2 += mu2[i][index] * edges[i].costs[key]
    for i in range(0, len(mu1)):
        for h in range(0, len(mu1[i])):
            summe1 += mu1[i][h] * nodes[i].costs[h]
    problem += summe1 + summe2

    # Solve the ILP
    LpSolverDefault.msg = 0

    # print(problem)
    # status = problem.solve()
    # Solve problem
    # status = problem.solve(GLPK(msg = 0))
    # m = GEKKO()
    # m.Minimize(problem)
    # m.solve()
    # print(time.clock() - start_time, "seconds")

    return problem, 12


####################################################################################################################################################
def lp_to_labeling(nodes, edges, problem, omtimalValue):
    solution = []
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
        for j in range(0, len(nodes[i])):
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
    return tuple(assignment)


#################################################################################################################################################### EXERCISE 2.4
def CalcExercise4():

    i = 1
    for nodes, edges in model.ACYCLIC_MODELS:
        ILP = str(convert_to_ilp(nodes, edges)[1])
        LP = str(convert_to_lp(nodes, edges)[1])
        print(i)
        print("ILP " + ILP)
        print("LP " + LP)
        print()
        i += 1

    print()
    print()
    print()

    i = 1
    for nodes, edges in model.CYCLIC_MODELS:
        ILP = str(convert_to_ilp(nodes, edges)[1])
        LP = str(convert_to_lp(nodes, edges)[1])
        print(i)
        print("ILP " + ILP)
        print("LP " + LP)
        print()
        i += 1


######################################################################################################################################### EXERCISE 2.6


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
