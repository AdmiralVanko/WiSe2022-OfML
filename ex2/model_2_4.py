# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

from collections import namedtuple
import random


Node = namedtuple('Node', 'costs')
Edge = namedtuple('Edge', 'left right costs')


def random_cost():
    return (random.random() - 0.5) * 100


def make_pairwise(shape):
    c = {}
    for x_u in range(shape[0]):
        for x_v in range(shape[1]):
            c[x_u, x_v] = random_cost()
    return c


def make_graph(cyclic):
    nodes = []
    for node in range(10 if cyclic else 30):
        nodes.append(Node(costs=[random_cost() for x in range(5)]))

    edges = []
    if cyclic:
        for left in range(len(nodes)):
            for right in range(left+1, len(nodes)):
                shape = tuple(len(nodes[x].costs) for x in (left, right))
                edges.append(Edge(left=left, right=right, costs=make_pairwise(shape)))
    else:
        for left in range(len(nodes)-1):
            right = left + 1
            shape = tuple(len(nodes[x].costs) for x in (left, right))
            edges.append(Edge(left=left, right=right, costs=make_pairwise(shape)))

    return nodes, edges


random.seed(0x42)
ACYCLIC_MODELS = [make_graph(cyclic=False) for i in range(10)]
CYCLIC_MODELS = [make_graph(cyclic=True) for i in range(10)]

__all__ = ['ACYCLIC_MODELS', 'CYCLIC_MODELS']

#
# To use the above definitions, create a new Python source file in the same
# directory and import this file:
#
# from model_2_4 import *
#
# You can now simply refer to `ACYCLIC_MODELS` and `CYCLIC_MODELS`.
#
