#!/usr/bin/env python3
#
# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

import exercise_1 as student

from collections import namedtuple


Node = namedtuple('Node', 'costs')
Edge = namedtuple('Edge', 'left right costs')


def potts(x_u, x_v, l=1):
    return 0 if x_u == x_v else l


def make_potts(shape, l = 1):
    c = {}
    for x_u in range(shape[0]):
        for x_v in range(shape[1]):
            c[x_u, x_v] = potts(x_u, x_v, l)
    return c


def make_graph(l = 1):
    nodes = [Node(costs=[0.3, 0.9, 0.4]),
             Node(costs=[0.8, 0.1, 0.3]),
             Node(costs=[0.2, 0.5]),
             Node(costs=[0.7, 0.3])]

    edges = []
    for u, v in ((0, 3), (1, 3), (2, 3)):
        shape = tuple(len(nodes[x].costs) for x in (u, v))
        edges.append(Edge(left=u, right=v, costs=make_potts(shape, l)))

    return nodes, edges


def run_example():
    nodes, edges = make_graph(.25)
    intermediates = student.dynamic_programming_tree(nodes, edges)
    print(intermediates)
    print(student.backtrack(nodes, edges, *intermediates))


if __name__ == '__main__':
    run_example()
