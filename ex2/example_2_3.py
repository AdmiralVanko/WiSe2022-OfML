#!/usr/bin/env python3
#
# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

import exercise_2 as student

from collections import namedtuple


Node = namedtuple('Node', 'costs')
Edge = namedtuple('Edge', 'left right costs')


def make_pairwise(shape):
    c = {}
    for x_u in range(shape[0]):
        for x_v in range(shape[1]):
            c[x_u, x_v] = 1 if x_u == x_v else 0
    return c


def make_graph():
    nodes = [Node(costs=[0.5, 0.5]),
             Node(costs=[0.0, 0.0]),
             Node(costs=[0.2, 0.2])]

    edges = []
    for u, v in ((0, 1), (0, 2), (1, 2)):
        shape = tuple(len(nodes[x].costs) for x in (u, v))
        edges.append(Edge(left=u, right=v, costs=make_pairwise(shape)))

    return nodes, edges


def run_example():
    nodes, edges = make_graph()
    lp = student.convert_to_lp(nodes, edges)
    res = lp.solve()
    assert res

    for var in lp.variables():
        print('{} -> {}'.format(var.name, var.value()))


if __name__ == '__main__':
    run_example()
