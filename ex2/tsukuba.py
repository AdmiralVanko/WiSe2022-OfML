# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

from collections import namedtuple
import gzip
import re


Node = namedtuple('Node', 'costs')
Edge = namedtuple('Edge', 'left right costs')


#
# Internal functions to read the input file format.
#

def tokenize_file(f):
    r = re.compile(r'[a-zA-Z0-9.]+')
    for line in f:
        line = line.rstrip('\r\n')
        for m in r.finditer(line):
            yield m.group(0)


def parse_uai(tokens):
    header = next(tokens)
    assert header == 'MARKOV'
    num_nodes = int(next(tokens))
    nodes = [None] * num_nodes
    edges = []
    for i in range(num_nodes):
        nodes[i] = Node(costs=[0] * int(next(tokens)))

    node_list = []
    num_costs = int(next(tokens))
    for i in range(num_costs):
        num_vars = int(next(tokens))
        node_list.append(tuple(int(next(tokens)) for j in range(num_vars)))

    cost_cache = {}
    for i in range(num_costs):
        size = int(next(tokens))
        if len(node_list[i]) == 1:
            u, = node_list[i]
            assert size == len(nodes[u].costs)
            for x_u in range(len(nodes[u].costs)):
                nodes[u].costs[x_u] = float(next(tokens))
        elif len(node_list[i]) == 2:
            u, v = node_list[i]
            costs = {}
            assert size == len(nodes[u].costs) * len(nodes[v].costs)
            for x_u in range(len(nodes[u].costs)):
                for x_v in range(len(nodes[v].costs)):
                    costs[(x_u, x_v)] = float(next(tokens))

            cache_key = repr(costs)
            try:
                costs = cost_cache[cache_key]
            except KeyError:
                cost_cache[cache_key] = costs

            edges.append(Edge(left=u, right=v, costs=costs))
        else:
            raise RuntimeError('Higher-order factors not supported.')
    return nodes, edges


def load_uai(filename):
    open_func = open
    if filename.endswith('.gz'):
        open_func = gzip.open

    with open_func(filename, 'rt') as f:
        return parse_uai(tokenize_file(f))


#
# Ready-to-use models for exercise.
#

ALL_MODEL_DOWNSAMPLINGS = [1, 2, 4, 8, 16, 32]


def load_downsampled_model(downsampling):
    assert downsampling in ALL_MODEL_DOWNSAMPLINGS
    filename = 'models/tsu_{:02d}.uai.gz'.format(downsampling)
    return load_uai(filename)


def all_models():
    for downsampling in reversed(ALL_MODEL_DOWNSAMPLINGS):
        yield load_downsampled_model(downsampling)


__all__ = ['ALL_MODEL_DOWNSAMPLINGS', 'load_downsampled_model', 'all_models']
