# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>


class Grid:

    def __init__(self, nodes, edges, width, height):
        self._nodes = nodes
        self._edges = edges
        self.width = width
        self.height = height

        if __debug__:
            for i in range(width*height):
                assert i == self.linear_index(*self.grid_index(i))

    def linear_index(self, i, j):
        assert i >= 0 and i < self.height
        assert j >= 0 and j < self.width
        return i * self.width + j

    def grid_index(self, u):
        assert u >= 0 and u < self.width*self.height
        return (u // self.width, u % self.width)

    def neighbors(self, u, isotropic=True):
        neighbors = []
        i, j = self.grid_index(u)

        if isotropic:
            if j > 0:
                neighbors.append(u - 1)

            if i > 0:
                neighbors.append(u - self.width)

        if j + 1 < self.width:
            neighbors.append(u + 1)

        if i + 1 < self.height:
            neighbors.append(u + self.width)

        return neighbors

    def edge_index(self, u, v):
        assert u >= 0 and u < self.height*self.width
        assert v >= 0 and u < self.height*self.width
        assert u < v
        assert v in self.neighbors(u, isotropic=False)
        i, j = self.grid_index(u)

        idx = (2*(self.width-1) + 1) * i + 2*j

        if i == self.height - 1:
            idx -= j

        if i != self.height-1 and j != self.width-1:
            idx = idx if (u + 1 == v) else idx + 1

        return idx

    def edge(self, u, v):
        edge = self._edges[self.edge_index(u, v)]
        assert edge.left == u and edge.right == v
        return edge

    def edges(self, u):
        return [self.edge(u, v) for v in self.neighbors(u, isotropic=False)]


def determine_grid(nodes, edges):
    i = 0
    while edges[2*i].left == i and edges[2*i].right == i+1:
        i += 1

    width = i + 1
    assert len(nodes) % width == 0
    height = len(nodes) // width

    return Grid(nodes, edges, width, height)


def row_column_decomposition(grid):
    decomposition = []

    # rows
    for i in range(grid.height):
        row = []
        for j in range(grid.width - 1):
            u = grid.linear_index(i, j)
            v = grid.linear_index(i, j+1)
            row.append(grid.edge(u, v))
        decomposition.append(row)

    # columns
    for j in range(grid.width):
        column = []
        for i in range(grid.height-1):
            u = grid.linear_index(i, j)
            v = grid.linear_index(i+1, j)
            column.append(grid.edge(u, v))
        decomposition.append(column)

    return decomposition
