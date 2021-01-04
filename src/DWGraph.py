from node1 import Vertex


class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_node(self, node):
        if self.vert_dict.keys().__contains__(node):
            self.num_vertices = self.num_vertices + 1
            new_vertex = Vertex(node)
            self.vert_dict[node] = new_vertex
            return True
        else:
            return False

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, src, dest, weight=0):
        if src not in self.vert_dict:
            self.add_node(src)
        if dest not in self.vert_dict:
            self.add_node(dest)

        self.vert_dict[src].add_neighbor(self.vert_dict[dest], weight)
        self.vert_dict[dest].add_neighbor(self.vert_dict[src], weight)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous
