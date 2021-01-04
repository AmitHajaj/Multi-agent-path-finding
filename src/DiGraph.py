from GraphInterface import GraphInterface


class DiGraph:

    # TODO construct a graph than can easily make a Json like:
    # {
    #    "edges": [{src, weigh, dest}...]
    #    "nodes": [{pos: (x, y, z), id}...]
    # }

    """
        edges = [
            from={src: {dest:w]],
            to={dest: {src:w]]
        ]
    """
    #TODO add comment for node structor
    def __init__(self):
        self.nodes = {}  # {node_id, node_data(like Gson)}
        self.edges = {}  # {src_id: {dest, weight}}
        self.mc = 0

    def add_node(self, node_id: int, pos: tuple = None) -> bool:
        if not self.nodes.__contains__(node_id):
            self.nodes[node_id] = {"neighbors": {}, "neighbor of": {}, "tag": 0, "info": "", "pos": tuple}
            self.edges[node_id] = {}
            self.mc += 1
            return True
        else:
            return False

    def v_size(self) -> int:
        return self.nodes.__sizeof__()

    def e_size(self) -> int:
        return self.edges.__sizeof__()

    def get_all_v(self) -> dict:
        return self.nodes

    def all_in_edges_of_node(self, id1: int) -> dict:
        if self.nodes.keys().__contains__(id1):
            return self.nodes[id1].get("neighbor of")

        else:
            return None

    def all_out_edges_of_node(self, id1: int) -> dict:
        if self.nodes.keys().__contains__(id1):
            return self.nodes[id1].get("neighbors")
        else:
            return None

    def get_mc(self) -> int:
        return self.mc

    def add_edge(self, id1: int, id2: int, weight: float) -> bool:
        if self.nodes.__contains__(id1) and self.nodes.__contains__(id2):
            self.edges[id1][id2] = weight
            self.nodes[id1].get("neighbors")[id2] = weight
            self.nodes[id2].get("neighbor of")[id1] = weight
            self.mc += 1
            return True
        else:
            return False

    def remove_node(self, node_id: int) -> bool:
        if self.nodes.keys().__contains__(node_id):
            for node in self.all_in_edges_of_node(node_id).keys():
                self.remove_edge(node, node_id)

            self.mc += self.all_out_edges_of_node(node_id).__sizeof__()
            del self.edges[node_id]
            del self.nodes[node_id]
            return True
        else:
            return False

    def remove_edge(self, node_id1: int, node_id2: int) -> bool:
        if self.nodes.keys().__contains__(node_id1) and self.nodes.keys().__contains__(node_id2):
            del self.edges[node_id1][node_id2]
            del self.nodes[node_id1].get("neighbors")[node_id2]
            del self.nodes[node_id2].get("neighbor of")[node_id1]
            self.mc += 1
            return True
        else:
            return False
