
class Node:

    def __init__(self, node_id: int):
        self.id = node_id
        self.neighbor = {}
        self.point_at_me = {}
        self.pos = None

    def add_neighbor(self, neighbor_id: int, weight: float):
        self.neighbor[neighbor_id] = weight

    """
    Adds new neighbor to this node.
    @param neighbor_id: the node we want to add.
    """

    def add_point_at_me(self, node_id: int, weight: float):
        self.point_at_me[node_id] = weight

    def remove_neighbor(self, neighbor_id: int) -> bool:
        if self.neighbor.__contains__(neighbor_id):
            del self.neighbor[neighbor_id]
            return True
        else:
            return False

    """
    Remove a node from this node neighbors list.
    @param neighbor_id: the neighbor we want to remove.
    """

    def get_neighbors(self) -> dict:
        return self.neighbor.keys()
    """
    Returns a dictionary of the id's of each neighbor.
    """

    def get_from_me(self) -> dict:
        return self.neighbor

    def get_to_me(self) -> dict:
        return self.point_at_me

    def set_pos(self, pos: tuple):
        self.pos = pos

    def get_pos(self) -> tuple:
        return self.pos





