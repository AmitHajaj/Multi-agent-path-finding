from GraphInterface import GraphInterface


class DiGraph:
    # TODO construct a graph than can easily make a Json like: {"edges": [{src, weigh, dest}...], "nodes": [{pos: (x, y, z), id}...]}
    # TODO add raises to methods

    """
      A class representing graph
    ...

    Attributes
    ----------
    nodes : dict
        a formatted dictionary contains all graph's nodes

    edges : dict
        a formatted dictionary contains all graph's edges

    Methods
    -------
    v_size()
        Returns the number of vertices in this graph

    e_size()
        Returns the number of edges in this graph

    get_all_v()
        Returns a dictionary of all the nodes in the Graph.
        node is represented using a pair (node_id, node_data)

    all_in_edges_of_node(int: node_id)
        return a dictionary of all the node's edges,
        edge is represented using a pair (destination, weight)

    all_out_edges_of_node(int: node_id)
        return a dictionary of all the edges that pointing at current node,
        each node is represented using a pair(source, weight)

    get_mc()
        Returns the current version of this graph,
        on every change in the graph state - the MC (mode counter) increased

    add_edge(int: source,int: destination, float: weight)
        Adds an edge to the graph.

    add_node(node_id: int, pos: tuple = None)
        Adds a node to the graph.

    remove_node(self, node_id: int)
        Removes a node from the graph.

    remove_edge(self, node_id1: int, node_id2: int)
        Removes an edge from the graph.

    """

    def __init__(self):
        """
        Parameters
        ----------
            nodes : {
                node_id:{
                    "tag": int,
                    "info": "",
                    "pos":(x,y,z),
                    "prev": int
                }
            }

            edges = {
                From: {src: {dest:w}},
                To: {dest: {src:w}}
            }
        """
        self.edges = {"From": {}, "To": {}}  # {src_id: {dest, weight}}
        self.nodes = {}  # {node_id, node_data(like Gson)}
        self.mc = 0

    def add_node(self, node_id: int, pos: tuple = None) -> bool:
        """
        Adds a node to the graph.

        Parameters
        ----------
        node_id: int
            node unique key
        pos: tuple, optional
            node's position (default None)

        Returns
        -------
        bool
            a flag used to indicate if node addition was successful
        """

        if node_id not in self.nodes:
            self.nodes[node_id] = {"tag": 0, "info": "", "pos": pos, "prev": None, "for_scc": {"index": -1, "low_link": node_id, "on_stack": False}}
            self.edges["From"][node_id] = {}
            self.edges["To"][node_id] = {}
            self.mc += 1
            return True
        else:
            return False

    def v_size(self) -> int:
        """"
        Returns
        -------
        int
            the number of vertices this graph contains

        """
        return len(self.nodes)

    def e_size(self) -> int:
        """"
               Returns
               -------
               int
                   the number of edges this graph contains

       """
        return len(self.edges)

    def get_all_v(self) -> dict:
        """"
               Returns
               -------
               dict
                   a dictionary of vertices this graph contains.
                   node is represented using a pair (node_id, node_data)

               """
        return self.nodes

    def all_in_edges_of_node(self, node_id: int) -> dict:
        """"
            Parameters
            ----------
            node_id: int
                node unique key

            Returns
            -------
            dict
                a dictionary of all the node's edges,
                edge is represented using a pair (destination, weight)

       """
        if node_id in self.nodes:
            return self.edges["To"][node_id]
        else:
            return None

    def all_out_edges_of_node(self, node_id: int) -> dict:
        """"
            Parameters
            ----------
            node_id: int
                node unique key

            Returns
            -------
            dict
                return a dictionary of all the edges that pointing at current node,
                each node is represented using a pair(source, weight)

       """
        if node_id in self.nodes:
            return self.edges["From"][node_id]
        else:
            return None

    def get_mc(self) -> int:
        """"
            Returns
            -------
            int
                Returns the current version of this graph
        """
        return self.mc

    def add_edge(self, src: int, dest: int, weight: float) -> bool:
        """"
            Parameters
            ----------
            src: int
                edge's source node
            dest: int
                edge's destination node

            Returns
            -------
            bool
                a flag used to indicate if edge addition was successful

       """
        if {src, dest} <= self.nodes.keys():
            self.edges["From"][src].update({dest: weight})
            self.edges["To"][dest].update({src: weight})

            self.mc += 1
            return True
        else:
            return False

    def remove_node(self, node_id: int) -> bool:
        """"
            Parameters
            ----------
            node_id: int
                node's unique key

            Returns
            -------
            bool
                a flag used to indicate if node's deletion was successful

       """
        if node_id in self.nodes:
            del self.nodes[node_id]
            removedNodeE = self.edges["From"].pop(node_id);
            for neighbor_id in removedNodeE:
                del self.edges["From"][neighbor_id]

            self.mc += 1
            return True
        else:
            return False

    def remove_edge(self, src: int, dest: int) -> bool:
        """"
            Parameters
            ----------
            src: int
                edge's source node
            dest: int
                edge's destination node

            Returns
            -------
            bool
                a flag used to indicate if edge's deletion was successful

       """
        if {src, dest} <= self.nodes.keys():
            del self.edges["From"][src][dest]
            del self.edges["To"][dest][src]
            self.mc += 1
            return True
        else:
            return False

    def equals(self, otherNodes: dict, otherEdges: dict) -> bool:
        # compare edges
        if otherEdges['From'] != self.edges['From']:
            return False

        # compare nodes
        for key in self.nodes:
            if self.nodes[key]['pos'] != otherNodes[key]['pos']:
                return False

        return True
