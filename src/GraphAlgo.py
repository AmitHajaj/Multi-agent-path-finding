import GraphInterface
from GraphAlgoInterface import GraphAlgoInterface
from DiGraph import DiGraph
import json
import heapq


class GraphAlgo(GraphAlgoInterface):

    def __init__(self, graph=None):
        self.graph = graph

    def get_graph(self) -> GraphInterface:
        """
        :return: the directed graph on which the algorithm works on.
        """
        return self.graph

    def load_from_json(self, filePath: str) -> bool:
        """
        Loads a graph from a json file.
        @param filePath: The path to the json file
        @returns True if the loading was successful, False o.w.
        """
        try:
            g = DiGraph()
            with open(filePath, "r") as graphJson:
                graphObj = json.load(graphJson)

                # initilaize graph's nodes
                for node in graphObj["Nodes"]:
                    # initilaize node's pos
                    pos = []
                    for coordinate in node["pos"].split(","):
                        pos.append(coordinate)
                    tuple(pos)
                    # add node to grap
                    g.add_node(node["id"], pos)

                # initialize graph's edges
                for edge in graphObj["Edges"]:
                    g.add_edge(edge["src"], edge["dest"], edge["w"])

                graphJson.close()
                self.graph = g
                return True

        except IOError as e:
            print("Couldn't open or read the file (%s)." % e)
            return False

    def save_to_json(self, filePath: str) -> bool:
        """
        Saves the graph in JSON format to a file
        @param graphJson: The path to the out file
        @return: True if the save was successful, False o.w.
        """
        # make graphObj suitable for json format
        graphObj = {"Nodes": [], "Edges": []}
        g = self.graph
        assert (isinstance(g, DiGraph))

        nodes = g.get_all_v()
        for key in nodes:
            posTuple = nodes[key]["pos"]
            posStr = ','.join(posTuple)
            graphObj["Nodes"].append({"id": key, "pos": posStr})

            neighbors = g.all_out_edges_of_node(key)
            for neiKey in neighbors:
                graphObj["Edges"].append({'src': key, 'w': neighbors[neiKey], 'dest': neiKey})

        try:
            with open(filePath, 'w') as graphJson:
                json.dump(graphObj, graphJson, indent=4)

                graphJson.close()

                return True

        except IOError as e:
            print("Couldn't open or write to file (%s)." % e)
            return False
        json.load(graphJson)
        pass

    def shortest_path(self, id1: int, id2: int) -> (float, list):
        """
        Returns the shortest path from node id1 to node id2 using Dijkstra's Algorithm
        @param id1: The start node id
        @param id2: The end node id
        @return: The distance of the path, a list of the nodes ids that the path goes through

        Example:
#      >>> from GraphAlgo import GraphAlgo
#       >>> g_algo = GraphAlgo()
#        >>> g_algo.addNode(0)
#        >>> g_algo.addNode(1)
#        >>> g_algo.addNode(2)
#        >>> g_algo.addEdge(0,1,1)
#        >>> g_algo.addEdge(1,2,4)
#        >>> g_algo.shortestPath(0,1)
#        (1, [0, 1])
#        >>> g_algo.shortestPath(0,2)
#        (5, [0, 1, 2])

        Notes:
        If there is no path between id1 and id2, or one of them dose not exist the function returns (float('inf'),[])
        More info:
        https://en.wikipedia.org/wiki/Dijkstra's_algorithm
        """
        # if the graph is none.
        if self.graph.v_size() == 0:
            return float('inf', [])

        # if one of those nodes is not in the graph.
        if id1 not in self.graph.nodes.keys() or id2 not in self.graph.nodes.keys():
            return float('inf', [])

        self.dijkstra(id1, id2)

        if self.graph.nodes[id2]["tag"] == 99999999:
            return float('inf', [])

        temp = self.graph.nodes[id2]
        path = [id2]

        while temp["prev"] is not None:
            path.append(temp["prev"])
            temp = self.graph.nodes[temp["prev"]]

        path.reverse()
        return self.graph.nodes[id2]["tag"], path

    def connected_component(self, id1: int) -> list:
        """
        Finds the Strongly Connected Component(SCC) that node id1 is a part of.
        @param id1: The node id
        @return: The list of nodes in the SCC

        Notes:
        If the graph is None or id1 is not in the graph, the function should return an empty list []
        """
        pass

    def connected_components(self) -> list[list]:
        """
        Finds all the Strongly Connected Component(SCC) in the graph.
        @return: The list all SCC

        Notes:
        If the graph is None the function should return an empty list []
        """
        pass

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """
        pass

    def dijkstra(self, src_node, dest_node,):
        """
            implementation of the Dijkstra algorithm for finding a shortest path
            from source to destination. applicable on directed weighted graphs.

            Parameters:
                src_node: int
                        the node we want to start from.
                dest_node: int
                        the node we want to go to.

            no returns. we make changes on the nodes variables and use it outside.
                dest_node.tag will be the weight of the shortest path.
                dest_node.prev will be the previous node in the shortest path.
        """
        # initially, all nodes are unvisited
        # unvisited = self.graph.nodes.keys()
        unvisited = []
        for k in self.graph.nodes.keys():
            unvisited.append(k)

        # make a minheap for the unvisited nodes.
        heapq.heapify(unvisited)

        # set all nodes distance to infinity,
        # and all prev to None.
        for node in self.graph.nodes.keys():
            self.graph.nodes[node]["tag"] = 99999999
            self.graph.nodes[node]["prev"] = None

        # set the src_node to zero.
        self.graph.nodes[src_node]["tag"] = 0

        while len(unvisited) > 0:
            # current node is the neighbor with the minimum weight.
            curr = heapq.heappop(unvisited)

            # take current node neighbors.
            v_neighbors = {k: v for (k, v) in self.graph.edges["From"].items() if curr == k}
            for neighbor in v_neighbors[curr].keys():
                if (self.graph.nodes[curr]["tag"] + v_neighbors[curr][neighbor]) < self.graph.nodes[neighbor]["tag"]:
                    # update the distance
                    self.graph.nodes[neighbor]["tag"] = self.graph.nodes[curr]["tag"] + v_neighbors[curr][neighbor]
                    # update parent node
                    self.graph.nodes[neighbor]["prev"] = curr

            # move the smallest element to the top of the hep for next iteration.
            heapq.heapify(unvisited)
