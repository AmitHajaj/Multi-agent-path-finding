import copy

import GraphInterface
from GraphAlgoInterface import GraphAlgoInterface
from DiGraph import DiGraph
import matplotlib.pyplot as plt
import math
import json
import heapq
import sys
# for debugging
import time


class GraphAlgo(GraphAlgoInterface):
    """
      A class that contains algorithms that could be done on graphs
    ...

    Attributes
    ----------
    graph : DiGraph
        object representing a graph

    Methods
    -------
    get_graph()
        Returns a graph that implements GraphInterface

    load_from_json(file_name: str)
        loads a graph object to class from json
        Returns true if successful

    save_to_json(file_name: str)
        Saves the graph in JSON format to a file
        Returns true if successful

    shortest_path( src: int, dest: int)
        calculate the shortest path from node src to node dest using Dijkstra's Algorithm
        Returns The distance of the path, and a list of the nodes ids that the path goes through

    connected_component(node_id: int)
        Finds the Strongly Connected Component(SCC) that node node_id is a part of.
        Returns The list of nodes in the SCC

    connected_component()
        Finds all the Strongly Connected Component(SCC) in the graph.
        Returns The list all SCC

    def plot_graph()
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed spraed on a circle.
    """

    def __init__(self, graph=None):
        """
        Parameters
        ----------
            graph : a class that implements GraphInterface

        """
        self.graph = graph if graph is not None else DiGraph()

    def get_graph(self) -> GraphInterface:
        """
        Returns
        -------
        GraphInterface
            the directed graph on which the algorithm works on.
        """
        return self.graph

    def load_from_json(self, filePath: str) -> bool:
        """
        loads a graph from a text file

        Parameters
        ----------
        filePath: str
            address to the text file

        Returns
        -------
        bool
            a flag used to indicate if the graph's load was successful
        """
        try:
            g = DiGraph()
            with open(filePath, "r") as graphJson:
                graphObj = json.load(graphJson)

                # initialize graph's nodes
                for node in graphObj["Nodes"]:
                    # initialize node's pos
                    pos = None
                    posStr = node["pos"]
                    if posStr is not None:
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

        Parameters
        ----------
        filePath: str
            address to the out file

        Returns
        -------
        bool
            a flag used to indicate if the graph's load was successful

        """
        # make graphObj suitable for json format
        graphObj = {"Nodes": [], "Edges": []}
        g = self.graph
        # assert (isinstance(g, DiGraph))

        if g is None:
            return False

        nodes = g.get_all_v()
        for key in nodes:
            posTuple = nodes[key]["pos"]

            pos = None
            if posTuple is not None:
                pos = ','.join(map(str, posTuple))

            graphObj["Nodes"].append({"id": key, "pos": pos})

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

    def shortest_path(self, src: int, dest: int) -> (float, list):
        """
        calculate the shortest path from node src to node dest using Dijkstra's Algorithm
        Parameters
        ----------
        src: int
            The start node id

        dest: int
            The end node id

        Returns
        -------
        float
            The distance of the path
        list
            a list of the nodes ids that the path goes through

        """
        # if the graph is none.
        if self.graph is None:
            return float('inf'), []

        # if one of those nodes is not in the graph.
        if src not in self.graph.nodes.keys() or dest not in self.graph.nodes.keys():
            return float('inf'), []

        self.dijkstra(src, dest)

        if self.graph.nodes[dest]["tag"] == sys.maxsize:
            return float('inf'), []

        temp = self.graph.nodes[dest]
        path = [dest]

        while temp["prev"] is not None:
            path.append(temp["prev"])
            temp = self.graph.nodes[temp["prev"]]

        path.reverse()
        path_length = self.graph.nodes[dest]["tag"]

        # before the end, the variables that holds info in this function need to be set to default.
        for node in self.graph.nodes.keys():
            self.graph.nodes[node]["tag"] = 0
            self.graph.nodes[node]["prev"] = None

        return path_length, path

    def connected_component(self, node_id: int) -> list:
        """
         Finds the Strongly Connected Component(SCC) that node node_id is a part of.

        Parameters
        ----------
        node_id: int
            node unique key

        Returns
        -------
        list
            a list of nodes in the
        """
        # empty graph, or node in that graph.
        if self.graph.v_size() == 0 or node_id not in self.graph.nodes.keys() or self.graph is None:
            return []
        # f there is only one node in the graph.
        if self.graph.v_size() == 1:
            return [node_id]

        # else, call tge heavy shit!!
        scc_of_graph = self.connected_components()

        # check at the returned SCC where node_id is.
        # check at the returned SCC's where id1 is.
        for component in scc_of_graph:
            if node_id in component:
                return component

        # when finished, set back the values used to deafult.
        for node in self.graph.nodes.keys():
            self.graph.nodes[node]["for_scc"] = {"index": -1, "low_link": node, "on_stack": False}

        return []

    def connected_components(self) -> list[list]:
        """
        Finds all the Strongly Connected Component(SCC) in the graph.

        Returns
        -------
        list
            a list all SCC (s list of lists),
            or an empty list if the graph is None.
        """
        # empty graph, or node in that graph.
        if self.graph is None or self.graph.v_size() == 0:
            return []

        components = []
        d_time = []

        # mark all node as unvisited.
        for node in self.graph.nodes.keys():
            self.graph.nodes[node]["for_scc"]["on_stack"] = False

        # make a list of all node based on their discovery time order.
        for node in self.graph.nodes.keys():
            if not self.graph.nodes[node]["for_scc"]["on_stack"]:
                self.dfs(node, d_time, False, self.graph)

        # create a transposed graph.
        t_graph = self.reverse()
        d_time.reverse()

        # mark all t_graph as not visited.
        for node in t_graph.nodes.keys():
            t_graph.nodes[node]["for_scc"]["on_stack"] = False

        # now make another dfs on t_graph.
        # but here we will consider the order of discover.
        while d_time:
            v = d_time.pop()
            if not t_graph.nodes[v]["for_scc"]["on_stack"]:
                components.append(self.dfs(v, d_time, True, t_graph))

        return components

    def reverse(self) -> DiGraph:
        """
            for a DiGraph, this method returns it's transposed graph.
            Returns
            -------
            DiGraph
                    the transposed graph.
        """

        graph = DiGraph()
        graph.nodes = self.graph.nodes
        graph.edges["From"] = self.get_graph().edges["To"]
        graph.edges["To"] = self.get_graph().edges["From"]
        return graph

    def dfs(self, s: int, l: list, stat: bool, g: DiGraph) -> list:
        """
            dfs method with some improvement. in case 0 we update l to list order by discovery time.
            in case 1 we make dfs to find the SCC for this node.

            Parameters
            ----------
            s: int
                source node.
            l: list
                list to set throw the case 0 dfs.
            stat: bool
                case to perform. false is case 0, true is case 1.
            g: DiGraph
                graph to perform dfs on.

            Returns
            -------
                    None: for case 0.
                    comp: for case 1 it returns a list that is a SCC for source node given.
        """
        # create stack for DFS
        stack = [s]
        g.nodes[s]["tag"] = 0
        l.insert(g.nodes[s]["tag"], s)
        if stat:
            comp = []
            stack = [s]

        while stack.__len__() != 0:
            s = stack.pop()
            # if first seen, add him to the current component or to the list of time discover.
            if not g.nodes[s]["for_scc"]["on_stack"]:
                g.nodes[s]["for_scc"]["on_stack"] = True
                if stat:
                    comp.append(s)

            indexer = 0
            # now we need te check the neighbors of this node.
            for ne in g.edges["From"][s].keys():

                if not g.nodes[ne]["for_scc"]["on_stack"]:
                    if stat:
                        comp.append(ne)
                        g.nodes[ne]["for_scc"]["on_stack"] = True
                        stack.append(ne)
                    else:
                        g.nodes[ne]["for_scc"]["on_stack"] = True
                        stack.append(ne)
                        l.insert(g.nodes[s]["tag"]+1+indexer, ne)
                        g.nodes[ne]["tag"] = g.nodes[s]["tag"]+1+indexer
                        indexer += 1
        if stat:
            return comp
        else:
            return None

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """
        if self.graph is None:
            return None

        # get nodes
        g = self.graph
        assert (isinstance(g, DiGraph))
        nodes = g.get_all_v()

        # initialize vars calculation
        i = 1
        maxX = maxY = -9999999
        minX = minY = 9999999

        # set a dict of nodes locations {id: pos}
        locations = {}
        for key in nodes:
            pos = nodes[key]['pos']

            # if node's location in None
            if pos is None:
                # calculate position of each node
                maxX = maxY = 2
                minY = minX = -2
                angle = math.radians(i * 360 / len(nodes))
                pos = (2 * math.cos(angle), 2 * math.sin(angle))

                i += 1

            # if node's location initialize
            else:
                nodeX, nodeY = float(pos[0]), float(pos[1])
                pos = tuple(pos)
                maxX, minX = max(maxX, nodeX), min(minX, nodeX)
                maxY, minY = max(maxY, nodeY), min(minY, nodeY)

            locations[key] = pos

        # set window size
        dx = (maxX - minX) / 4
        dy = (maxY - minY) / 4
        plt.xlim([minX - dx, maxX + dx])
        plt.ylim([minY - dy, maxY + dy])

        # draw graph
        r = min((maxY - minY), (maxX - minX)) * (3 / 80)
        for key in locations:
            # draw nodes
            x = float(locations[key][0])
            y = float(locations[key][1])
            circle = plt.Circle((x, y), label=key, edgecolor='k', facecolor='r', radius=r / 2, zorder=5)
            plt.gcf().gca().add_artist(circle)

            # draw edges
            for neiKey in g.all_out_edges_of_node(key).keys():
                # slope = (neiY - y) / (neiX - x)
                # arrowY = y + arrowLen * (slope / math.sqrt(1 + slope * slope))
                # arrowX = x + arrowLen * (1 / math.sqrt(1 + slope * slope))
                neiX = float(locations[neiKey][0])
                neiY = float(locations[neiKey][1])

                # draw a line (for a non-directed edge)
                if self.isDirectedE(neiKey, key):
                    lineX, lineY = [x, neiX], [y, neiY]
                    plt.plot(lineX, lineY, color='k', zorder=0)

                # draw an arrow (for a directed edge)
                else:
                    distance = math.hypot((neiY - y), (neiX - x))
                    cosAngle = (neiX - x) / distance
                    sinAngle = (neiY - y) / distance
                    dx = (distance - 2 * r) * cosAngle
                    dy = (distance - 2 * r) * sinAngle
                    plt.arrow(x, y, dx, dy, head_width=r, width=r / 10, zorder=0)

        plt.show()
        # time.sleep(60)

    def isDirectedE(self, node1, node2) -> bool:
        return node1 in self.graph.edges['To'][node2] and node1 in self.graph.edges['From'][node2]

    # def dfs(self, visited, node_id: int):
    #     if node_id not in visited:
    #         visited.add(node_id)
    #         for neighbor in self.graph.edges["From"][node_id]:
    #             self.dfs(neighbor)

    def dijkstra(self, src_node, dest_node):
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
        q = []

        # set all nodes distance to infinity,
        # and all prev to None.
        for node in self.graph.nodes.keys():
            self.graph.nodes[node]["tag"] = sys.maxsize
            self.graph.nodes[node]["prev"] = None
            q.append(node)

        # set the src_node to zero.
        self.graph.nodes[src_node]["tag"] = 0
        curr = src_node

        while len(q) > 0:
            # choose the minimum weigh in the graph edges
            temp = sys.maxsize
            for node in self.graph.nodes.keys():
                if self.graph.nodes[node]["tag"] < temp and node in q:
                    temp = self.graph.nodes[node]["tag"]
                    curr = node
            # in case all other node are not reachable
            if temp == sys.maxsize:
                curr = q.pop()
            else:
                q.remove(curr)
            # relaxation on the neighbors of curr.
            v_neighbors = {k: v for (k, v) in self.graph.edges["From"].items() if curr == k}
            for neighbor in v_neighbors[curr].keys():
                if (self.graph.nodes[curr]["tag"] + v_neighbors[curr][neighbor]) < self.graph.nodes[neighbor]["tag"]:
                    # update the distance
                    self.graph.nodes[neighbor]["tag"] = self.graph.nodes[curr]["tag"] + v_neighbors[curr][neighbor]
                    # update parent node
                    self.graph.nodes[neighbor]["prev"] = curr
