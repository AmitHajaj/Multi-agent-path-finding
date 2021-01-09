import GraphInterface
from GraphAlgoInterface import GraphAlgoInterface
from DiGraph import DiGraph
import matplotlib.pyplot as plt
import math
import json
import heapq
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
        self.graph = graph

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

                # initilaize graph's nodes
                for node in graphObj["Nodes"]:
                    # initilaize node's pos
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
        assert (isinstance(g, DiGraph))

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
        pass

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
        bool
            The distance of the path, a list of the nodes ids that the path goes through

        """
        # if the graph is none.
        if self.graph.v_size() == 0:
            return float('inf'), []

        # if one of those nodes is not in the graph.
        if src not in self.graph.nodes.keys() or dest not in self.graph.nodes.keys():
            return float('inf'), []

        self.dijkstra(src, dest)

        if self.graph.nodes[dest]["tag"] == 99999999:
            return float('inf'), []

        temp = self.graph.nodes[dest]
        path = [dest]

        while temp["prev"] is not None:
            path.append(temp["prev"])
            temp = self.graph.nodes[temp["prev"]]

        path.reverse()
        return self.graph.nodes[dest]["tag"], path

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
        if self.graph.v_size() == 0 or id not in self.graph.nodes.keys() or self.graph is None:
            return []
        # f there is only one node in the graph.
        if self.graph.v_size() == 1:
            return [node_id]

        # else, call tge heavy shit!!
        scc_of_graph = self.connected_components()

        # check at the returned SCC where node_id is.
        for component in scc_of_graph:
            if node_id in component:
                return component

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
        # initialize some variables
        index = 0
        stack = []
        sc_comp = [[]]

        # helper function.
        # recursive function that dfs over the tree.
        # it does it with some changes to fit our goal of finding SCC
        def strong_connect(node_id: int) -> list:
            nonlocal index
            nonlocal stack
            nonlocal sc_comp

            self.graph.nodes[node_id]["for_scc"]["index"] = index
            self.graph.nodes[node_id]["for_scc"]["low_link"] = index
            index += 1
            stack.push(node_id)
            self.graph.nodes[node_id]["for_scc"]["on_stack"] = True

            # Consider successors of node_id
            for neighbor in self.graph.edges["From"][node_id].keys():
                if self.graph.nodes[neighbor]["for_scc"]["index"] == -1:
                    # Successor w has not yet been visited; recurse on it
                    strong_connect(neighbor)
                    self.graph.nodes[neighbor]["for_scc"]["low_link"] = min(
                        self.graph.nodes[neighbor]["for_scc"]["low_link"],
                        self.graph.nodes[node_id]["for_scc"]["low_link"])
                elif self.graph.nodes[neighbor]["for_scc"]["on_stack"]:
                    # Successor neighbor is in stack, and hence in the current SCC
                    # If w is not on stack, then (node_id, neighbor) is an edge pointing to an SCC already found and must be ignored
                    # Note: The next line may look odd - but is correct.
                    # It says w.index not w.lowlink; that is deliberate and from the original paper
                    self.graph.nodes[node_id]["for_scc"]["low_link"] = min(
                        self.graph.nodes[neighbor]["for_scc"]["low_link"],
                        self.graph.nodes[neighbor]["for_scc"]["index"])

            # if node_id is a root node, pop the stack and generate an SCC
            if self.graph.nodes[node_id]["for_scc"]["low_link"] == self.graph.nodes[node_id]["for_scc"]["index"]:
                temp = []

                while True:
                    w = stack.pop()
                    self.graph.nodes[w]["for_scc"]["on_stack"] = False
                    temp.append(w)
                    if node_id is not w:
                        break

                # it will return the SCC for node_id
                return temp

        for node in self.graph.nodes.keys():
            if self.graph.nodes[node]["for_scc"]["index"] == -1:
                sc_comp.append(strong_connect(node))

        ans = [[]]
        for l0 in sc_comp:
            ans.append(l0)
            break
        # iterate over the list we got back, if there is same list in it, take only the first you see.
        # "make a set"
        for l1 in sc_comp:
            for l2 in ans:
                # if the list we want to add is already appended before.
                if sorted(l1) == sorted(l2):
                    break
                else:
                    ans.append(l1)

        return ans

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        """
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

    def dfs(self, visited, node_id: int):
        if node_id not in visited:
            visited.add(node_id)
            for neighbor in self.graph.edges["From"][node_id]:
                self.dfs(visited, neighbor)

    def dijkstra(self, src_node, dest_node, ):
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
