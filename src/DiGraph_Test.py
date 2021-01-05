from DiGraph import DiGraph
from GraphAlgo import GraphAlgo
import heapq
import unittest


class test:

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
        unvisited = self.graph.nodes.keys()

        # make a minheap for the unvisited nodes.
        heapq.heapify(unvisited)

        # set all nodes distance to infinity,
        # and all prev to None.
        for node in self.graph.nodes:
            node["tag"] = 99999999
            node["prev"] = None

        # set the src_node to zero.
        self.graph.nodes[src_node]["tag"] = 0

        while len(unvisited) > 0:
            # current node is the neighbor with the minimum weight.
            curr = heapq.heappop(unvisited)

            # take current node neighbors.
            v_neighbors = {k:v for (k,v) in self.graph.edges["From"].items() if curr == k}
            for neighbor in v_neighbors.keys():
                if self.graph.nodes[curr]["tag"] + self.graph.edges["From"][curr][neighbor] < self.graph.nodes[neighbor]["tag"]:
                    # update the distance
                    self.graph.nodes[neighbor]["tag"] = self.graph.nodes[curr]["tag"] + self.graph.edges["From"][curr][neighbor]
                    # update parent node
                    self.graph.nodes[neighbor]["perv"] = curr

            # move the smallest element to the top of the hep for next iteration.
            heapq.heapify(unvisited)

def check(self):
    graph = DiGraph()
    ga = GraphAlgo(graph)

    for n in range(9):
        graph.add_node(n)
        graph.add_node(n+1)
            
        graph.add_edge(n, n+1, n+1)
        graph.add_edge(n+1, n, n)

    self.dijkstra(0, 5)
    print(self.assertEqual(15, graph.nodes[5]["tag"]))


if __name__ == '__main__':
    test()


