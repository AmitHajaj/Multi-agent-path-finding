import os
import unittest
import random
import time
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from DiGraph import DiGraph
from GraphAlgo import GraphAlgo
from itertools import islice


class TestGraphAlgo(unittest.TestCase):

    # TODO check seeds
    def test_get_graph(self):
        #  create graphs
        emptyGraph = DiGraph()
        g = (create_graph(10, 40))[0]
        ga = GraphAlgo()

        self.assertEqual(ga.get_graph(), emptyGraph, msg='GraphAlgo should contain empty graph when first initialized')
        ga.graph = emptyGraph
        self.assertEqual(ga.get_graph(), emptyGraph, msg='GraphAlgo should contain empty graph')

        ga = GraphAlgo(g)

        self.assertNotEqual(ga.get_graph(), emptyGraph, msg='GraphAlgo should contain different graph')
        self.assertEqual(ga.get_graph(), g, msg='GraphAlgo should contain g2')

        # TODO check init wrong types to GraphAlgo

    def test_load_and_save_(self):
        # create graphs
        g1 = (create_graph(10, 45))[0]
        g2 = (create_graph(9, 10))[0]
        ga = GraphAlgo(g1)

        # try save wrong types
        ga.graph = None
        b = ga.save_to_json('g')
        self.assertFalse(b)
        # TODO is assert right

        # use pre-existing path
        filePath = '../data/g1'
        saveAndLoad(filePath, g1, ga)
        saveAndLoad(filePath, g2, ga)

        self.assertEqual(ga.get_graph(), g2, msg='rewrite a json file should`nt be a problem')

        # save and load graphs with missing elements (position, nodes, edges):

        #  no position
        noPos_g = (create_graph(10, 10))[0]
        filePath = '../data/g2'
        saveAndLoad(filePath, noPos_g, ga)

        self.assertEqual(ga.get_graph(), noPos_g, msg='graph should`nt have change after save and load')

        #  no nodes (emptyGraph)
        noNodes_g = (create_graph(0))[0]
        filePath = '../data/g3'
        saveAndLoad(filePath, noNodes_g, ga)

        self.assertEqual(ga.get_graph(), noNodes_g, msg='graph should`nt have change after save and load')

        #  no edges
        noEdges_g = (create_graph(10))[0]
        filePath = '../data/g4'
        saveAndLoad(filePath, noEdges_g, ga)

        self.assertEqual(ga.get_graph(), noEdges_g, msg='graph should`nt have change after save and load')

    def test_shortest_path(self):
        # get graph and path
        g, pathTuple = (create_graph(10, 20, pathLen=5))

        pathList = pathTuple[0]
        pathDist = pathTuple[1]
        # check algorithm with valid input
        ga = GraphAlgo(g)
        src = pathList[0]
        dest = pathList[-1]
        dist, path = ga.shortest_path(src, dest)

        self.assertEqual(dist, pathDist, msg='wrong path length')
        self.assertEqual(path, pathList, msg='path found in shortest_path is wrong')

        # check shortest_path when there is no path
        pathList = []
        newKey = g.v_size()
        g.add_node(newKey)
        dist, path = ga.shortest_path(src, newKey)

        self.assertEqual(dist, float('inf'), msg='when there is no path, the path length should be float(`inf`) => '
                                                 'infinity')
        self.assertEqual(path, pathList, msg='when there is no path, path should be an empty list => []')

        # check shortest_path when src doesn't exist
        nonExistingNode = newKey + 1
        dist, path = ga.shortest_path(nonExistingNode, 0)

        self.assertEqual(dist, float('inf'), msg='when source doesn`t exists, the path length should be float('
                                                 '`inf`) => infinity')
        self.assertEqual(path, pathList, msg='when source doesn`t exists, path should be an empty list => []')

        # check shortest_path when dest doesn't exist
        dist, path = ga.shortest_path(0, nonExistingNode)

        self.assertEqual(dist, float('inf'), msg='when destination doesn`t exists, the path length should be float('
                                                 '`inf`) => infinity')
        self.assertEqual(path, pathList, msg='when destination doesn`t exists, path should be an empty list => []')

    def test_connected_component(self):
        g = create_graph(30)[0]
        emptyGraph = DiGraph()

        keysList = g.get_all_v().keys()
        chunkSizes = [2, 5, 14, 9]
        itr = iter(keysList)
        chunkList = [list(islice(itr, 0, ele)) for ele in chunkSizes]

        for chunk in chunkList:
            i = 0
            while i < len(chunk):
                j = i + 1
                while j < len(chunk):
                    w = random.uniform(0, 1)
                    g.add_edge(chunk[i], chunk[j], w)
                    g.add_edge(chunk[j], chunk[i], w)
                    a = chunk[i]
                    b = chunk[j]
                    j += 1
                i += 1

        # check algorithm with valid input
        ga = GraphAlgo(g)
        scc = ga.connected_component(chunkList[1][1])
        b = sorted(scc) == sorted(chunkList[1])
        self.assertTrue(b, msg='wrong component was found')

        components = ga.connected_components()

        b = False
        if len(components) == len(chunkList):
            # tempComponent = list(components)  # make a mutable copy
            tempChunkList = []
            for element in chunkList:
                tempChunkList.append(sorted(element))

            counter = 0
            for element in components:
                element = sorted(element)
                if element in tempChunkList:
                    counter += 1

            if counter == len(chunkList):
                b = True

        self.assertTrue(b, msg='wrong components was found')

        # check component of non-existing node
        scc = ga.connected_component(40)
        self.assertEqual(scc, [], msg='if node dosn`t exists the function should return empty list =>[]')

        # check component if graph is None
        ga.graph = emptyGraph
        scc = ga.connected_component(1)
        self.assertEqual(scc, [], msg='if node dosnt exists the function should return empty list =>[]')

    def test_plot_graph(self):
        # check valid input
        g = create_graph(10, 20, setPos=True)[0]
        ga = GraphAlgo(g)

        ga.plot_graph(setTimer=True)

        # --- plot a graph with missing elements (position, nodes, edges): ---

        #  --no positions--
        ga.graph = (create_graph(10, 10))[0]
        ga.plot_graph(setTimer=True)

        #  --no nodes (emptyGraph)--
        ga.graph = (create_graph(0))[0]
        ga.plot_graph(setTimer=True)

        # --no edges--
        ga.graph = (create_graph(10))[0]
        ga.plot_graph(setTimer=True)

    def test_time(self):
        timeTable = {'SP': {}, 'SCC': {}, 'netWorkX': {}, 'plt': {}}

        # load graph's folder
        ga = GraphAlgo()
        folderPath = '../TestGraphs'
        directory = os.fsencode(folderPath)

        # iterate graphs
        for file in os.listdir(directory):
            # load graph
            fileName = os.fsdecode(file)
            ga.load_from_json('../TestGraphs/' + fileName)
            nSize = ga.get_graph().v_size()

            if True:

                start = time.time()
                ga.plot_graph()
                mid = time.time()
                networkxBuild(ga)
                end = time.time()

                # update timeTable
                plt_graph_time = end - mid
                networkx_time = mid - start
                timeTable['plt'].update({fileName: plt_graph_time})
                timeTable['netWorkX'].update({fileName: networkx_time})

                # check shortest_path
                start = time.time()
                ga.shortest_path(0, nSize / 2)
                end = time.time()

                # update timeTable
                shortest_path_Time = end - start
                timeTable['SP'].update({fileName: shortest_path_Time})

                # check connected_components
                start = time.time()
                ga.connected_components()
                end = time.time()

                # update timeTable
                shortest_path_Time = end - start
                timeTable['SCC'].update({fileName: shortest_path_Time})
        filePath = '../data/test'
        with open(filePath, 'w') as graphJson:
            json.dump(timeTable, graphJson, indent=4)


def networkxBuild(ga: GraphAlgo):

    nxG = nx.DiGraph()
    g = ga.get_graph()

    # draw nodes
    nxG.add_nodes_from(ga.nodesDict)

    # draw edge
    nxG.add_edges_from(ga.edgesList)

    nx.draw(nxG)
    plt.show()


def saveAndLoad(filePath: str, g: DiGraph, ga: GraphAlgo) -> None:
    emptyGraph = DiGraph()

    # load graph
    filePath = '../data/g3'
    ga.graph = g
    ga.save_to_json(filePath)

    # init emptyGraph to GraphAlgo and load g
    ga.graph = emptyGraph
    ga.load_from_json(filePath)


def create_graph(nodeSize: int, edgeSize: int = 0, pathLen: int = 0, setPos: bool = False) -> (DiGraph, list):
    random.seed(1)
    g = DiGraph()

    # add nodes
    keyList = range(nodeSize)
    for key in keyList:
        if setPos:
            pos = (random.uniform(32, 35), random.uniform(32, 35), 0)
            g.add_node(key, pos)
        else:
            g.add_node(key)

    # add edges
    while 0 < edgeSize:
        key1 = random.choice(range(nodeSize))
        key2 = random.choice(range(nodeSize))
        if key1 != key2:
            w = random.uniform(0, 1) if pathLen == 0 else random.uniform(pathLen + 1, pathLen + 3)
            g.add_edge(key1, key2, w)
            edgeSize -= 1

    # pave a path in graph
    dist = 0
    pathKeys = []
    if pathLen > 0:
        pathKeys = random.sample(keyList, pathLen)
        i = 0
        while pathLen - 1 > i:
            w = random.uniform(0, 1)
            g.add_edge(pathKeys[i], pathKeys[i + 1], w)
            dist += w
            i += 1

    return g, (pathKeys, dist)


if __name__ == '__main__':
    unittest.main()
