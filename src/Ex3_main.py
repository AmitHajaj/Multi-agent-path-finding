from DiGraph import DiGraph
from GraphAlgo import GraphAlgo
import numpy as np
import matplotlib.pyplot as plt

def check():
    """
    Graph: |V|=4 , |E|=5
    {0: 0: |edges out| 1 |edges in| 1, 1: 1: |edges out| 3 |edges in| 1, 2: 2: |edges out| 1 |edges in| 1, 3: 3: |edges out| 0 |edges in| 2}
    {0: 1}
    {0: 1.1, 2: 1.3, 3: 10}
    (3.4, [0, 1, 2, 3])
    [[0, 1], [2], [3]]
    (2.8, [0, 1, 3])
    (inf, [])
    2.062180280059253 [1, 10, 7]
    17.693921758901507 [47, 46, 44, 43, 42, 41, 40, 39, 15, 16, 17, 18, 19]
    11.51061380461898 [20, 21, 32, 31, 30, 29, 14, 13, 3, 2]
    inf []
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
    """

    # markers = {'.': 'point',  ',': 'pixel',    'o': 'circle',     'v': 'triangle_down',   '^': 'triangle_up',
    #            '<': 'triangle_left',   '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up', '3': 'tri_left',
    #            '4': 'tri_right', '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star', 'h': 'hexagon1',
    #            'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', '_': 'hline',
    #            'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 1: 'tickright', 2: 'tickup', 3: 'tickdown',
    #            4: 'caretleft', 5: 'caretright', 6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 9: 'caretrightbase',
    #            10: 'caretupbase', 11: 'caretdownbase', 'None': 'nothing', None: 'nothing', ' ': 'nothing',
    #            '': 'nothing'}
    # x1, y1 = [-1, 12], [1, 4]
    # x2, y2 = [1, 10], [3, 2]
    # plt.plot(x1, y1, x2, y2, marker='o')
    # plt.plot(x1, y1, x2, y2, marker='d')
    # plt.show()
    check0()
    check1()
    check2()


def check0():
    """
    This function tests the naming (main methods of the DiGraph class, as defined in GraphInterface.
    :return:
    """
    g = DiGraph()  # creates an empty directed graph
    for n in range(4):
        g.add_node(n)
    g.add_edge(0, 1, 1)
    g.add_edge(1, 0, 1.1)
    g.add_edge(1, 2, 1.3)
    g.add_edge(2, 3, 1.1)
    g.add_edge(1, 3, 1.9)
    g.remove_edge(1, 3)
    g.add_edge(1, 3, 10)
    print(g)  # prints the __repr__ (func output)
    print(g.get_all_v())  # prints a dict with all the graph's vertices.
    print(g.all_in_edges_of_node(1))
    print(g.all_out_edges_of_node(1))


    #TODO add comperrator to graph in order to fix this text and move it to tester
    #short test
    g_algo = GraphAlgo(g)
    g_algo.plot_graph()
    file1 = '../data/A5'
    file2 = '../data/A6'
    g_algo.save_to_json(file2)
    g_algo.load_from_json(file1)
    g_algo.plot_graph()

    if not g_algo.get_graph() != g:
        print("good")
    else:
        print("sh!t")

    g_algo.load_from_json(file2)
    if g_algo.get_graph() == g:
        print("good")
    else:
        print("sh!t")

    # g_algo = GraphAlgo(g)
    # print(g_algo.shortest_path(0, 3))
    # g_algo.plot_graph()
#
# def plot():
#     barWidth = 0.2
#
#     # set height of bar
#
#     networkX = [0.0,0.0,0.0]
#     java = [0.0,0.0,0.0]
#     python = [0.0,0.0,0.0]
#
#     print("networkX")
#     networkX[0] = float(input("shortest path: "))
#     networkX[1] = float(input("connected components: "))
#
#     print("java")
#     java[0] = float(input("shortest path: "))
#     java[1] = float(input("connected components: "))
#     java[2] = float(input("connected component: "))
#
#
#     print("python")
#     python[0] = float(input("shortest path: "))
#     python[1] = float(input("connected components: "))
#     python[2] = float(input("connected component: "))
#
#     # Set position of bar on X axis
#     r1 = np.arange(len(networkX))
#     r2 = [x + barWidth for x in r1]
#     r3 = [x + barWidth for x in r2]
#
#     # Make the plot
#     plt.bar(r1, networkX, color='blue', width=barWidth, edgecolor='black', label='networkX')
#     plt.bar(r2, java, color='purple', width=barWidth, edgecolor='black', label='java')
#     plt.bar(r3, python, color='orange', width=barWidth, edgecolor='black', label='python')
#
#     # Add xticks on the middle of the group bars
#     plt.xlabel('methods', fontweight='bold')
#     plt.xticks([r + barWidth for r in range(len(networkX))], ['shortest path', 'connected components', 'connected component'])
#
#     # Create legend & Show graphic
#     plt.xlabel("methods")
#     plt.ylabel("run-time in Seconds ")
#     plt.title("avg run-time")
#
#     plt.legend()
#     plt.show()
#
# if _name_ == '_main_':
#     plot()

def check1():
    """
       This function tests the naming (main methods of the GraphAlgo class, as defined in GraphAlgoInterface.
    :return:
    """
    g_algo = GraphAlgo()  # init an empty graph - for the GraphAlgo
    file = "../data/T0.json"
    g_algo.load_from_json(file)  # init a GraphAlgo from a json file
    print(g_algo.connected_components())
    print(g_algo.shortest_path(0, 3))
    print(g_algo.shortest_path(3, 1))
    g_algo.save_to_json(file + '_saved')



def check2():
    """ This function tests the naming, basic testing over A5 json file.
      :return:
      """
    graph = DiGraph()
    for i in range(8):
        graph.add_node(i)

    graph.add_edge(0, 1, 1)
    graph.add_edge(1, 2, 1)
    graph.add_edge(1, 4, 1)
    graph.add_edge(1, 5, 1)
    graph.add_edge(4, 5, 1)
    graph.add_edge(4, 0, 1)
    graph.add_edge(2, 6, 1)
    graph.add_edge(2, 3, 1)
    graph.add_edge(5, 6, 1)
    graph.add_edge(6, 5, 1)
    graph.add_edge(7, 6, 1)
    graph.add_edge(3, 7, 1)
    graph.add_edge(7, 3, 1)
    graph.add_edge(3, 2, 1)

    ga = GraphAlgo(graph)
    print(ga.connected_components())


    # g_algo = GraphAlgo()
    # file = '../data/A5'
    # g_algo.load_from_json(file)
    # g_algo.get_graph().remove_edge(13, 14)
    # g_algo.save_to_json(file + "_edited")
    # dist, path = g_algo.shortest_path(1, 7)
    # print(dist, path)
    # dist, path = g_algo.shortest_path(47, 19)
    # print(dist, path)
    # dist, path = g_algo.shortest_path(20, 2)
    # print(dist, path)
    # dist, path = g_algo.shortest_path(2, 20)
    # print(dist, path)
    # print(g_algo.connected_component(0))
    # print(g_algo.connected_components())
    # g_algo.plot_graph()


if __name__ == '__main__':
    check()
