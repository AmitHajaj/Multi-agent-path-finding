import unittest
from DiGraph import DiGraph


class TestDiGraph(unittest.TestCase):
    def test_add_node(self):
        graph = DiGraph()
        self.assertTrue(graph.add_node(0))
        self.assertTrue(graph.add_node(1))
        self.assertTrue(graph.add_node(2))
        self.assertFalse(graph.add_node(0))
        self.assertEqual(graph.v_size(), 3)

    def test_v_size(self):
        graph = DiGraph()
        self.assertEqual(graph.v_size(), 0)
        for i in range(15):
            graph.add_node(i)
        self.assertEqual(graph.v_size(), 15)
        graph.add_node(10)
        self.assertEqual(graph.v_size(), 15)

    def test_e_size(self):
        graph = DiGraph()
        self.assertEqual(graph.e_size(), 0)
        for i in range(15):
            graph.add_node(i)

        for i in range(14):
            graph.add_edge(i, i+1, 1)

        self.assertEqual(graph.e_size(), 14)
        # edge from node to it self.
        graph.add_edge(1, 1, 1)
        self.assertEqual(graph.e_size(), 14)
        # negative weight edge
        graph.add_edge(1, 5, -1)
        self.assertEqual(graph.e_size(), 14)

    def test_all_in_edges_of_node(self):
        graph = DiGraph()
        for i in range(15):
            graph.add_node(i)

        for i in range(14):
            graph.add_edge(i, i+1, 1)
        for n in range(13):
            self.assertTrue(n in graph.edges["To"][n + 1])

    def test_all_out_edges_of_node(self):
        graph = DiGraph()
        for i in range(15):
            graph.add_node(i)

        for i in range(14):
            graph.add_edge(i, i+1, 1)
        for n in range(13):
            self.assertTrue(n+1 in graph.edges["From"][n])

    def test_get_mc(self):
        graph = DiGraph()
        for i in range(15):
            graph.add_node(i)
        self.assertEqual(graph.get_mc(), 15)
        for i in range(14):
            graph.add_edge(i, i+1, 1)
        self.assertEqual(graph.get_mc(), 29)

        graph.add_node(1)
        self.assertEqual(graph.get_mc(), 29)

        graph.remove_node(1)
        self.assertEqual(graph.get_mc(), 30)

    def test_add_edge(self):
        graph = DiGraph()
        for i in range(15):
            graph.add_node(i)
        self.assertEqual(graph.get_mc(), 15)
        for i in range(14):
            graph.add_edge(i, i+1, 1)
        self.assertEqual(graph.e_size(), 14)
        self.assertFalse(graph.add_edge(0, -2, 0.9))

        self.assertTrue(graph.add_edge(0, 2, 1.3))
        self.assertEqual(graph.e_size(), 15)

    def test_remove_node(self):
        graph = DiGraph()
        for i in range(15):
            graph.add_node(i)
        self.assertEqual(graph.get_mc(), 15)
        self.assertFalse(graph.remove_node(17))
        self.assertTrue(graph.remove_node(2))
        self.assertEqual(graph.v_size(), 14)

    def test_remove_edge(self):
        graph = DiGraph()
        for i in range(15):
            graph.add_node(i)
        self.assertEqual(graph.get_mc(), 15)
        for i in range(14):
            graph.add_edge(i, i+1, 1)

        self.assertFalse(graph.remove_edge(0, -2))
        self.assertTrue(graph.remove_edge(0, 1))
        self.assertEqual(graph.e_size(), 13)

    def test___eq__(self):
        graph = DiGraph()
        for i in range(15):
            graph.add_node(i)

        graph1 = DiGraph()
        for i in range(15):
            graph1.add_node(i)

        self.assertTrue(graph == graph1)
        graph.add_node(16)
        self.assertFalse(graph == graph1)


if __name__ == '__main__':
    unittest.main()
