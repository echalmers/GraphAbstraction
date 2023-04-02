from hippocluster.algorithms.hippocluster import Hippocluster
from hippocluster.graphs.abstract import RandomWalkGraph
from hippocluster.graphs.lfr import RandomWalkLFR
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import math

LENGTH_RAND = 0.25

class AbstractedGraph:

    def print_all(self, num_iterations, env_node_size):
        plt.clf()

        for i in range(1, num_iterations + 1):
            plt.subplot(2, num_iterations, i)
            node_colors = {node: self.colors[node][::-1] for node in self.graphs[i].nodes}

            nx.draw(
                self.graphs[i],
                with_labels=True,
                node_color=[node_colors.get(node, [0, 0, 0]) for node in self.graphs[i]],
                node_size=100,
            )

            testcolors = {}

            for node in self.graphs[i]:
                for parent in self.get_lower_abstraction_nodes(node, i - 1, 0):
                    testcolors[parent] = self.colors[node][::-1]

            plt.subplot(2, num_iterations, i + num_iterations)

            nx.draw(
                self.graphs[0].G,
                with_labels=False,
                pos=nx.get_node_attributes(self.graphs[0].G, "pos"),
                node_color=[
                    testcolors.get(node, [0, 0, 0]) for node in self.graphs[0].G
                ],
                node_size=env_node_size,
            )

        plt.tight_layout()
        plt.show()
    def get_higher_abstraction_nodes(self, node_num, cur_layer, to_layer) -> list:
        while cur_layer < to_layer:
            node_num = self.assignmentslist[cur_layer][node_num]
            cur_layer += 1
        return node_num

    def get_lower_abstraction_nodes(self, node_num, cur_layer, to_layer) -> list:
        pointsfinal = []
        points = []

        #print(self.assignmentslist[cur_layer-1])

        for node in self.assignmentslist[cur_layer-1]:
            if self.assignmentslist[cur_layer-1][node] == node_num:
                points.append(node)
        #points = {node for node, cluster in self.assignmentslist[cur_layer-1].keys() if cluster == node_num}
        if (cur_layer-1 > to_layer):
            for point in points:
                pointsfinal += (self.get_lower_abstraction_nodes(point, cur_layer - 1, to_layer))
        else:
            pointsfinal = points
        return pointsfinal

    def tree_circle_pos(self, g: nx.DiGraph, root, depth=2):
        pos = {root: (0, 0)}
        angles = {root: 0}

        node_levels = {0: [root]}
        for level in range(1, depth + 1):
            node_levels[level] = []
            for node in node_levels[level - 1]:
                node_levels[level].extend(list(g.successors(node)))

        for level in range(depth, 0, -1):
            radius = 1 / depth * level
            for i in range(len(node_levels[level])):
                node = node_levels[level][i]
                successors = list(g.successors(node))
                if len(successors) > 0:
                    angles[node] = np.mean([angles[child] for child in successors])
                else:
                    angles[node] = 2 * math.pi / len(node_levels[level]) * i
                pos[node] = (radius * math.cos(angles[node]), radius * math.sin(angles[node]))

        return pos

    def __init__(self, graph, iterations):
    
        self.colors = np.random.rand(300, 3)

        num = iterations
        self.n_panels = 2
        self.graphs = []
        self.assignmentslist = []
        self.nclusters = int(graph.G.order()/10)
        self.graphs.insert(0,graph)

        

        for count in range(num):

            hippocluster = Hippocluster(
                lr=0.05,
                n_clusters=self.nclusters,
                batch_size=500,
                max_len=int(math.ceil(graph.G.order() / self.nclusters * (1+LENGTH_RAND))),
                min_len=int(math.ceil(graph.G.order() / self.nclusters * (1-LENGTH_RAND))),
                n_walks=10*graph.G.order(),
                drop_threshold=0.00001,
                n_jobs=1
            )
            print(type(graph.G))
            hippocluster.fit(graph)
            assignments = hippocluster.get_assignments(graph)


            # instantiate Hippocluster object
            # hippocluster = Hippocluster(
            #     n_clusters=self.nclusters,
            #     drop_threshold=0.001
            # )
            #
            # for step in range(1000):
            #
            #     # get a batch of random walks
            #
            #     walks = [
            #         set(graph.unweighted_random_walk(length=random.randint(int(self.nclusters/2) + 2, self.nclusters+2)))
            #         for _ in range(self.nclusters*5 if step == 0 else self.nclusters)
            #     ]
            #
            #     # update the clustering
            #     hippocluster.update(walks)
            #     assignments = hippocluster.get_assignments(graph)



            self.assignmentslist.insert(count, assignments)
     
            newG = nx.DiGraph()
            for nodenum in set(assignments.values()):
                newG.add_node(nodenum)

            for edge in graph.G.edges:
                cluster1 = assignments.get(edge[0])
                cluster2 = assignments.get(edge[1])

                if(cluster1 != cluster2 or edge[0] == edge[1]):
                    if newG.has_edge(cluster1, cluster2):
                        test = newG[cluster1][cluster2]['weight']
                        newG[cluster1][cluster2]['weight'] = test + graph.G[edge[0]][edge[1]]['weight']
                    else:
                        if(cluster1 != None and cluster2 != None):
                            newG.add_edge(cluster1, cluster2, weight = 0)
                            test = newG[cluster1][cluster2]['weight']
                            newG[cluster1][cluster2]['weight'] = test + graph.G[edge[0]][edge[1]]['weight']

            self.graphs.insert(count+1, newG)
            
            self.nclusters = int(len(set(assignments.values()))/2)
            graph = RandomWalkGraph(newG, walk_type='diffusion')


    def printgraphs(self, g1, g2):

        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
        plt.clf()
        
        plt.subplot(1, 2, 2)

        root = self.get_higher_abstraction_nodes((0, 0, 0, 0, 0, 0, 0, 0, 0), 0, g2)

        node_colors={node: self.colors[node][::-1] for node in self.graphs[g2].nodes}

        pos = graphviz_layout(self.graphs[g2], prog="twopi")
        nx.draw(self.graphs[g2],
                pos=pos,
                with_labels=False,
                node_color=[node_colors.get(node, [0, 0, 0]) for node in self.graphs[g2]],
                node_size=400)
        # add edge weights
        labels = nx.get_edge_attributes(self.graphs[g2], 'weight')
        nx.draw_networkx_edge_labels(self.graphs[g2], pos, edge_labels=labels)


        testcolors={}
            
        for node in self.graphs[g2]:
            for parent in self.get_lower_abstraction_nodes(node, g2, g1):
                #print(parent)
                testcolors[parent] = self.colors[node][::-1]
        
       
        
        plt.subplot(1, 2, 1)


        if type(self.graphs[g1]) is RandomWalkGraph:

            nx.draw(self.graphs[g1].G, with_labels=False,
            pos = graphviz_layout(self.graphs[g1].G, prog="twopi"),
            node_color=[testcolors.get(node, [0, 0, 0]) for node in self.graphs[g1].G],
            node_size=50)
        else:
            nx.draw(self.graphs[g1], with_labels=False,
            pos=graphviz_layout(self.graphs[g1], prog="twopi"),
            node_color=[testcolors.get(node, [0, 0, 0]) for node in self.graphs[g1]],
            node_size=400)
        
        plt.show()
