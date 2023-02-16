from hippocluster.graphs.abstract import RandomWalkGraph
from absgraph import AbstractedGraph
import pickle

# filename = "../sample_data/T_tograph_empty_gridworld.pkl"
filename = "../GraphAbstraction/sample_data/T_tograph_walls_gridworld.pkl"

with open(filename, "rb") as f:
    G = pickle.load(f)

absgraph = AbstractedGraph(RandomWalkGraph(G),4)
#absgraph.get_lower_abstraction_nodes(1,1,0)
absgraph.print_all(4,8)