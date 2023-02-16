from hippocluster.graphs.abstract import RandomWalkGraph
from absgraph import AbstractedGraph
import pickle

# filename = "T_tograph_empty_gridworld.pkl"
filename = "T_tograph_walls_gridworld.pkl"

with open(filename, "rb") as f:
    G = pickle.load(f)

absgraph = AbstractedGraph(RandomWalkGraph(G),1)
#absgraph.get_lower_abstraction_nodes(1,1,0)
absgraph.print_all(1,10)