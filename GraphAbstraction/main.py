from hippocluster.graphs.abstract import RandomWalkGraph
from absgraph import AbstractedGraph
import pickle
import time

# filename = "../sample_data/T_tograph_empty_gridworld.pkl"
filename = "../sample_data/T_tograph_walls_gridworld.pkl"

with open(filename, "rb") as f:
    G = pickle.load(f)

start = time.time()
absgraph = AbstractedGraph(RandomWalkGraph(G),4)
print(f'abstraction finished in {time.time() - start: .2f} seconds')

#absgraph.get_lower_abstraction_nodes(1,1,0)
absgraph.print_all(4,8)