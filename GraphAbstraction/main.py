from hippocluster.graphs.abstract import RandomWalkGraph
from absgraph import AbstractedGraph
import pickle
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


def state_string(obs) -> str:
    """
    returns an ascii printout of the tic-tac-toe board
    :param obs: a board state (as returned by step and reset). defaults to current state
    :return: string
    """
    string = f'{obs[0]}│{obs[1]}│{obs[2]}\n' \
             f'─┼─┼─\n{obs[3]}│{obs[4]}│{obs[5]}\n' \
             f'─┼─┼─\n{obs[6]}│{obs[7]}│{obs[8]}\n'
    string = string.replace('0', ' ').replace('-1', 'O').replace('1', 'X')
    return string

def state_mat(obs) -> str:
    mat = [[obs[0],obs[1],obs[2]],[obs[3],obs[4],obs[5]],[obs[6],obs[7],obs[8]]]

    return np.matrix(mat)

def state_arr(obs) -> str:
    arr = [[obs[0],obs[1],obs[2]],[obs[3],obs[4],obs[5]],[obs[6],obs[7],obs[8]]]

    return arr

def setpos(tree, node, depth, offset):
    print(len(list(tree.successors(node))))
    if len(list(tree.successors(node))) > 0:
        for i in range(len(list(tree.successors(node)))):
            pos[list(tree.successors(node))[i]] = (len(list(tree.successors(node)))*offset + i, depth)
            setpos(tree, list(tree.successors(node))[i], depth+1, i)
    return

# filename = "../sample_data/T_tograph_empty_gridworld.pkl"
filename = "../sample_data/tic.pkl"

with open(filename, "rb") as f:
    G = pickle.load(f)

# print(type(G))
# tree = nx.dfs_tree(G, source=(0, 0, 0, 0, 0, 0, 0, 0, 0), depth_limit=3)
#
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1

# root = (0, 0, 0, 0, 0, 0, 0, 0, 0)
#
# depth = 0
#
# pos = {}
#
#
# curnode = root
#
#
# pos[root] = (0,0)
# pos = {}
# setpos(tree,root,1, 0)

#nx.draw(tree,pos=pos)
#plt.show()

num = 3

absgraph = AbstractedGraph(RandomWalkGraph(G, walk_type='diffusion'),num)
print(absgraph.graphs[num])

Gr = absgraph.graphs[num]
GrAdd = []
for node in Gr:
    stateAdd = (0,0,0,0,0,0,0,0,0)
    for state in absgraph.get_lower_abstraction_nodes(node,num, 0):
        stateAdd = tuple(map(lambda i, j: i + j, stateAdd, state))
    GrAdd.append((stateAdd, node))

posWin = []
negWin = []
tie = []
def add_board(list, board, num):
    if board not in list:
        list.append((board, num))

for tic, num in GrAdd:
    tic = state_arr(tic)
    for tac in [tic, np.transpose(tic)]:
        for row in tac:
            if row[0] < 0 and row [1] < 0 and row[2] < 0:
                add_board(negWin, tic, num)
            if row[0] > 0 and row [1] > 0 and row[2] > 0:
                add_board(posWin, tic, num)
    for toe in [tic, np.transpose(tic)]:
        if toe[0][0] < 0 and toe[1][1] < 0 and toe[2][2] < 0:
            add_board(negWin, tic, num)
        if toe[0][0] > 0 and toe[1][1] > 0 and toe[2][2] > 0:
            add_board(posWin, tic, num)
    if tic not in posWin or tic not in negWin:
        add_board(tie, tic, num)

print("Positive Wins\n")

i = 1
print(len(posWin))

for pos,num in posWin:
    plt.subplot(math.ceil(len(posWin)/4),4,i)
    print(num, '\n')
    plt.imshow((pos), cmap='coolwarm')
    i += 1


plt.show()
print('\n', "Negative Wins", '\n')

for neg in negWin:
    print(np.matrix(neg), '\n')

print('\n',"Tie", '\n')

for ti in tie:
    print(np.matrix(ti))








