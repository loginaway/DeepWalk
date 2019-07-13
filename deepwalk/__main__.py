# coding: utf-8

from DeepWalk import DeepWalk
from Graphtools import Graphtools
import pickle

if __name__=='__main__':
    gtools=Graphtools()
    # G=gtools.fromText('doubanUser.txt')
    # print(G)
    with open('deepwalk.model', 'rb') as f:
        deepwalk=pickle.load(f)

    # deepwalk=DeepWalk(G, 2, embedding_size=64, walks_per_vertex=2, walk_length=8)
    deepwalk(unit_iter=25, multiProcess=4, showLoss=False)
    deepwalk.save(filename='deepwalk.model')
