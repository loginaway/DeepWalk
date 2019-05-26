# coding: utf-8

from SkipGram import SkipGram
from Graphtools import Graphtools
import multiprocessing as mp
from time import sleep
import numpy as np
import pickle


class DeepWalk(object):
    
    def __init__(self, G, window_size, embedding_size, walks_per_vertex, walk_length):
        self.window_size=window_size
        self.embedding_size=embedding_size
        self.walks_per_vertex=walks_per_vertex
        self.walk_length=walk_length
        self.whole_size=len(G.nodes)
        
        self.G=G
        self.G.ndata['X']=np.random.rand(self.whole_size, embedding_size)
        print(G.ndata['X'])
        self.skipgram=SkipGram(window_size, embedding_size, self.whole_size, 0.01, Theta=G.ndata['X'])

    def __call__(self, unit_iter=100, multiProcess=1, showLoss=False):
        '''
        Run DeepWalk algorithm.
        '''
        if multiProcess==1:
            for i in range(self.walks_per_vertex):
                nodes=G.nodes().numpy()
                np.random.shuffle(nodes)
                for v in nodes:
                    print('-----------', v, '------------')
                    walk=self.RandomWalk(v, In_Out='Both')
                    self.skipgram.walk_train(walk, unit_iter=unit_iter,  showLoss=showLoss)
            print('Deepwalk finished.')
        elif multiProcess>1 and isinstance(multiProcess, int):
            proc=[mp.Process(target=self.__call__, args=(unit_iter, 1, showLoss)) \
                    for i in range(multiProcess)]
            for i in range(multiProcess):
                proc[i].start()
                sleep(1)
            for i in range(multiProcess):
                proc[i].join()
            print('Multi-deepwalk finished. Process:', multiProcess)

    def RandomWalk(self, vi_index, In_Out='Both'):
        '''
        Return a walk on graph G from vi, 
        controlled by parameters window_size, walks_per_vertex and walk_length.
        ***** DO NOT distinguish in_edges from out_edges by default *****
        In_Out='In': In edges only; 'Out': Out edges only.
        '''
        length=0
        current=vi_index
        walk=np.empty(self.walk_length, dtype=int)
        if In_Out=='Both':
            while length<self.walk_length:
                walk[length]=current
                pool_out=self.G.out_edges(current)[1].numpy()
                pool_in=self.G.in_edges(current)[0].numpy()
                pool=np.concatenate((pool_in, pool_out))
                current=np.random.choice(pool)
                length+=1
        elif In_Out=='In':
            while length<self.walk_length:
                walk[length]=current
                pool=self.G.in_edges(current)[0].numpy()
                current=np.random.choice(pool)
                length+=1
        elif In_Out=='Out':
            while length<self.walk_length:
                walk[length]=current
                pool=self.G.out_edges(current)[1].numpy()
                current=np.random.choice(pool)
                length+=1
        return walk

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


if __name__=='__main__':
    gtools=Graphtools()
    G=gtools.fromText('p2p-Gnutella08.txt')
    # with open('deepwalk.model', 'rb') as f:
    #     deepwalk=pickle.load(f)
    deepwalk=DeepWalk(G, 3, 30, 3, 10)
    deepwalk(unit_iter=100, multiProcess=4, showLoss=False)
    deepwalk.save()

    
    # deepwalk=DeepWalk(G, 3, 30, 3, 10)
    # deepwalk.save()
    # deepwalk(unit_iter=100)
    print('Done.')
