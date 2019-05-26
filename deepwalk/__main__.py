# coding: utf-8

from DeepWalk import DeepWalk
from Graphtools import Graphtools

if __name__=='__main__':
    deepwalk=DeepWalk(None, None, None, None, None, load_file_address='deepwalk.model')
    deepwalk(100, 4, False)
    # deepwalk.save('deepwalk.model')
