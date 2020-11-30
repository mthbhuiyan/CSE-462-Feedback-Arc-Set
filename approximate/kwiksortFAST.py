import graph
import numpy as np

def KwikSort(G):
  return KwikSortRec(G.m, G.V())

def KwikSortRec(M, V):
  if V.shape[0] == 0:
    return V
  v = np.random.choice(V)
  VL = np.argwhere(M.T[v][V] == 1).flatten()
  VR = np.argwhere(M[v][V] == 1).flatten()
  return np.concatenate((KwikSortRec(M, VL), [v], KwikSortRec(M, VR)))