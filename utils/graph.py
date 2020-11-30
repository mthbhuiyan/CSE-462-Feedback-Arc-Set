import numpy as np
from numba import jitclass, int32

@jitclass([('m',int32[:,:])])
class Graph(object):
  def __init__(self, m: np.ndarray):
    self.m = m
  
  def reverse(self, u, v, copy):
    g = Graph(self.m.copy()) if copy else self
    m = g.m
    m[u,v], m[v,u] = m[v,u], m[u,v]
    return g

  def V(self):
    return np.arange(self.m.shape[0])
  
  def A(self):
    return np.argwhere(self.m == 1)

  def c(self, v, S):
    return self.m[v][S].sum()

  def fromArcs(n: int, arcs: np.ndarray):
    m = np.zeros((n, n), dtype=np.int32)

    for i in range(arcs.shape[0]):
      m[arcs[i,0], arcs[i,1]] = 1
    
    return Graph(m)