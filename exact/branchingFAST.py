import graph
import numpy as np
from numba import njit

@njit
def IsFAST(G, k):
  if k < 0:
    return False
  
  n = G.m.shape[0]
  vs = None
  for u in range(0,n-2):
    for v in range(u+1,n-1):
      for w in range(v+1,n):
        if G.m[u,v] == G.m[v,w] == G.m[w,u]:
          vs = [u,v,w]
          break
  
  if vs is None:
    return True
  
  for u, v in zip(vs,np.roll(vs,-1)):
    isFAST = IsFAST(G.reverse(u,v,False), k-1)
    G.reverse(u,v,False)
    if isFAST:
      return True
  
  return False
  

def IsFASTKernelized(G, k):
  G, k = KernelizeFAST(G, k)
  return IsFAST(G, k)