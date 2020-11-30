import graph
import numpy as np

def KernelizeFAST(G, k):
  M = G.m
  t = 1
  while k >= 0 and t > 0:
    # Na: number of triangles through (u, v)
    Na = (M.T @ M.T) * M
    # Nv: number of triangles through v
    Nv = Na.sum(axis=1)
 
    ## reduction 2
    ix = np.argwhere(Nv == 0)
    M = np.delete(np.delete(M, ix, 0), ix, 1)
 
    Na = np.delete(np.delete(Na, ix, 0), ix, 1)
 
    ## reduction 1
    ix = np.argwhere(Na > k)
    u, v = ix[:,0], ix[:,1]
    M[u,v], M[v,u] = M[v,u], M[u,v]
    t = ix.shape[0]
    k -= t
  
  n = M.shape[0]
  if k**2 + 2*k < n:
    M = np.empty(shape=(0,0), dtype=M.dtype)
    k = -1
 
  return Graph(M), k
def KernelizeFAST(G, k):
  M = G.m
  t = 1
  while k >= 0 and t > 0:
    # Na: number of triangles through (u, v)
    Na = (M.T @ M.T) * M
    # Nv: number of triangles through v
    Nv = Na.sum(axis=1)
 
    ## reduction 2
    ix = np.argwhere(Nv == 0)
    M = np.delete(np.delete(M, ix, 0), ix, 1)
 
    Na = np.delete(np.delete(Na, ix, 0), ix, 1)
 
    ## reduction 1
    ix = np.argwhere(Na > k)
    u, v = ix[:,0], ix[:,1]
    M[u,v], M[v,u] = M[v,u], M[u,v]
    t = ix.shape[0]
    k -= t
  
  n = M.shape[0]
  if k**2 + 2*k < n:
    M = np.empty(shape=(0,0), dtype=M.dtype)
    k = -1
 
  return Graph(M), k