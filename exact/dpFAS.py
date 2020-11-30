import graph
import numpy as np
from numba import njit

@njit
def vertices(S_bit):
  v = 0
  v_bit = 1

  while v_bit <= S_bit:
    if v_bit & S_bit > 0:
      yield v, v_bit
    
    v += 1
    v_bit <<= 1

# optimization version
@njit
def OptFAS(G):
  n = len(G.V())
  nS = 1 << n
  OPT = np.full(nS, np.iinfo(np.int8).max)
  OPT[0] = 0

  for S_bit in range(1, nS):
    vbS = list(vertices(S_bit))
    S = np.array([v for v,_ in vbS])

    for v, v_bit in vbS:
      OPT[S_bit] = min(OPT[S_bit], OPT[S_bit^v_bit] + G.c(v,S))
  
  return OPT[-1]

# decision version
# recursive step
@njit
def IsFAS_Rec(S_bit, k, G, memo):
  m_bit, m_val = memo

  if m_bit[S_bit] == 0:
    return m_val[S_bit] <= k

  vbS = list(vertices(m_bit[S_bit]))
  S = np.array([v for v,_ in vbS])
  
  for v, v_bit in vbS:
    m_bit[S_bit] ^= v_bit
    c = G.c(v, S)
    St_bit = S_bit ^ v_bit
    if IsFAS_Rec(St_bit, k - c, G, memo):
      return True
    m_val[S_bit] = min(m_val[S_bit], m_val[St_bit] + c)
  
  return False

# top-down approach
@njit
def IsFAS_TD(G, k):
  if k < 0:
    return False
  
  n = len(G.V())
  nS = 1 << n
  m_bit = np.arange(nS)
  m_val = np.full(nS, np.iinfo(np.int8).max)
  m_val[0] = 0

  memo = (m_bit, m_val)
  
  return IsFAS_Rec(nS-1, k, G, memo)

# bottom-up approach
@njit
def IsFAS_BU(G, k):
  if k < 0:
    return False
  
  n = len(G.V())
  nS = 1 << n
  OPT = np.full(nS, np.iinfo(np.int8).max)
  OPT[0] = 0

  for S_bit in range(1, nS):
    vbS = list(vertices(S_bit))
    S = np.array([v for v,_ in vbS])

    for v, v_bit in vbS:
      OPT[S_bit] = min(OPT[S_bit], OPT[S_bit ^ v_bit] + G.c(v, S))

    if OPT[S_bit] > k:
      return False
  
  return True




# n = 4
# arcs = np.array([[0,1],[1,2],[2,3],[3,0]],dtype=np.int32)

# g = Graph.fromArcs(n, arcs)
# print(g.m)

# print(OptFAS(g))


# M = np.array([[0,1,0,1,0],[0,0,1,0,1],[1,0,0,0,0],[0,1,1,0,0],[1,0,1,1,0]],dtype=np.int32)
# g = Graph(M)
# n = 4
# arcs = np.array([[0,1],[1,2],[2,3],[3,0]],dtype=np.int32)

# g = Graph.fromArcs(n, arcs)
# print(g.m)

# print(IsFAS_TD(g, 1))