import numpy as np

def countFAS(G, s):
  return np.tril(G.m[s][:,s]).sum()