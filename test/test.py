def test(arcs, n, k):
  G = Graph.fromArcs(n, arcs)
  Gp, kp = KernelizeFAST(G, k)
  np = Gp.m.shape[0]
  return np, kp

# ass[0] : kernelization outputs a trivial yes instance.
# ass[1] : kernelization outputs a trivial no instance.
# ass[2] : kernelization outputs a significantly smaller instance that you have to solve later.
# ass[3] : kernelization does not make much progress (already a kernel).
with open('test.npy', 'rb') as f:
  n,k = np.load(f)
  ass = [np.load(f) for _ in range(4)]
  for arcs in ass[:-1]:
    print(test(arcs, n, k))