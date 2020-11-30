import numpy as np

def allRandTournament(n):
  m = np.random.randint(2, size=(n,n), dtype=np.int32)
  t = np.triu(np.ones((n,n), dtype=np.int32), 1)
  m = m * t
  m = m + ((1 - m) * t).T
  return m

def kRandTournament(n, k):
  x, y = np.triu_indices(n, 1)
  c = x.shape[0]
  i = np.random.choice(c, k, replace=False)
  x[i], y[i] = y[i], x[i]
  m = np.zeros((n,n), dtype=np.int32)
  m[x,y] = 1
  return m

def interleaveTournaments(ti, to, loc=0):
  ni = ti.shape[0]
  no = to.shape[0]
  loc = min(loc, no)
  n = no + ni

  m = np.triu(np.ones((n,n), dtype=np.int32), 1)

  m[loc:loc+ni,loc:loc+ni] = ti

  m[:loc,:loc] = to[:loc,:loc]
  m[:loc,loc+ni:] = to[:loc,loc:]
  m[loc+ni:,:loc] = to[loc:,:loc]
  m[loc+ni:,loc+ni:] = to[loc:,loc:]

  return m

# np.random.seed(0)
# kRandTournament(5, 2)

# np.random.seed(0)
# allRandTournament(5)

# np.random.seed(0)

# t0 = allRandTournament(5)
# t1 = kRandTournament(5, 3)
# t = interleaveTournaments(t0, t1, loc=2)