import numpy as np
import csv

ns = np.load('data/neg_sequences.npy')
ps = np.load('data/neg_sequences.npy')

sz_ns = ns.size()
sz_ps = ps.size()
print()
np.savetxt("neg_sequences.csv", ns)
len_ns = len(ns)
len_ps = len(ps)
