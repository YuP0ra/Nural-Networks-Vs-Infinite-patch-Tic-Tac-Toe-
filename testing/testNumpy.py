import numpy as np

# 10M
length = 10000000

tons_of_zeros = np.zeros(length)


tons_of_zeros.tofile('uncompressed')
np.savez_compressed(file='compressed', data=tons_of_zeros)
