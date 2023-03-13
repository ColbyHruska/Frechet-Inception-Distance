import numpy as np
import os
import sys

import features
import imageloader ; imageloader.set_path(sys.argv[1])

batch_size = 500
feature_arr = features.batch_features(imageloader.get_batch(0, batch_size), False)
rem = sys.argv[2]
rem -= batch_size
i = 0
while rem > 0:
    n = min(rem, batch_size)
    try:
        feature_arr = np.concatenate((feature_arr, features.batch_features(imageloader.get_batch(i, n), False)), axis=0)
    except(imageloader.OutOfImages):
        break
    i += n
    rem -= n

mu, sigma = features.feature_distribution(feature_arr)

dir = os.path.dirname(__file__)
mu_dir = os.path.join(dir, 'mu.npy')
sigma_dir = os.path.join(dir, 'sigma.npy')

def try_del(dir):
    if os.path.exists(dir):
        os.remove(dir)

try_del(mu_dir)
try_del(sigma_dir)

np.save(mu_dir, mu)
np.save(sigma_dir, sigma)