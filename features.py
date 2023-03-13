import numpy as np
from scipy import stats
import tensorflow as tf
from scipy.linalg import sqrtm
from skimage.transform import resize
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import os
import math
from keras import backend as K
model = InceptionV3(weights="imagenet", include_top=False, pooling='avg', input_shape=(299,299,3))

def scale(imgs, shape):
	img_list = []
	for img in imgs:
		new_img = resize(img, shape, 0)
		img_list.append(new_img)
	return np.asarray(img_list)

def features(arr, preprocess=True):
	arr = scale(arr, (299,299,3))
	arr = arr.astype('float32')
	if preprocess:
		arr = preprocess_input(arr)

	distribution = model.predict(tf.convert_to_tensor(arr))
	return np.array(distribution)

def batch_features(arr, preprocess=True):
    rem = arr.shape[0]
    m = 100
    distribution = np.array(features(arr[:min(m, rem)], preprocess))
    rem -= m
    i = m
    while rem > 0:
        n = min(m, rem)
        distribution = np.concatenate((distribution, np.array(features(arr[i:i+n], preprocess))), axis=0)
        rem -= n
        i += n
        K.clear_session()
    return distribution

def find_distribution(arr, preprocess=True):
    distribution = batch_features(arr, preprocess)
    return feature_distribution(distribution)

def feature_distribution(feature_arr):
    mu = feature_arr.mean(axis=0)
    sigma = np.cov(feature_arr, rowvar=False)
    return mu, sigma