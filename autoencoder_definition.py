

import tensorflow as tf
import numpy as np
import math


# %% Autoencoder definition
def autoencoder(dimensions=[784, 512, 256, 64]):
	"""Build a deep autoencoder w/ tied weights.

	Parameters
	----------
	dimensions : list, optional
		The number of neurons for each layer of the autoencoder.

	Returns
	-------
	x : Tensor
		Input placeholder to the network
	z : Tensor
		Inner-most latent representation
	y : Tensor
		Output reconstruction of the input
	cost : Tensor
		Overall cost to use for training
	"""
	# %% input to the network
	x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
	current_input = x

	# %% Build the encoder
	encoder = []
	for layer_i, n_output in enumerate(dimensions[1:]):
		n_input = int(current_input.get_shape()[1])
		W = tf.Variable(
			tf.random_uniform([n_input, n_output],
							  #-1.0 / math.sqrt(n_input),
							  -math.sqrt(6.0/(n_input+n_output)),
							  #1.0 / math.sqrt(n_input)),
							  math.sqrt(6.0/(n_input+n_output))),
							  name='encoderW')
		b = tf.Variable(tf.zeros([n_output]),name='encoderb')
		encoder.append(W)
		output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
		current_input = output

	# %% latent representation
	z = W
	encb=b;
	representation=current_input;
	encoder.reverse()

	# %% Build the decoder using the same weights
	for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
		n_input = int(current_input.get_shape()[1])
		W = tf.Variable(
			tf.random_uniform([n_input, n_output],
							  #-1.0 / math.sqrt(n_input),
							  -math.sqrt(6.0/(n_input+n_output)),
							  #1.0 / math.sqrt(n_input)),
								math.sqrt(6.0/(n_input+n_output))),
								name='decoderW')
		tf.assign(W,tf.transpose(encoder[layer_i]))
		# W = tf.transpose(encoder[layer_i])
		b = tf.Variable(tf.zeros([n_output]),name='decoderb')
		output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
		current_input = output

	# %% now have the reconstruction through the network
	y = current_input
	# %% cost function measures pixel-wise difference
	cost = tf.reduce_mean(tf.nn.l2_loss(x-y))
	return {'x': x, 'z': z, 'y': y, 
		'cost': cost,'enc':encoder,'represent':representation,
		'encW':z,'decW':W,'encb':encb,'decb':b}

