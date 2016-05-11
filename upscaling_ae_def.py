import tensorflow as tf;
import numpy as np;
import math;
import upscaling;

def upscaling_ae_init(input, representation,encW,decW,coords):
	x = tf.placeholder(tf.float32, [None, input], name='high_x')
	current_input = x;
	
	encoder=[];
	W=tf.Variable([input,representation],
		name="high_encW");
	