import tensorflow as tf;
import numpy as np;
import math;
import upscaling;

def autoencoder(input_dim, representation,
	encW,encb,decW,decb,lowres_coords,highres_coords,
	encWname,encbname,decWname,decbname):
	x = tf.placeholder(tf.float32, [None, input_dim]);
	current_input = x;
	high_W=np.zeros([input_dim,representation]);
	i_num=np.zeros(len(lowres_coords));
	print "upscaling encoder...";
	for i in range(input_dim):
		notNa,li=upscaling.get_lowres_coords(highres_coords[i][0],
			highres_coords[i][1],lowres_coords,highres_coords);
		i_num[li]+=1;
		for j in range(representation):
			high_W[i][j]=encW[li][j];
	print "upscaling encoder finished...";
	print "adjusting encoder weights...";
	for i in range (input_dim):
		o,li=upscaling.get_lowres_coords(highres_coords[i][0],
			highres_coords[i][1],lowres_coords,highres_coords);
		for j in range(representation):
				high_W[i][j]/=i_num[li];
	print "adjusting encoder weights...";
	encoder=[];
	high_encW=tf.Variable(initial_value=high_W,
		name=encWname,
		dtype=tf.float32);
	high_encb=tf.Variable(initial_value=encb,
		dtype=tf.float32,
		name=encbname);
	encoder.append(encW);
	# %%latent representation
	z=tf.nn.sigmoid(tf.matmul(x,high_encW) + high_encb);
	hidden_weights=high_encW;
	high_W=np.zeros([representation,input_dim]);
	for i in range(input_dim):
		o,li=upscaling.get_lowres_coords(highres_coords[i][0],
			highres_coords[i][1],lowres_coords,highres_coords);
		for j in range(representation):
			high_W[j][i]=decW[j][li];
	high_decW=tf.Variable(initial_value=high_W,
		dtype=tf.float32,
		name=decWname);
	dec_b_init=np.zeros([input_dim]);
	for i in range(input_dim):
		o,li=upscaling.get_lowres_coords(highres_coords[i][0],
			highres_coords[i][1],lowres_coords,highres_coords);
		dec_b_init[i]=decb[li];
	high_decb=tf.Variable(initial_value=dec_b_init,
		dtype=tf.float32,
		name=decbname);
	output=tf.nn.sigmoid(tf.matmul(z,high_decW)+high_decb);
	y=output;
	cost=tf.reduce_mean(tf.nn.l2_loss(tf.abs(x-y)));
	return {'x':x,'z':z,'y':y,'cost':cost,
		'weights':hidden_weights,
		'encW':high_encW,'decW':high_decW,
		'encb':high_encb,'decb':high_decb,
		};

def autoencoder2x(input_dim, representation,
	encW,encb,decW,decb,lowres_coords,highres_coords,
	encWname,encbname,decWname,decbname):
	x = tf.placeholder(tf.float32, [None, input_dim]);
	current_input = x;
	high_W=np.zeros([input_dim,representation*2]);
	i_num=np.zeros(len(lowres_coords));
	print "upscaling encoder...";
	for i in range(input_dim):
		notNa,li=upscaling.get_lowres_coords(highres_coords[i][0],
			highres_coords[i][1],lowres_coords,highres_coords);
		i_num[li]+=1;
		for j in range(representation):
			high_W[i][2*j]=encW[li][j];
			high_W[i][2*j+1]=encW[li][j];
	print "upscaling encoder finished...";
	print "adjusting encoder weights...";
	for i in range (input_dim):
		o,li=upscaling.get_lowres_coords(highres_coords[i][0],
			highres_coords[i][1],lowres_coords,highres_coords);
		for j in range(representation):
				high_W[i][j]/=i_num[li];
	print "adjusting encoder weights...";
	encoder=[];
	high_encW=tf.Variable(initial_value=high_W,
		name=encWname,
		dtype=tf.float32);
	init_encb=np.zeros([representation*2]);
	for i in range(representation):
		init_encb[2*i]=encb[i];
		init_encb[2*i+1]=encb[i];

	high_encb=tf.Variable(initial_value=init_encb,
		dtype=tf.float32,
		name=encbname);
	encoder.append(encW);
	# %%latent representation
	z=tf.nn.sigmoid(tf.matmul(x,high_encW) + high_encb);
	hidden_weights=high_encW;
	high_W=np.zeros([representation*2,input_dim]);
	for i in range(input_dim):
		o,li=upscaling.get_lowres_coords(highres_coords[i][0],
			highres_coords[i][1],lowres_coords,highres_coords);
		for j in range(representation):
			high_W[2*j][i]=decW[j][li]/2;
			high_W[2*j+1][i]=decW[j][li]/2;
	high_decW=tf.Variable(initial_value=high_W,
		dtype=tf.float32,
		name=decWname);
	dec_b_init=np.zeros([input_dim]);
	for i in range(input_dim):
		o,li=upscaling.get_lowres_coords(highres_coords[i][0],
			highres_coords[i][1],lowres_coords,highres_coords);
		dec_b_init[i]=decb[li];
	high_decb=tf.Variable(initial_value=dec_b_init,
		dtype=tf.float32,
		name=decbname);
	output=tf.nn.sigmoid(tf.matmul(z,high_decW)+high_decb);
	y=output;
	cost=tf.nn.l2_loss(x-y);
	return {'x':x,'z':z,'y':y,'cost':cost,
		'weights':hidden_weights,
		'encW':high_encW,'decW':high_decW,
		'encb':high_encb,'decb':high_decb
		};



def autoencoder_contd(input_dim, representation):
	x = tf.placeholder(tf.float32, [None, input_dim]);
	high_decW=tf.Variable(
		initial_value=tf.random_normal(
			[representation,input_dim],
			-math.sqrt(6.0/(input_dim+representation)),
			math.sqrt(6.0/(input_dim+representation))),
		dtype=tf.float32,
		name='high_decW');
	# high_encW=tf.transpose(high_decW);
	high_encW=tf.Variable(
		initial_value=tf.random_normal(
			[input_dim, representation],
			-math.sqrt(6.0/(input_dim+representation)),
			math.sqrt(6.0/(input_dim+representation))),
		name='high_encW');
	high_encb=tf.Variable(tf.zeros([representation]),
		name='high_encb');
	z=tf.nn.sigmoid(tf.matmul(x,high_encW) + high_encb);
	hidden_weights=high_encW;
	
	high_decb=tf.Variable(
		tf.zeros([input_dim]),
		name='high_decb');
	y=tf.nn.sigmoid(tf.matmul(z,high_decW)+high_decb);
	cost=tf.nn.l2_loss(x-y);
	loss_per_pixel=tf.reduce_mean(tf.abs(x-y));
	return {'x':x,'z':z,'y':y,'cost':cost,
		'weights':hidden_weights,
		'encW':high_encW,'decW':high_decW,
		'encb':high_encb,'decb':high_decb,
		'ppx':loss_per_pixel
		};