"""Tutorial on how to create an autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
# %% Imports
import tensorflow as tf
import numpy as np
import math
from datetime import date
import visualization
import copy

from os import listdir
from os.path import isfile, join
mypath="termalmaps"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

instances=[]
borders=[]
img_dims={'x':0,'y':0};
coords=[]
for filename in onlyfiles:
	f=open(mypath+"/"+filename,'r')
	values=[]
	NAcell=True;
	borders=[];
	img_dims['y']=0;
	coords=np.zeros(len(onlyfiles));
	add_counter=0;
	y=0;
	coords=[]
	for line in f:
		img_dims['y']+=1;
		img_dims['x']=0;
		border=[];
		cells=line.split(' ');
		counter=0;
		x=0;
		for cell in cells:
			img_dims['x']+=1;
			if(cell!='NA'):
				try:
					values.append(float(cell));
					coords.append((x,y));
					add_counter+=1;
					if(NAcell==True):
						border.append(counter);
						NAcell=False;
				except ValueError:
					continue
			else:
				if(NAcell==False):
					border.append(counter);
					NAcell=True;
			counter+=1;
			x=x+1;
		border.append(img_dims['x']);
		borders.append(border);
		y=y+1;
	instances.append(values);

instance_min=min(min(instances[:]));
instance_max=max(max(instances[:]));
print("max:",instance_max)
print("min",instance_min)
instance_max-=instance_min;
original_instances=instances;
instances=(np.array(instances)-instance_min)/instance_max;
#for i in range(len(instances)):
	#if((original_instances[i]-(instances[i]*instance_max+instance_min)).any()>1):
		#print(i);
#quit()
#for instance in range(len(instances)):
	#for entry in range(len(instances[instance])):
		#instances[instance][entry]-=instance_min;
		#instances[instance][entry]/=instance_max;

def cosine_sim(a,b):
	sim=0;
	for i in range(len(a)):
		sim+=a[i]*b[i];
	a_norm=0;
	for i in range(len(a)):
		a_norm+=a[i]*a[i];
	b_norm=0;
	for i in range(len(a)):
		b_norm+=b[i]*b[i];
	a_norm=math.sqrt(a_norm)
	b_norm=math.sqrt(b_norm)
	sim/=a_norm*b_norm;
	return sim;

def euclid(a,b):
	return math.sqrt(sum((np.array(a)-np.array(b))**2))

def dayspan(a,b):
	date1=a.split('.')[0]
	date2=b.split('.')[0]
	month1=date1[4:6]
	month2=date2[4:6]
	day1=date1[6:]
	day2=date2[6:]
	date1=date(1990,int(month1),int(day1))
	date2=date(1990,int(month2),int(day2))
	date11=date(1991,int(month1),int(day1))
	date12=date(1989,int(month1),int(day1))
	return min(abs((date1-date2).days),(date11-date2).days,(date2-date12).days)


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
							  -1.0 / math.sqrt(n_input),
							  1.0 / math.sqrt(n_input)),name='encoderW')
		b = tf.Variable(tf.zeros([n_output]),name='encoderb')
		encoder.append(W)
		output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
		current_input = output

	# %% latent representation
	z = W
	representation=current_input;
	encoder.reverse()

	# %% Build the decoder using the same weights
	for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
#        W = tf.transpose(encoder[layer_i])
		n_input = int(current_input.get_shape()[1])
		W = tf.Variable(
			tf.random_uniform([n_input, n_output],
							  -1.0 / math.sqrt(n_input),
							  1.0 / math.sqrt(n_input)),name='decoderW')
		tf.assign(W,tf.transpose(encoder[layer_i]))
		b = tf.Variable(tf.zeros([n_output]),name='decoderb')
		output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
		current_input = output

	# %% now have the reconstruction through the network
	y = current_input
	# %% cost function measures pixel-wise difference
	cost = tf.reduce_sum(tf.square(y - x))
	return {'x': x, 'z': z, 'y': y, 'cost': cost,'enc':encoder,'represent':representation}


# %% Basic test
def test_mnist():
	"""Test the autoencoder using MNIST."""
	import tensorflow as tf
	import tensorflow.examples.tutorials.mnist.input_data as input_data
	import matplotlib.pyplot as plt

	# %%
	# load MNIST as before
	#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	#mean_img = np.mean(instances, axis=0)
	
	trainingset=instances[1:int(0.8*len(instances))];
	validationset=instances[int(0.8*len(instances)):int(0.9*len(instances))];
	testset=instances[int(0.9*len(instances)):];
	hidden_node_number=3
	ae = autoencoder(dimensions=[len(trainingset[0]), hidden_node_number])

	# %%
	learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

	# %%
	# We create a session to use the graph
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver()
	# %%
	# Fit all training data
	
	batch_size = 20
	n_epochs = 3000
	trainingNow=True;
	filename='./models/';
	filename+=str(hidden_node_number)+'n'+str(batch_size)+'b'+str(n_epochs)+'e';
	filenamenoExtension=filename;
	filename+='.ckpnt';
	if(isfile(filename)):
		decision=input(
			"File already exists,choose:\n0 to read,\n1 to add a v to the name\n2 to overwrite\n")
		if(decision==0):
			trainingNow=False;
		elif(decision==1):
			filenamenoExtension+='v'
			while(isfile(filenamenoExtension+'.ckpnt')):
				filenamenoExtension+='v'
			filename=filenamenoExtension+'.ckpnt'
	if(trainingNow==True):
		from_epoch=2000;
		if(from_epoch>0):
			partialmodelfile='./models/'+str(
				hidden_node_number)+'n'+str(
					batch_size)+'b'+str(
						from_epoch)+'e'+'.ckpnt';
			saver.restore(sess, partialmodelfile)
			print("Partial model restored.")
		for epoch_i in range(from_epoch,n_epochs):
			i_batch=0;
			c=0;
			for batch_i in range(len(trainingset) // batch_size):
				batch_xs= trainingset[i_batch:i_batch+batch_size]
				i_batch=i_batch+batch_size
				train = np.array(batch_xs)
				sess.run(optimizer, feed_dict={ae['x']: train})
			costtr,latent=sess.run([ae['cost'],ae['z']], feed_dict={ae['x']: trainingset})
			costtr/=len(trainingset)*len(trainingset[0])
			costval=sess.run(ae['cost'], feed_dict={ae['x']: validationset})
			costval/=len(validationset)*len(validationset[0])		
			print(epoch_i, math.sqrt(costtr),math.sqrt(costval),[len(latent),len(latent[0])])
		save_path = saver.save(sess, filename)
		print("Model saved in file: %s" % save_path)
	else:
		saver.restore(sess, filename)
		print("Model restored.")
	#hidden layer visualization
	latent=sess.run(ae['z']);
	hidden_layer=np.transpose(latent);
	fig, axs = plt.subplots(1, len(hidden_layer))
	node_counter=0
	for node in hidden_layer:
		axs[node_counter].imshow(visualization.mapnodestoimg(
			node,
			img_dims['x'],img_dims['y'],
			coords))
		node_counter+=1;
	fig.canvas.set_window_title('Skriveni neuroni')
	fig.show()
	plt.draw()
	#end
	n_examples = 5
	test_xs=testset[0:n_examples]
	test_xs_norm = test_xs
	recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
	fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
	for example_i in range(n_examples):
		print(math.sqrt(sum((np.array(test_xs[example_i])-np.array(recon[example_i]))**2)/len(test_xs[example_i])))
		axs[0][example_i].imshow(visualization.mapnodestoimg(
			test_xs[example_i],
			img_dims['x'],img_dims['y'],
			coords))
		axs[1][example_i].imshow(visualization.mapnodestoimg(
			recon[example_i],
			img_dims['x'],img_dims['y'],
			coords))
	fig.canvas.set_window_title('Rekonstrukcija')
	fig.show()
	plt.draw()
	
	recon = sess.run(ae['y'], feed_dict={ae['x']: testset})
	compression_loss=0;
	for i in range(len(testset)):
		img_compression_loss=0;
		for j in range(len(testset[i])):
			img_compression_loss+=abs(testset[i][j]-recon[i][j])
		img_compression_loss/=len(testset[i])
		compression_loss+=img_compression_loss
	compression_loss/=len(testset)
	print('compression loss:',compression_loss)

	representations=sess.run(ae['represent'],feed_dict={ae['x']:testset})
	min_dist=10000000;
	ijmindist=(-1,-1)
	closest=[]
	datedistavg=0;
	datedistn=0;
	min_span=1000000
	original_distances=[]
	representation_distances=[]
	for i in range(len(testset)):
		closestI=-1;
		closestDistance=1000000;
		for j in range(len(testset)):
			if(i!=j):
				original_distances.append(euclid(testset[i],testset[j]))
				representation_distances.append(
					euclid(representations[i],representations[j]))
				distance=euclid(representations[i],representations[j])
				if(distance<min_dist):
					min_dist=distance;
					ijmindist=(i,j)
				if(distance<closestDistance):
					closestI=j;
					closestDistance=distance;
		closest.append((onlyfiles[int(0.9*len(original_instances))+i],
				  onlyfiles[int(0.9*len(original_instances))+closestI],closestDistance))
		day_span=dayspan(onlyfiles[int(0.9*len(original_instances))+i],
				   onlyfiles[int(0.9*len(original_instances))+closestI])
		if(min_span>day_span):
			min_span=day_span;
			if(day_span==1):
				print("najblizi:",
		  onlyfiles[int(0.9*len(original_instances))+i],
		  onlyfiles[int(0.9*len(original_instances))+closestI],closestDistance)
		datedistavg+=dayspan(onlyfiles[int(0.9*len(original_instances))+i],
					   onlyfiles[int(0.9*len(original_instances))+closestI])
		datedistn+=1;
	print("correlation:",np.corrcoef(
		original_distances, representation_distances)[0, 1])
	most_apart=0;
	for c in sorted(closest, key=lambda x: x[2], reverse=True):
		print c;
		most_apart=c;
	datedistavg/=datedistn;
	print("date distance average:",datedistavg)
	print("min dayspan for min distance:",min_span)
	print(onlyfiles[int(0.9*len(original_instances))+ijmindist[0]],
		onlyfiles[int(0.9*len(original_instances))+ijmindist[1]],min_dist)
	print("minimax1:",min(testset[ijmindist[0]]),max(testset[ijmindist[0]]))
	print("minimax1original:",
	   min(original_instances[int(0.9*len(original_instances))+ijmindist[0]]),
	   max(original_instances[int(0.9*len(original_instances))+ijmindist[0]]))
	print("minimax2:",min(testset[ijmindist[1]]),max(testset[ijmindist[1]]))
	print("minimax2original:",
	   min(original_instances[int(0.9*len(original_instances))+ijmindist[1]]),
	   max(original_instances[int(0.9*len(original_instances))+ijmindist[1]]))
	print(representations[ijmindist[0]],representations[ijmindist[1]])
	fig, axs = plt.subplots(1, 2)
	ax1=axs[0].imshow(visualization.mapnodestoimg(
		testset[ijmindist[0]],
		img_dims['x'],img_dims['y'],
		coords));
	ax2=axs[1].imshow(visualization.mapnodestoimg(
		testset[ijmindist[1]],
		img_dims['x'],img_dims['y'],
		coords))
	fig.canvas.set_window_title('Najblizi datumi')
	fig.colorbar(ax1)
	fig.colorbar(ax2)
	fig.show()
	plt.draw()
	plt.waitforbuttonpress()
# %%
if __name__ == '__main__':
	test_mnist()
