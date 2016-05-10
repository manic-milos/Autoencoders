# %% Imports
import tensorflow as tf
import numpy as np
import math
from datetime import date
import visualization
import model_evaluation as me
import copy
import sys
import data_loading as dl
import autoencoder_definition as autoenc_def

# [instances, coords, 
# 	original_instances, img_dims, 
# 	onlyfiles]=dl.load_maps("termalmaps");
[instances,
	coords,
	img_dims,
	onlyfiles]=dl.load_nodes_with_schema("termalmaps",
	"mapschema.csv");
original_instances=instances;
from os import listdir
from os.path import isfile, join

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
	np.random.seed(3)
	trainingset=np.random.permutation(np.array(trainingset));
	validationset=instances[int(0.8*len(instances)):int(0.9*len(instances))];
	testset=instances[int(0.9*len(instances)):];
	testset_beggining=int(0.9*len(instances))
	hidden_node_number=8
	ae = autoenc_def.autoencoder(dimensions=[len(trainingset[0]), hidden_node_number])

	# %%
	learning_rate = 0.000001
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
	traincostsum=tf.scalar_summary('traincost',ae['cost'])
	testcostsum=tf.scalar_summary('valcost',ae['cost'])
	# merge=tf.merge_all_summaries()
	
	# %%
	# We create a session to use the graph
	sess = tf.Session()
	train_writer = tf.train.SummaryWriter("." + '/train')
	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver()
	# %%
	# Fit all training data
	
	batch_size = 25
	n_epochs =4000;
	trainingNow=True;
	filename='./models/';
	filename+=str(hidden_node_number)+'n'+str(batch_size)+'b'+str(n_epochs)+'e';
	filenamenoExtension=filename;
	filename+='.ckpnt';
	continuedTraining=False
	if(isfile(filename)):
		decision=input("File already exists,choose:\n0 to read,\n1 to add a v to the name\n2 to overwrite\n")
		if(decision==0):
			trainingNow=False;
		elif(decision==1):
			filenamenoExtension+='v'
			while(isfile(filenamenoExtension+'.ckpnt')):
				filenamenoExtension+='v'
			filename=filenamenoExtension+'.ckpnt'
		elif(decision!=2):
			print("not an allowed choice...")
			quit()
	if(trainingNow==True):
		from_epoch=3800;
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
			print("epoch "+str(epoch_i)+":    "),
			for batch_i in range(len(trainingset) // batch_size):
				batch_xs= trainingset[i_batch:i_batch+batch_size]
				i_batch=i_batch+batch_size
				train = np.array(batch_xs)
				sess.run(optimizer, feed_dict={ae['x']: train})
				digits=2
				print "{0}{1:{2}}%".format(
					"\b" * (digits + 1+1), 
					int((batch_i+0.0)/(len(trainingset) // batch_size)*100),
					digits),
				sys.stdout.flush()
			digits=3
			print "{0}{1:{2}}%".format(
					"\b" * (digits + 1), 
					100,
					digits)
			traincostsumcurr,costtr,latent=sess.run([traincostsum,ae['cost'],ae['z']], feed_dict={ae['x']: trainingset})
			train_writer.add_summary(traincostsumcurr,epoch_i);
			costtr/=len(trainingset)*len(trainingset[0])
			valcost,costval=sess.run([testcostsum, ae['cost']], feed_dict={ae['x']: validationset})
			train_writer.add_summary(valcost,epoch_i);
			costval/=len(validationset)*len(validationset[0])		
			print(epoch_i, math.sqrt(costtr),math.sqrt(costval),[len(latent),len(latent[0])])
			
		print("filename",filename);
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
		fig.colorbar(axs[node_counter].imshow(visualization.mapnodestoimg(
			node,
			img_dims['x'],img_dims['y'],
			coords)),ax=axs[node_counter])
		node_counter+=1;
	
	fig.canvas.set_window_title('Skriveni neuroni')
	fig.show()
	plt.draw()
	#end
	#get testset results
	recon,representations = sess.run([ae['y'],ae['represent']], 
								  feed_dict={ae['x']: testset})
	n_examples = 5
	test_xs=testset[0:n_examples]
	test_xs_norm = test_xs
	fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
	for example_i in range(n_examples):
		print(math.sqrt(sum((np.array(test_xs[example_i])-np.array(recon[example_i]))**2)/len(test_xs[example_i])))
		fig.colorbar(axs[0][example_i].imshow(visualization.mapnodestoimg(
			test_xs[example_i],
			img_dims['x'],img_dims['y'],
			coords)),ax=axs[0][example_i])
		fig.colorbar(axs[1][example_i].imshow(visualization.mapnodestoimg(
			recon[example_i],
			img_dims['x'],img_dims['y'],
			coords)),ax=axs[1][example_i])
	fig.canvas.set_window_title('Rekonstrukcija')
	fig.show()
	plt.draw()
	##compression loss
	print("compression loss:",me.compression_loss(testset,recon))
	#end
	#statistics
	print("correlation on closest calculating...")
	print("correlation on closest="+str(
		me.closestRepCorrelation(testset,representations)))
	print("correlation calculating...")
	#print("correlation="+str(me.correlation(testset,representations)))
	print("index test calculation...")
	#print("indextest="+str(me.indexTest(testset,representations)))
	#end
	#closest representations list
	closestRepresentations=me.closestRepresentationList(testset,representations)
	for c in closestRepresentations:
		print (onlyfiles[testset_beggining+c[0]],
				onlyfiles[testset_beggining+c[1]],
				c[2])
	#end
	#average closest representation dayspan
	
	avgdayspan=0
	for c in closestRepresentations:
		avgdayspan+=me.dayspan(onlyfiles[testset_beggining+ c[0]],
								onlyfiles[testset_beggining+ c[1]])
	avgdayspan/=len(closestRepresentations)
	print("average day span="+str(avgdayspan))
	#end
	#closest distance representations
	c=min(closestRepresentations,key=lambda x:x[2])
	#end
	#minimal representation distance
	print("minimal distance dates:")
	print(onlyfiles[testset_beggining+c[0]],
		onlyfiles[testset_beggining+c[1]],c[2])
	#end
	print("minimax1:",min(testset[c[0]]),max(testset[c[0]]))
	# print("minimax1original:",
	#    min(original_instances[testset_beggining+c[0]]),
	#    max(original_instances[testset_beggining+c[0]]))
	print("minimax2:",min(testset[c[1]]),max(testset[c[1]]))
	# print("minimax2original:",
	#    min(original_instances[testset_beggining+c[1]]),
	#    max(original_instances[testset_beggining+c[1]]))
	print(representations[c[0]],representations[c[1]])
	fig, axs = plt.subplots(2, 2)
	recons=sess.run(ae['y'],feed_dict={
		ae['x']:[testset[c[0]],
		   testset[c[1]]]})
	ax1=axs[0][0].imshow(visualization.mapnodestoimg(
		testset[c[0]],
		img_dims['x'],img_dims['y'],
		coords));
	ax2=axs[0][1].imshow(visualization.mapnodestoimg(
		testset[c[1]],
		img_dims['x'],img_dims['y'],
		coords))
	ax3=axs[1][0].imshow(visualization.mapnodestoimg(
		recons[0],
		img_dims['x'],img_dims['y'],
		coords));
	ax4=axs[1][1].imshow(visualization.mapnodestoimg(
		recons[1],
		img_dims['x'],img_dims['y'],
		coords))
	fig.canvas.set_window_title('Najblizi datumi')
	fig.colorbar(mappable=ax1,ax=axs[0][0])
	fig.colorbar(mappable=ax2,ax=axs[0][1])
	fig.colorbar(mappable=ax3,ax=axs[1][0])
	fig.colorbar(mappable=ax4,ax=axs[1][1])
	fig.show()
	plt.draw()
	plt.waitforbuttonpress()
# %%
if __name__ == '__main__':
	test_mnist()
