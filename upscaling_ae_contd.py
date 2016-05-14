import upscaling_ae_def as aedef;
import data_loading as dl;
import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
import visualization as vis;

load_filename=raw_input("filename for import: ");
print "loading filename=",load_filename;
save_filename=raw_input("filename for export: ");
print "saving filename=",save_filename;

n_epochs=input("number of epochs to train: ");#todo drugacije
print "number_of_epochs",n_epochs;

print "loading maps..."
[instances,
	coords,
	img_dims,
	onlyfiles]=dl.load_nodes_with_schema("highres",
	"mapschemahighres.csv");
print "loading maps completed";
print "initializing sets...";
trainingset=instances[1:int(0.8*len(instances))];
np.random.seed(3);
trainingset=np.random.permutation(np.array(trainingset));
validationset=instances[int(0.8*len(instances)):int(0.9*len(instances))];
testset_beggining=int(0.9*len(instances))
testset=instances[testset_beggining:];
print "sets initialized...";
print "initializing model params...";
hidden_node_number=8;
input_dim=len(coords);
ae=aedef.autoencoder_contd(input_dim,hidden_node_number);
print "model params initialized";
print "initializing training params";
learning_rate=tf.Variable(0.00001,
	name="learning_rate",
	dtype=tf.float32);
adapt_learning_rate=tf.assign(learning_rate,learning_rate/1.1);
batch_size=tf.Variable(25,name="batch_size");
start_epoch=tf.Variable(0,name="start_epoch");
optimizer=tf.train.AdamOptimizer(
	learning_rate).minimize(
		loss=ae['cost'],
		var_list=[
			ae['encW'],
			ae['decW'],
			ae['decb'],
			ae['encb']
			]);
sess=tf.Session();
sess.run(tf.initialize_all_variables());
loader=tf.train.Saver(var_list=[ae['encW'],
			ae['decW'],
			ae['encb'],
			ae['decb'],
			learning_rate,
			batch_size,
			start_epoch]);
if(load_filename!=""):
	loader.restore(sess,load_filename);
saver=tf.train.Saver(var_list=[ae['encW'],
			ae['decW'],
			ae['encb'],
			ae['decb'],
			learning_rate,
			batch_size,
			start_epoch]);

print sess.run(ae['cost'],feed_dict={
	ae['x']:trainingset
	});
print sess.run(ae['cost'],feed_dict={
	ae['x']:validationset
	});
print sess.run(ae['cost'],feed_dict={
	ae['x']:testset
	});

# exit();
# summary_writer=tf.train.SummaryWriter("./"+"summaries");
c=0;
error=10000000000;
start_epoch_now=sess.run(start_epoch);
for epoch_i in range(start_epoch_now,start_epoch_now+n_epochs):
	i_batch=0;
	print ("epoch "+str(epoch_i)+":\t"),
	for batch_i in range(len(trainingset) // sess.run(batch_size)):
		batch_xs= trainingset[i_batch:i_batch+sess.run(batch_size)];
		i_batch=i_batch+sess.run(batch_size);
		o,c=sess.run(fetches=[optimizer,ae['cost']],
			feed_dict={ae['x']:batch_xs});
	print "%2.6f\t%2.15f\t%2.2f"%(
		c,c/len(coords),c/len(coords)*100);
	if(c>error*1.01):
		print "new lr",sess.run(adapt_learning_rate);
	error=c;
	if(epoch_i%100==0 and epoch_i!=start_epoch_now):
		sess.run(tf.assign(start_epoch,epoch_i));
		saved_filename=saver.save(sess,save_filename);
		print "tmp model saved in %s"%(saved_filename)
	# summary_writer.add_summary(c,epoch_i);
sess.run(tf.assign(start_epoch,start_epoch_now+n_epochs));
saved_filename=saver.save(sess,save_filename);
print "model saved in %s"%(saved_filename)
recon=sess.run(fetches=ae['y'],
	feed_dict={ae['x']:trainingset[0:5]});
influence,impact=sess.run(
	fetches=[ae['encW'],ae['decW']]);
influence=np.transpose(influence);
vis.plot_maps([trainingset[0:5],recon],coords,img_dims['x'],img_dims['y']);
vis.plot_maps([influence,impact],coords,img_dims['x'],img_dims['y']);
# vis.plot_maps(impact,coords,img_dims['x'],img_dims['y']);
