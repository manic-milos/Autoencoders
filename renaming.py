import data_loading as dl;
import upscaling_ae_def as aedef;
import visualization as vis;
import tensorflow as tf;
import numpy as np;
import math;

load_filename=raw_input("filename:");
# load_filename="upscaled.ckpnt"
print "load file:",load_filename;

save_filename=raw_input("saving filename:");
print "save file:",save_filename;

[coords,img_dims]=dl.load_schema("mapschemahighres.csv");
representation=8;
hidden_node_number=representation;
input_dim=len(coords);

print "initializing loading vars";
high_decW=tf.Variable(
		initial_value=tf.random_normal(
			[representation,input_dim],
			-math.sqrt(6.0/(input_dim+representation)),
			math.sqrt(6.0/(input_dim+representation))),
		dtype=tf.float32,
		name='newDecW');
high_encW=tf.Variable(
	initial_value=tf.random_normal(
		[input_dim, representation],
		-math.sqrt(6.0/(input_dim+representation)),
		math.sqrt(6.0/(input_dim+representation))),
	name='newEncW');
high_encb=tf.Variable(tf.zeros([representation]),
	name='newEncb');
high_decb=tf.Variable(
		tf.zeros([input_dim]),
		name='newDecb');
learning_rate=tf.Variable(0.00001,
	name="learning_rate",
	dtype=tf.float32);
batch_size=tf.Variable(25,name="batch_size");
start_epoch=tf.Variable(0,name="start_epoch");
sess=tf.Session();
sess.run(tf.initialize_all_variables());
loader=tf.train.Saver(var_list=[
	high_decW,
	high_encW,
	high_encb,
	high_decb,
	learning_rate,
	start_epoch,
	batch_size
	])
print "restoring";
loader.restore(sess,load_filename);
vis.plot_maps(sess.run(high_decW),coords,
	img_dims['x'],img_dims['y']);
rundecW=sess.run(high_decW);
runencW=sess.run(high_encW);
runencb=sess.run(high_encb);
rundecb=sess.run(high_decb);
ae=aedef.autoencoder_contd(input_dim,hidden_node_number);
sess.run(tf.initialize_all_variables());

print "assigning";
sess.run(tf.assign(ae['encW'],runencW));
sess.run(tf.assign(ae['encb'],runencb));
sess.run(tf.assign(ae['decW'],rundecW));
sess.run(tf.assign(ae['decb'],rundecb));
vis.plot_maps(sess.run(ae['decW']),coords,
	img_dims['x'],img_dims['y']);
vis.plot_maps(sess.run(high_decW),coords,
	img_dims['x'],img_dims['y']);
print "saving"

saver=tf.train.Saver(var_list=[
	ae['encW'],
	ae['decW'],
	ae['encb'],
	ae['decb'],
	learning_rate,
	batch_size,
	start_epoch]);
saver.save(sess,save_filename);