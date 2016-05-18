import data_loading as dl;
import upscaling_ae_def as aedef;
import visualization as vis;
import tensorflow as tf;
import numpy as np;

load_filename=raw_input("filename:");
print "load file:",load_filename;

print "loading schemas";
[lcoords,limg_dims]=dl.load_schema("mapschema.csv");
[hcoords,himg_dims]=dl.load_schema("mapschemahighres.csv");

print "initializing ae";
hidden_node_number=8;
linput_dim=len(lcoords);
hinput_dim=len(hcoords);
lae=aedef.autoencoder_contd(linput_dim,hidden_node_number);

print "loading values";
sess=tf.Session();
sess.run(tf.initialize_all_variables());
loader=tf.train.Saver(var_list=[lae['encW'],
			lae['decW'],
			lae['encb'],
			lae['decb']]);
loader.restore(sess,load_filename);

print "plotting";
vis.plot_maps(sess.run(lae['decW']),
	lcoords,limg_dims['x'],limg_dims['y']);
vis.plot_maps(sess.run(tf.transpose(lae['encW'])),
	lcoords,limg_dims['x'],limg_dims['y']);

print "upscaling"
hae=aedef.autoencoder(hinput_dim,
	hidden_node_number,
	sess.run(lae['encW']),
	sess.run(lae['encb']),
	sess.run(lae['decW']),
	sess.run(lae['decb']),
	lcoords,hcoords,
	'newEncW','newEncb','newDecW','newDecb');


learning_rate=tf.Variable(0.00001,
	name="learning_rate",
	dtype=tf.float32);
batch_size=tf.Variable(25,name="batch_size");
start_epoch=tf.Variable(0,name="start_epoch");
hidden_node_numbers=tf.Variable(16,name="hidden_node_number");
sess.run(tf.initialize_all_variables());
print "plotting";

vis.plot_maps(sess.run(hae['decW']),
	hcoords,himg_dims['x'],himg_dims['y']);
vis.plot_maps(sess.run(tf.transpose(hae['encW'])),
	hcoords,himg_dims['x'],himg_dims['y']);

saver=tf.train.Saver(var_list=[hae['encW'],
			hae['decW'],
			hae['encb'],
			hae['decb'],
			learning_rate,
			batch_size,
			start_epoch,
			hidden_node_numbers]);
save_filename=raw_input("saving filename:");
print "saving in %s"%(save_filename);
saver.save(sess,save_filename);