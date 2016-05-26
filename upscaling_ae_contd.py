import upscaling_ae_def as aedef;
import data_loading as dl;
import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
import visualization as vis;
import math;
import model_evaluation as me;

load_filename=raw_input("filename for import(leave empty if new): ");
print "loading filename=",load_filename;
#load training parametara
learning_rate_init=0.00001;
batch_size_init=25;
map_schema_filename="mapschemahighres.csv";
maps_foldername="highres";
hidden_node_number=8;
if(load_filename==""):
	maps_foldername=raw_input("folder containing the maps:");
	map_schema_filename=raw_input("filename for map_schema:");
	hidden_node_number=input("number of nodes in hidden layer:");
	learning_rate_init=input("initial learning rate:");
	batch_size_init=input("batch size:");
learning_rate=tf.Variable(learning_rate_init,
	name="learning_rate",
	dtype=tf.float32);
batch_size=tf.Variable(batch_size_init,name="batch_size");
start_epoch=tf.Variable(0,name="start_epoch");
hidden_node_var=tf.Variable(hidden_node_number,
	name="hidden_node_number");
map_schema_var=tf.Variable(map_schema_filename,
	dtype=tf.string,
	name="map_schema");
maps_folder_var=tf.Variable(maps_foldername,
	dtype=tf.string,
	name="map_folder");
sess=tf.Session();
sess.run(tf.initialize_all_variables());
if(load_filename!=""):
	loader=tf.train.Saver(var_list=[
		learning_rate,
		batch_size,
		hidden_node_var,
		start_epoch,
		map_schema_var,
		maps_folder_var]);
	loader.restore(sess,load_filename)
learning_rate_init=sess.run(learning_rate);
batch_size_init=sess.run(batch_size);
hidden_node_number=sess.run(hidden_node_var);
map_schema_filename=sess.run(map_schema_var);
maps_foldername=sess.run(maps_folder_var);
print "loading schema..."
[coords,img_dims]=dl.load_schema(map_schema_filename);
print "schema loaded";
print "loaded parameters:";
print "layers:[",len(coords),sess.run(hidden_node_var),len(coords),"]";
print "learning rate:",sess.run(learning_rate);
print "batch_size:",sess.run(batch_size);
print "epochs trained",sess.run(start_epoch);
print "map schema filename:",sess.run(map_schema_var);
print "maps folder:", sess.run(maps_folder_var);

save_filename=raw_input("filename for export: ");
print "saving filename=",save_filename;
n_epochs=input("number of epochs to train: ");
print "number_of_epochs",n_epochs;

print "initializing model params...";
input_dim=len(coords);
ae=aedef.autoencoder_contd(input_dim,hidden_node_number);
print "model params initialized";
print "loading maps..."
[instances,
	coords,
	img_dims,
	onlyfiles]=dl.load_nodes_with_schema(maps_foldername,
	map_schema_filename);
print "loading maps completed";
print "initializing sets...";
trainingset=instances[1:int(0.8*len(instances))];
np.random.seed(3);
trainingset=np.random.permutation(np.array(trainingset));
validationset=instances[int(0.8*len(instances)):int(0.9*len(instances))];
testset_beggining=int(0.9*len(instances))
testset=instances[testset_beggining:];
print "sets initialized...";

adapt_learning_rate=tf.assign(learning_rate,learning_rate*0.99);
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
			start_epoch,
			map_schema_var,
			maps_folder_var,
			hidden_node_var]);
if(load_filename!=""):
	loader.restore(sess,load_filename);
print "settings:";
print "learning rate:",sess.run(learning_rate);
print "batch_size:",sess.run(batch_size);
print "start_epoch",sess.run(start_epoch);
saver=tf.train.Saver(var_list=[ae['encW'],
			ae['decW'],
			ae['encb'],
			ae['decb'],
			learning_rate,
			batch_size,
			start_epoch,
			hidden_node_var,
			map_schema_var,
			maps_folder_var]);

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
summary_writer=tf.train.SummaryWriter("./"+save_filename+"summary");
training_error_summary=tf.scalar_summary("training_error",
	ae['cost']);
validation_error_summary=tf.scalar_summary("validation_error",
	ae['cost']);
test_error_summary=tf.scalar_summary("test_error",
	ae['cost']);
sess.run(tf.assign(learning_rate,learning_rate_init));
c=0;
error=10000000000;
start_epoch_now=sess.run(start_epoch);
for epoch_i in range(start_epoch_now,start_epoch_now+n_epochs):
	i_batch=0;
	max=0;
	print ("epoch "+str(epoch_i)+":\t"),
	for batch_i in range(len(trainingset) // batch_size_init):
		batch_xs= trainingset[i_batch:i_batch+batch_size_init];
		i_batch=i_batch+batch_size_init;
		o,c=sess.run(fetches=[optimizer,ae['cost']],
			feed_dict={ae['x']:batch_xs});
		if(c>max):
			max=c;
	c,cpp,tesum=sess.run(fetches=[ae['cost'],ae['ppx'],
		training_error_summary],
		feed_dict={
		ae['x']:trainingset
		});
	summary_writer.add_summary(tesum,epoch_i);
	cppv,vesum=sess.run(fetches=[ae['ppx'],
		validation_error_summary],feed_dict={
		ae['x']:validationset
		});
	summary_writer.add_summary(vesum,epoch_i);
	print "%2.6f\t%2.10f\t%2.10f"%(
		c,cpp,cppv);
	if(c>error*1.01):
		print "new lr",sess.run(adapt_learning_rate);
	if(c<error):
		error=c;
	if(epoch_i%100==0 and epoch_i!=start_epoch_now):
		sess.run(tf.assign(start_epoch,epoch_i));
		saved_filename=saver.save(sess,save_filename);
		print "tmp model saved in %s"%(saved_filename)
		print "test set error per pixel:",
		teppx,tes=sess.run([ae['ppx'],test_error_summary],
			feed_dict={
			ae['x']:testset
			});
		print teppx;
		summary_writer.add_summary(tes,epoch_i);
sess.run(tf.assign(start_epoch,start_epoch_now+n_epochs));
saved_filename=saver.save(sess,save_filename);
print "model saved in %s"%(saved_filename)

print "test set error per pixel:",
print sess.run(ae['ppx'],feed_dict={
	ae['x']:testset
	});

recon=sess.run(fetches=ae['y'],
	feed_dict={ae['x']:testset[0:5]});
influence,impact=sess.run(
	fetches=[ae['encW'],ae['decW']]);
influence=np.transpose(influence);
vis.plot_maps([testset[0:5],recon],coords,img_dims['x'],img_dims['y']);
vis.plot_maps([influence,impact],coords,img_dims['x'],img_dims['y']);
# vis.plot_maps(impact,coords,img_dims['x'],img_dims['y']);
