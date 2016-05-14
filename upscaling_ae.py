import upscaling;
import data_loading as dl;
import tensorflow as tf;
import matplotlib.pyplot as plt;
import numpy as np;
import data_loading as dl;
import visualization as vis;
import upscaling_ae_def as uae;
import autoencoder_definition as lowae;

filename=raw_input("filename\n");
print filename;
print "loading highres maps..."
[instances,
	coords,
	img_dims,
	onlyfiles]=dl.load_nodes_with_schema("highres",
	"mapschemahighres.csv");
print "loading highres maps completed..."
print "loading lowres maps..."
[linstances,
	lcoords,
	limg_dims,
	lonlyfiles]=dl.load_nodes_with_schema("termalmaps",
	"mapschema.csv");
print "loading lowres maps completed..."
print "training set init...";
trainingset=instances[1:int(0.8*len(instances))];
print "training set permutation...";
trainingset=np.random.permutation(np.array(trainingset));
print "training set init complete..."
print "validation set init..."
validationset=instances[int(0.8*len(instances)):int(0.9*len(instances))];
print "test set init..."
testset=instances[int(0.9*len(instances)):];
testset_beggining=int(0.9*len(instances))
print "sets initialized..."
hidden_node_number=8;
print "lowres coords loading";
[lowres_coords,lowres_img_dims]=dl.load_schema("mapschema.csv");
print "lowres coords loaded...";
print "defining lowres autoencoder..."
low_ae=lowae.autoencoder(dimensions=[len(lowres_coords),
	hidden_node_number]);
print "lowres ae defined...";

saver=tf.train.Saver();
rfilename="only_enc3.ckpnt";
sess=tf.Session();
# sess.run(tf.initialize_all_variables())
saver.restore(sess, rfilename)
print "highres ae definition...";
low_encW,low_encb,low_decW,low_decb=sess.run(
	[low_ae['encW'],low_ae['encb'],
	low_ae['decW'],low_ae['decb']]);
low_representation,low_recon=sess.run(
	[low_ae['represent'],low_ae['y']],
	feed_dict={low_ae['x']:[linstances[1]]}
	);

high_ae=uae.autoencoder(len(coords),
	hidden_node_number,
	low_encW,low_encb,low_decW,low_decb,
	lowres_coords,coords);
print "highres ae defined...";
print "initializing highres ae..."
sess.run(tf.initialize_variables(
	var_list=[high_ae['encW'],
				high_ae['decW'],
				high_ae['encb'],
				high_ae['decb']]));
print sess.run(high_ae['encb']);

for i in range(len(coords)):
	if(upscaling.get_lowres_coords(coords[i][0],coords[i][1],
		lcoords,coords)<0):
		print "negativan",i;

fig, axs=plt.subplots(2,8);
nlow_encW=np.transpose(low_encW);

nhw=sess.run(high_ae['encW']);
nnhw=np.transpose(nhw);
#encoder influence
for i in range(hidden_node_number):
	axs[0][i].imshow(upscaling.show_lowres_in_highres(
		vis.mapnodestoimg(nlow_encW[i],
		limg_dims['x'],limg_dims['y'],
		lcoords),limg_dims['x'],limg_dims['y'],2,2));
	axs[1][i].imshow(vis.mapnodestoimg(nnhw[i],
		img_dims['x'],img_dims['y'],
		coords));
fig.show();
plt.draw();
plt.waitforbuttonpress();

#decoder impact
nhw=sess.run(high_ae['decW']);

for i in range(hidden_node_number):
	axs[0][i].imshow(upscaling.show_lowres_in_highres(
		vis.mapnodestoimg(low_decW[i],
		limg_dims['x'],limg_dims['y'],
		lcoords),limg_dims['x'],limg_dims['y'],2,2));
	axs[1][i].imshow(vis.mapnodestoimg(nhw[i],
		img_dims['x'],img_dims['y'],
		coords));
fig.show();
plt.draw();
plt.waitforbuttonpress();

trainingset=instances[1:int(0.8*len(instances))];
np.random.seed(3)
trainingset=np.random.permutation(np.array(trainingset));
validationset=instances[int(0.8*len(instances)):int(0.9*len(instances))];
testset=instances[int(0.9*len(instances)):];
testset_beggining=int(0.9*len(instances));

learning_rate = tf.Variable(initial_value=0.001,
	dtype=tf.float32,
	name="learning_rate");
batch_size = 25;
saved_epoch=tf.Variable(initial_value=0,name='saved_epoch');
n_epochs=500;
sess.run(tf.initialize_all_variables());

optimizer_def = tf.train.AdamOptimizer(
		learning_rate);
optimizer=optimizer_def.minimize(high_ae['cost']);
sess.run(tf.initialize_all_variables());
rep,rec=sess.run([high_ae['z'],high_ae['y']],feed_dict={
	high_ae['x']:[upscaling.upscale_nodes_of_instance(
		linstances[1],lcoords,coords)]});


print "representations:";
print type(low_representation);
print low_representation;
print rep;
print (rep-low_representation);
print "....";

print sess.run(high_ae['cost'],feed_dict={high_ae['x']:[
	upscaling.upscale_nodes_of_instance(
		linstances[0],lcoords,coords)
	]});

print "...";

vis.plot_maps([instances[1],
	upscaling.upscale_nodes_of_instance(
		linstances[1],lcoords,coords),
	upscaling.upscale_nodes_of_instance(
		low_recon[0],lcoords,coords),
	rec[0]],coords,img_dims['x'],img_dims['y']);


filesaver=tf.train.Saver(var_list=[
	high_ae['encW'],
	high_ae['decW'],
	high_ae['encb'],
	high_ae['decb']]);
filesaver.save(sess,filename);
error=100000000000;
for epoch_i in range(n_epochs):
	i_batch=0;
	c=0;
	print("epoch "+str(epoch_i)+":    "),
	for batch_i in range(len(trainingset) // batch_size):
		batch_xs= trainingset[i_batch:i_batch+batch_size]
		i_batch=i_batch+batch_size
		train = np.array(batch_xs)
		o,c=sess.run([optimizer,high_ae['cost']],
		 feed_dict={high_ae['x']: train});
	print c;
	if(error<c):
		lop=tf.assign(learning_rate,sess.run(learning_rate)/2);
		print 'lr;',sess.run(lop);
	error=c;
	if(epoch_i%100==0):
		saved_epoch=epoch_i;
		savedfile=filesaver.save(sess,filename);
		print "progress saved in %s"%(savedfile);