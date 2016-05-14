import numpy as np

def mapnodestoimg(ionodes,x,y,coords):
	imgdata=np.zeros([x,y],dtype=np.float32);
	for i in range(len(coords)):
		imgdata[coords[i][0],coords[i][1]]=ionodes[i];
	imgdata=np.transpose(imgdata);
	return imgdata

def mapnodestoimgtf(ionodes,x,y,coords):
	imgdata=tf.Variable(tf.zeros([x,y],dtype=np.float32));
	for i in range(len(coords)):
		imgdata[coords[i][0],coords[i][1]]=ionodes[i];
	imgdata=np.transpose(imgdata);
	return imgdata

# %% maps=list(termalmaps)
def plot_maps(maps,coords,width,height):
	import matplotlib.pyplot as plt;
	maps=np.array(maps);
	shape=maps.shape;
	num_vert=1;
	num_hor=len(maps);
	if(len(shape)>2):
		num_vert=shape[0];
		num_hor=shape[1];

	fig, axs=plt.subplots(num_vert,num_hor);
	for j in range(num_vert):
		for i in range(num_hor):
			if(num_vert==1):
				fig.colorbar(
					axs[i].imshow(
						mapnodestoimg(maps[i],width,
					height,coords)),ax=axs[i]);
			else:
				fig.colorbar(
					axs[j][i].imshow(
						mapnodestoimg(maps[j][i],width,
					height,coords)),ax=axs[j][i]);
	fig.show();
	plt.draw();
	plt.waitforbuttonpress();