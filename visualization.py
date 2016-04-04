import numpy as np

def mapnodestoimg(ionodes,x,y,coords):
	imgdata=np.zeros([x,y],dtype=np.float32);
	for i in range(len(coords)):
		imgdata[coords[i][0],coords[i][1]]=ionodes[i];
	imgdata=np.transpose(imgdata);
	return imgdata