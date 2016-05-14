import numpy as np;

# %% 2x upscaling
def get_lowres_coords(x,y,lowres_coords,highres_coords,first=True):
	if(x<0 or y<0):
		return [0,-1];
	li=-1;
	for i in range(len(lowres_coords)):
		if(lowres_coords[i][0]==x//2 and lowres_coords[i][1]==y//2):
			li=i;
	if(li>=0):
		return [1,li];
	if(not first):
		return [0,-1];
	h=1;
	if(x%2==0):
		h=-1;
	v=1;
	if(y%2==0):
		v=-1;
	o,li=get_lowres_coords(x+h,y,lowres_coords,highres_coords,False);
	if(li>=0):
		return [0,li];
	o,li=get_lowres_coords(x,y+v,lowres_coords,highres_coords,False);
	if(li>=0):
		return [0,li];
	o,li=get_lowres_coords(x+v,y+v,lowres_coords,highres_coords,False);
	if(li>=0):
		return [0,li];
	return [0,-1];

def show_lowres_in_highres(img,x,y,xfactor,yfactor):
	hx=x*xfactor;
	hy=y*yfactor;
	himg=np.zeros([hy,hx]);
	for i in range(x):
		for j in range(y):
			for k in range(xfactor):
				for f in range(yfactor):
					tmp=img[j][i];
					himg[yfactor*j+f][xfactor*i+k]=tmp;
	return himg;

def upscale_nodes_of_instance(instance,lowres_coords,
		highres_coords):
	highres_nodes=[];
	for i in range(len(highres_coords)):
		x=highres_coords[i][0];
		y=highres_coords[i][1];
		o,li=get_lowres_coords(x,y,lowres_coords,highres_coords);
		highres_nodes.append(instance[li]);
	return highres_nodes;