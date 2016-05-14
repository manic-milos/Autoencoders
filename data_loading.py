from os import listdir
from os.path import isfile, join

import sys

import numpy as np
import math

def load_maps(mypath):
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	onlyfiles.sort()
	instances=[]
	img_dims={'x':0,'y':0};
	coords=[]
	percentindex=0
	print "opening files:	  ",
	for filename in onlyfiles:
		f=open(mypath+"/"+filename,'r')
		values=[]
		img_dims['y']=0;
		y=0;
		coords=[]
		for line in f:
			img_dims['y']+=1;
			img_dims['x']=0;
			border=[];
			cells=line.split(' ');
			x=0;
			for cell in cells:
				img_dims['x']+=1;
				if(cell!='NA'):
					try:
						values.append(float(cell));
						coords.append((x,y));
					except ValueError:
						continue
				x=x+1;
			y=y+1;
		instances.append(values);
		digits=2
		print "{0}{1:{2}}%".format(
						"\b" * (digits + 1+1), 
						int((percentindex+0.0)/len(onlyfiles)*100),
						digits),
		sys.stdout.flush()
		percentindex+=1
	print "{0}{1:{2}}%".format(
						"\b" * (digits + 1+1), 
						100,
						digits)
	instance_min=min(min(instances[:]));
	instance_max=max(max(instances[:]));
	print("max:",instance_max)
	print("min",instance_min)
	print("height:",img_dims['y'],"width:",img_dims['x'])
	instance_max-=instance_min;
	original_instances=instances;
	instances=(np.array(instances)-instance_min)/instance_max;
	return [instances,coords,original_instances,
		img_dims,onlyfiles]
def load_nodes_with_schema(mypath,schema,limit=-1):
	files=[f for f in listdir(mypath) if isfile(join(mypath,f))]
	instances=[];
	percentindex=0;
	files.sort();
	for filename in files:
		f=open(mypath+"/"+filename,'r');
		values=[];
		for line in f:
			cells=line.split(' ');
			for cell in cells:
				try:
					values.append(float(cell));
				except ValueError:
					continue;
		instances.append(values);
		digits=2;
		print "{0}{1:{2}}%".format(
						"\b" * (digits + 1+1), 
						int((percentindex+0.0)/len(files)*100),
						digits),
		sys.stdout.flush()
		percentindex+=1
		if(limit>=0):
			if(percentindex==limit):
				break;
	print "{0}{1:{2}}%".format(
						"\b" * (digits + 1+1), 
						100,
						digits)
	coords,img_dims=load_schema(schema);
	instance_min=min(min(instances[:]));
	instance_max=max(max(instances[:]));
	print("max:",instance_max)
	print("min",instance_min)
	instance_max-=instance_min;
	instances=(np.array(instances)-instance_min)/instance_max;
	return [instances,coords,img_dims,files];

def load_schema(schemapath):
	f=open(schemapath,'r');
	coords=[];
	line=f.readline();
	xy=line.split(' ');
	img_dims={'x':int(xy[0]),'y':int(xy[1])};
	line=f.readline();
	while(line!=""):
		xy=line.split(' ');
		try:
			coords.append((int(xy[0]),int(xy[1])));
		except ValueError:
			continue;
		line=f.readline();
	return coords,img_dims;
			