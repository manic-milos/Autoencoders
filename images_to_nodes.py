from os import listdir, makedirs
from os.path import isfile, join, exists

import sys

mypath="termalmaps"
files=[f for f in listdir(mypath) if isfile(join(mypath,f))]
#create schema
filename=files[0];
f=open(mypath+"/"+filename,"r");
w=open("mapschema.csv","w");
width=0;
height=0;
line=f.readline();
coords=[];
while(line !=""):
	cells=line.split(' ');
	width=0;
	for cell in cells:
		if(cell!='NA'):
			try:
				floatvalue=float(cell);
				coords.append((width,height));
			except ValueError:
				continue;
		width+=1;
	height+=1;
	line=f.readline();
print len(coords);
print "width="+str(width);
print "height="+str(height);
w.write(str(width)+" "+str(height)+"\n");

for item in coords:
	w.write(str(item[0])+" "+str(item[1]));
	w.write("\n");

w.close();
converted_path="converted";
if not exists(converted_path):
    	makedirs(converted_path)
#convert files
for filename in files:
	f=open(mypath+"/"+filename,"r");
	w=open(converted_path+"/"+filename,"w");
	for line in f:
		cells=line.split(' ');
		for cell in cells:
			if(cell!='NA'):
				try:
					value=float(cell);
					w.write(cell+" ");
				except ValueError:
					continue;
	f.close();
	w.close();