# import upscaling;
import data_loading as dl;
import tensorflow as tf;
import matplotlib.pyplot as plt;
import numpy as np;

[instances,
	coords,
	img_dims,
	onlyfiles]=dl.load_nodes_with_schema("highres",
	"mapschemahighres.csv");
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

