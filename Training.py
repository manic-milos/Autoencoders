import tensorflow as tf;

class Training:
	'Class for training the autoencoder'
	learning_rate=tf.Variable(0.00001,
		name="learning_rate",
		dtype=tf.float32);
	batch_size=tf.Variable(25,name="batch_size");
	start_epoch=tf.Variable(0,name="start_epoch");
	def __init__(self,load_filename):
		print "training initialized";

	def load_from filename(filename):
		loader=tf.train.Saver(var_list=[
			learning_rate,
			batch_size,
			start_epoch]);
		