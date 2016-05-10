# import upscaling;
import data_loading as dl;

[instances,
	coords,
	img_dims,
	onlyfiles]=dl.load_nodes_with_schema("highres",
	"mapschemahighres.csv");