# %% 2x upscaling
def get_lowres_coords(x,y,lowres_coords,highres_coords):
	li=-1;
	for i range(len(lowres_coords)):
		if(lowres_coords[i][0]==x//2 and lowres_coords[i][1]==y//2):
			li=i;
	if(li>=0):
		return li;
	li=get_lowres_coords(x-1,y);
	if(li>=0):
		return li;
	li=get_lowres_coords(x,y-1);
	if(li>=0):
		return li;
	li=get_lowres_coords(x-1,y-1);
	if(li<0):
		raise ValueError("Coordinates are NA in lowres");
	return li;