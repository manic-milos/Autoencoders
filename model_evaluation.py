import numpy as np
import math
from datetime import date
def euclid(a,b):
	return math.sqrt(sum((np.array(a)-np.array(b))**2))

def dayspan(a,b):
	date1=a.split('.')[0]
	date2=b.split('.')[0]
	month1=date1[4:6]
	month2=date2[4:6]
	day1=date1[6:]
	day2=date2[6:]
	date1=date(1990,int(month1),int(day1))
	date2=date(1990,int(month2),int(day2))
	date11=date(1991,int(month1),int(day1))
	date12=date(1989,int(month1),int(day1))
	return min(abs((date1-date2).days),(date11-date2).days,(date2-date12).days)

def compression_loss(testset,recon):
	compression_loss=0;
	for i in range(len(testset)):
		img_compression_loss=0;
		for j in range(len(testset[i])):
			img_compression_loss+=abs(testset[i][j]-recon[i][j])
		img_compression_loss/=len(testset[i])
		compression_loss+=img_compression_loss
	compression_loss/=len(testset)
	return compression_loss;

def correlation(testset,representations):
	original_img_distances=[];
	representation_distances=[];
	for i in range(len(testset)):
		for j in range(len(testset)):
			if(i!=j):
				original_img_distances.append(
					euclid(testset[i],testset[j]))
				representation_distances.append(
					euclid(representations[i],representations[j]))
	return np.corrcoef(
		original_img_distances, representation_distances)[0, 1]

def indexTest(testset,representations):
	indexTestSum=0
	for i in range(len(testset)):
		sortedRepresentationDist=[]
		minOriginalDist=10000000;
		minOriginalDisti=-1;
		for j in range(len(testset)):
			if(i!=j):
				original_dist=euclid(
					testset[i],testset[j])
				representation_dist=euclid(
					representations[i],representations[j])
				sortedRepresentationDist.append((j,representation_dist))
				if(original_dist<minOriginalDist):
					minOriginalDist=original_dist
					minOriginalDisti=j
		sortedRepresentationDist.sort(key=lambda x:x[1])
		for si in range(len(sortedRepresentationDist)):
			if(sortedRepresentationDist[si][0]==minOriginalDisti):
				indexTestSum+=si+1;
	return indexTestSum/len(testset)

def closestRepresentationList(testset,representations):
	closestRepresentations=[]
	for i in range(len(testset)):
		closestI=-1;
		closestDistance=1000000;
		for j in range(len(testset)):
			if(i!=j):
				distance=euclid(representations[i],representations[j])
				if(distance<closestDistance):
					closestDistance=distance;
					closestI=j
		closestRepresentations.append((i,closestI,closestDistance))
	return sorted(closestRepresentations,key=lambda x: x[2])

def closestRepCorrelation(testset,representations):
	closestRepresentations=[]
	original_distance=[]
	for i in range(len(testset)):
		closestI=-1;
		closestDistance=1000000;
		for j in range(len(testset)):
			if(i!=j):
				distance=euclid(representations[i],representations[j])
				if(distance<closestDistance):
					closestDistance=distance;
					closestI=j
		closestRepresentations.append(closestDistance)
		original_distance.append(euclid(testset[i],testset[closestI]))
	return np.corrcoef(
		closestRepresentations, original_distance)[0, 1]

def cosine_sim(a,b):
	sim=0;
	for i in range(len(a)):
		sim+=a[i]*b[i];
	a_norm=0;
	for i in range(len(a)):
		a_norm+=a[i]*a[i];
	b_norm=0;
	for i in range(len(a)):
		b_norm+=b[i]*b[i];
	a_norm=math.sqrt(a_norm)
	b_norm=math.sqrt(b_norm)
	sim/=a_norm*b_norm;
	return sim;
