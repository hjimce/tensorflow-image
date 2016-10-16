#coding=utf-8
#对输入的图片，变换填充成正方形，并resize到256*256
import cv2
import os
from multiprocessing import  Pool
import  numpy as np
def squartimage(path):
	oldpath=path[0]
	newpath=path[1]
	bgrImg = cv2.imread(oldpath)

	if bgrImg is None:
		return
	rectshape=bgrImg.shape
	maxlenght=max(rectshape[0],rectshape[1])
	img0=np.ones((maxlenght,maxlenght,3))*255
	img0[int(maxlenght*.5-rectshape[0]*.5):int(maxlenght*.5+rectshape[0]*.5),int(maxlenght*.5-rectshape[1]*.5):int(maxlenght*.5+rectshape[1]*.5)]=bgrImg
	print newpath
	img0=cv2.resize(img0,(256,256))
	cv2.imwrite(newpath,img0)

def getimagepath(root):
	class_file=os.listdir(root)
	imagepaths={}
	for c in class_file:
		cpath=os.path.join(root,c)
		imagepaths[cpath]=os.listdir(cpath)
	return  imagepaths
def run(root):
	imagepaths=getimagepath(root)
	oldpaths=[]
	newpaths=[]
	for p in imagepaths:
		newp='new'+p
		if os.path.exists(newp) is False:
			os.makedirs(newp)
		for f in imagepaths[p]:

			oldpaths.append(os.path.join(p,f))
			newpaths.append(os.path.join(newp,f))

	#print oldpaths
	#print newpaths
	pool=Pool()
	pool.map(squartimage,zip(oldpaths,newpaths))


run('race')


