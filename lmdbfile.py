#coding=utf-8
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import shutil
import  random
import  math
from  augment import  augmentimg
import  dlib
from cython.parallel import prange
face_detector=dlib.get_frontal_face_detector()
def getface(imgpath,cropimgname):

	bgrImg = cv2.imread(imgpath)
	if bgrImg is None:
		return False
	print bgrImg.shape
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)


	#img = io.imread('1.jpg')
	faces = face_detector(rgbImg, 1)
	if len(faces) <=0:
		return False
	face=max(faces, key=lambda rect: rect.width() * rect.height())
	[x1,x2,y1,y2]=[face.left(),face.right(),face.top(),face.bottom()]
	img = bgrImg
	height, weight =np.shape(img)[:2]
	x=int(x1)
	y=int(y1)
	w=int(x2-x1)
	h=int(y2-y1)
	scale=0.4
	miny=max(0,y-scale*h)
	minx=max(0,x-scale*w)
	maxy=min(height,y+(1+scale)*h)
	maxx=min(weight,x+(1+scale)*w)
	roi=img[miny:maxy,minx:maxx]
	rectshape=roi.shape
	maxlenght=max(rectshape[0],rectshape[1])
	img0=np.zeros((maxlenght,maxlenght,3))
	img0[(maxlenght*.5-rectshape[0]*.5):(maxlenght*.5+rectshape[0]*.5),(maxlenght*.5-rectshape[1]*.5):(maxlenght*.5+rectshape[1]*.5)]=roi

	cv2.imwrite(cropimgname,img0)

	return  True
#对训练数据标人脸准化裁剪
def stdcrop():
	gender_listsrc=['0','1','2','3','4','5']
	for sf in gender_listsrc:
		sfimgs=os.listdir(sf)
		cropfile='crop'+sf
		if os.path.exists(cropfile) is False:
			os.mkdir(cropfile)
		for i in prange(len(sfimgs)):#prange多线程并行
			imgpath=sfimgs[i]
			oldpath=sf+'//'+imgpath
			newpath='crop'+oldpath
			getface(oldpath,newpath)





def GetFileList(FindPath,FlagStr=[]):      
	import os
	FileList=[]
	FileNames=os.listdir(FindPath)
	if len(FileNames)>0:
		for fn in FileNames:
			if len(FlagStr)>0:
				if IsSubString(FlagStr,fn):
					fullfilename=os.path.join(FindPath,fn)
					FileList.append(fullfilename)
			else:
				fullfilename=os.path.join(FindPath,fn)
				FileList.append(fullfilename)

   
	if len(FileList)>0:
		FileList.sort()

	return FileList
def IsSubString(SubStrList,Str):      
	flag=True
	for substr in SubStrList:
		if not(substr in Str):
			flag=False

	return flag
#因为caffe中,不允许文件名中有空格,所有需要重命名去除空格
def stdrename(imgfiles):
	newimgfiles=[]
	for l in imgfiles:
		x_list=l.split(' ')
		y = ''.join(x_list)
		os.rename(l,y)
		newimgfiles.append(y)
	return newimgfiles




def writetrainlist():
	txttrain=open('train.txt','w')
	txtval=open('val.txt','w')
	filename=['crop0','crop1','crop2','crop3','crop4','crop5']
	#filename=['batch1//black','batch1//brown','batch1//white','batch1//yellow']
	#filename=['batch2//black','batch2//brown','batch2//white','batch2//yellow']
	#filename=['test']
	numsample=4000
	nval=20
	strtrain=''
	strval=''
	for i,f in enumerate(filename):
		imgfiles=GetFileList(f)
		random.shuffle(imgfiles)
		imgfiles=stdrename(imgfiles)#caffe 文件名不允许有空格
		imgfvals=imgfiles[:nval]
	#验证数据文件列表
		for f in imgfvals:
			strval=strval+f+' '+str(i)+'\n'


	#训练数据文件列表
		imgfiles=imgfiles[nval:]
		for j,img in enumerate(imgfiles):
			if j>numsample:
				break
			strtrain=strtrain+img+' '+str(i)+'\n'
		#如果数据数少于numsample,那么我们就启用数据扩充
		for j in range(int(numsample-len(imgfiles))):
			print 'augment:%s' %str(j)
			imgpath=random.sample(imgfiles, 1)[0]
			newpath=os.path.dirname(imgpath)+'/augment'+str(random.randint(0,1000))+os.path.basename(imgpath)
			augmentimg(imgpath,newpath,type=None)
			strtrain=strtrain+newpath+' '+str(i)+'\n'

	txttrain.write(strtrain)
	txttrain.close()
	txtval.write(strval)
	txtval.close()



#stdcrop()
writetrainlist()


