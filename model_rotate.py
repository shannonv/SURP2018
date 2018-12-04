#!/usr/bin/env python



import numpy as np
import matplotlib.pyplot as plt
import sys
import astropy.io.fits as ast
import pickle



## Read in magnetic field file
## This piece of program thanks to C. Purcell's RMtools package.
def read_box(infile,inshape,dtype="float32"):
    with open(infile,'rb') as f:
        #  C floats of 4 bytes aren't python floats which are really
        #  doubles at 8 bytes.  So specify f4.
        if dtype=="float32":
            data=np.fromstring(f.read(),dtype='<f4')
        elif dtype=="float64":
            data=np.fromstring(f.read(),dtype='<f8')
        else:
            print("ERROR:  don't know how to read type %s." % dtype)
            exit(1)
        return np.reshape(data,inshape)



## Dimensions of magnetic field file
xdims=64
ydims=64
zdims=64

infield=sys.argv[1]


## Read in and separate magnetic field into x, y, and z components
vdim=3
box_in=read_box(infield,(xdims,ydims,zdims,vdim),dtype="float64")

Bx_model=box_in[:,:,:,0]
By_model=box_in[:,:,:,1]
Bz_model=box_in[:,:,:,2]



## Uncomment if you want to add random component of magnetic field here instead of in 
## model_galaxy.py
'''
Bx_model+=np.reshape(np.random.normal(loc=0.0,scale=1.0,size=xdims*ydims*zdims),(xdims,ydims,zdims))
By_model+=np.reshape(np.random.normal(loc=0.0,scale=1.0,size=xdims*ydims*zdims),(xdims,ydims,zdims))
Bz_model+=np.reshape(np.random.normal(loc=0.0,scale=1.0,size=xdims*ydims*zdims),(xdims,ydims,zdims))
'''


## transforms coordinates of galaxy
def CoordTransform(BX,BY,BZ,theta,phi):
		
	B_x=BX*np.cos(phi)*np.cos(theta)+BY*np.cos(theta)*np.sin(phi)+BZ*np.sin(theta)
	B_y=BY*np.cos(phi)-BX*np.sin(phi)
	B_z=BZ*np.cos(theta)-BX*np.cos(phi)*np.sin(theta)-BY*np.sin(phi)*np.sin(theta)
	
	return B_x,B_y,B_z




## Transform coordinates of magnetic field to change LOS
## Depending the orientation you want, may need to rotate in the phi direction first 
## and then in the theta direction
Bx,By,Bz=CoordTransform(Bx_model,By_model,Bz_model,0*(np.pi/180.),90*(np.pi/180.))
#Bx,By,Bz=CoordTransform(Bx,By,Bz,90*(np.pi/180.),0*(np.pi/180.))


## Rotate the magnetic field component cube to change LOS
## theta: axes=(2,1); phi: axes=(2,0)
Bx=np.rot90(Bx,k=1,axes=(2,0))
Bz=np.rot90(Bz,k=1,axes=(2,0))
By=np.rot90(By,k=1,axes=(2,0))



## Create new, rotated magnetic field file
Bx=Bx[:,:,:,np.newaxis]
By=By[:,:,:,np.newaxis]
Bz=Bz[:,:,:,np.newaxis]
field=np.concatenate((Bx,By,Bz),axis=3)
outarray=np.ravel(field)

f=open('m0_rot_phi90.p','wb')
f.write(outarray.tobytes())
f.close()








