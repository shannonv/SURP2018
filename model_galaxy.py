#!/usr/bin/env python




import numpy as np
import matplotlib.pyplot as plt
import sys
import astropy.io.fits as ast



## Reads in magnetic field file
def read_box(infile,inshape,dtype="float32"):
    with open(infile,'rb') as f:
        #  C floats of 4 bytes aren't python floats which are really
        #  doubles at 8 bytes.  So spcify f4.
        if dtype=="float32":
            data=np.fromstring(f.read(),dtype='<f4')
        elif dtype=="float64":
            data=np.fromstring(f.read(),dtype='<f8')
        else:
            print("ERROR: don't know how to read type %s."%dtype)
            exit(1)
        return np.reshape(data,inshape)



infield=sys.argv[1]


## Choose frequency bandwidth and channel width
#freqs=np.arange(7.0e8,1.801e9,1.e6)
freqs=np.arange(2.e9,4.e9,1.e6)
#freqs=np.arange(7.0e8,4.e9,1.e6)


## Dimensions of magnetic field file
ydims=64
zdims=64
xdims=64


## Read in and separate magnetic field into x, y, and z components
vdim=3
box_in=read_box(infield,(xdims,ydims,zdims,vdim),dtype="float64")

Bx_model=box_in[:,:,:,0]
By_model=box_in[:,:,:,1]
Bz_model=box_in[:,:,:,2]


## Add random component to magnetic field
## Assumes a random component with average value of a few microGauss
Bx_model+=np.reshape(np.random.normal(loc=0.0,scale=1.0,size=xdims*ydims*zdims),(xdims,ydims,zdims))
By_model+=np.reshape(np.random.normal(loc=0.0,scale=1.0,size=xdims*ydims*zdims),(xdims,ydims,zdims))
Bz_model+=np.reshape(np.random.normal(loc=0.0,scale=1.0,size=xdims*ydims*zdims),(xdims,ydims,zdims))


## Electron density of galaxy
ne_cube=np.full((xdims,ydims,zdims),.01)



## Constants including spectral index (s), cosmic ray factors (Ci and Cp), and length of each 
## step in the galaxy (deltar)
s=3.
K=.812	       # rad m^-2 cm^-3 microG^-1 pc^-1
c=2.998e8      # m/s
wave=c/freqs   # m
Ci=1.
Cp=.7
deltar=1.e3    # pc



## Creates RM cube for galaxy
def RMcal(Bx,ne):
	
	Bpara=Bx
	delta_RM=K*ne*Bpara*deltar
	
	RM=np.cumsum(delta_RM[::-1],axis=0)[::-1]
	
	return RM



## Outputs Stokes parameters I, Q and U for galaxy
def Stokes(Bx,By,Bz,RM,freqs,wave):
	
	Bperp=(By**2+Bz**2)**(1/2)
	#alpha=np.arctan(Bz/By)
	
	#chi_0=45.*(np.pi/180)
	chi_0=0
	chi=RM*wave**2
	
	I_i=Ci*(Bperp**((1+s)/2))*(freqs**((1-s)/2))*deltar
	PI_i=Cp*(Bperp**((1+s)/2))*(freqs**((1-s)/2))*deltar

	
	I_sum=np.sum(I_i,axis=0)
	
	Q_i=PI_i*np.cos(2.*chi)
	U_i=PI_i*np.sin(2.*chi)
	Q_sum=np.sum(Q_i,axis=0)
	U_sum=np.sum(U_i,axis=0)
	
	#PA=.5*np.arctan(U_sum/Q_sum)
	
	return I_sum,Q_sum,U_sum




RM=RMcal(Bx_model,ne_cube)

Iimage=np.zeros((len(freqs),ydims,zdims),dtype=float)
Qimage=np.zeros((len(freqs),ydims,zdims),dtype=float)
Uimage=np.zeros((len(freqs),ydims,zdims),dtype=float)
for i in range(len(freqs)):
	StokesI,StokesQ,StokesU=Stokes(Bx_model,By_model,Bz_model,RM,freqs[i],wave[i])

	Iimage[i]=StokesI
	Qimage[i]=StokesQ
	Uimage[i]=StokesU



## Average the Stokes parameters for each frequency into three LOSs (horizontal)
avgI=np.zeros((len(freqs),3,1),dtype=float)
avgQ=np.zeros((len(freqs),3,1),dtype=float)
avgU=np.zeros((len(freqs),3,1),dtype=float)
for i in range(len(freqs)):
	ILOS1=np.average(Iimage[i][:][:21])
	ILOS2=np.average(Iimage[i][:][21:42])
	ILOS3=np.average(Iimage[i][:][42:64])
	avgI[i]=np.array([[ILOS1],[ILOS2],[ILOS3]])
	QLOS1=np.average(Qimage[i][:][:21])
	QLOS2=np.average(Qimage[i][:][21:42])
	QLOS3=np.average(Qimage[i][:][42:64])
	avgQ[i]=np.array([[QLOS1],[QLOS2],[QLOS3]])
	ULOS1=np.average(Uimage[i][:][:21])
	ULOS2=np.average(Uimage[i][:][21:42])
	ULOS3=np.average(Uimage[i][:][42:64])
	avgU[i]=np.array([[ULOS1],[ULOS2],[ULOS3]])



rmsNoise=.02/1e3

## Add error onto Stokes parameters
avgI+=np.random.normal(loc=0.0,scale=rmsNoise,size=avgI.shape)
avgQ+=np.random.normal(loc=0.0,scale=rmsNoise,size=avgQ.shape)
avgU+=np.random.normal(loc=0.0,scale=rmsNoise,size=avgU.shape)

## Create error arrays for Stokes parameters
errI=rmsNoise*np.random.normal(loc=1.0,scale=.05,size=avgI.shape)
errQ=rmsNoise*np.random.normal(loc=1.0,scale=.05,size=avgQ.shape)
errU=rmsNoise*np.random.normal(loc=1.0,scale=.05,size=avgU.shape)


Pimage=np.sqrt(avgQ**2+avgU**2)/avgI


## Save .dat files for each LOS - files can be used with RMtools_1D package
np.savetxt('galaxy_LOS0_1D.dat',np.column_stack([freqs,avgI[:,0],avgQ[:,0],avgU[:,0], errI[:,0],errQ[:,0],errU[:,0]]))
np.savetxt('galaxy_LOS1_1D.dat',np.column_stack([freqs,avgI[:,1],avgQ[:,1],avgU[:,1], errI[:,1],errQ[:,1],errU[:,1]]))
np.savetxt('galaxy_LOS2_1D.dat',np.column_stack([freqs,avgI[:,2],avgQ[:,2],avgU[:,2], errI[:,2],errQ[:,2],errU[:,2]]))


## Save frequency array
np.savetxt('freqs_Hz.dat',freqs)



## Header components for FITS files
head=ast.Header()
head['SIMPLE']='T'
head['BITPIX']=-32
head['NAXIS']=3
head['NAXIS1']=ydims
head['NAXIS2']=zdims
head['NAXIS3']=len(freqs)
head['EXTEND']='T'
head['CTYPE1']='Y-coordinate'
head['CRVAL1']=31
head['CDELT1']=deltar
head['CTYPE2']='Z-coordinate'
head['CRVAL2']=31
head['CDELT2']=deltar
head['CTYPE3']='FREQUENCY'
head['CRVAL3']=freqs[0]
head['CDELT3']=freqs[1]-freqs[0]
head['BUNIT']='mJy'
head['BTYPE']='intensity'
head['CELLSCAL']='CONSTANT'


## Create FITS files of Stokes parameters and fractional polarization
outfileI='galaxy_StokesI.fits'
outfileQ='galaxy_StokesQ.fits'
outfileU='galaxy_StokesU.fits'
outfileP='galaxy_P.fits'

ast.writeto(outfileI,avgI,head,output_verify='fix',overwrite=True)
ast.writeto(outfileQ,avgQ,head,output_verify='fix',overwrite=True)
ast.writeto(outfileU,avgU,head,output_verify='fix',overwrite=True)
ast.writeto(outfileP,Pimage,head,output_verify='fix',overwrite=True)






