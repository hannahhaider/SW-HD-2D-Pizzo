#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:51:47 2022

@author: hannahhaider

creating a script of helper functions for running Pizzo 2D governing equations 
Adapted from github.com/opaliss/3D_HD_SW
"""
#%% MASweb
"""Read in MAS results from PSI (Predictive Science Inc.) Website"""
from pathlib import Path
from parfive import Downloader
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import scipy
import time
from scipy.sparse import diags
import datetime as dt

def get_mas_path(cr, folder="hmi_mast_mas_std_0201"):
    """Get MAS website results.

    :param cr: carrington rotation. ex: 2210
    :return: mas_path
    """
    download_dir = Path.cwd() / '..' / '..' / 'data'
    
    cr_string = 'mas_helio/cr' + str(cr)
    mas_helio_dir = download_dir / cr_string 
    mas_helio_dir.mkdir(parents=True, exist_ok=True)

    base_url = 'http://www.predsci.com/data/runs/cr' + str(cr) + '-medium/' + folder + '/helio/{var}002.hdf'

    # Create a downloader to queue the files to be downloaded
    dl = Downloader()

    for var in ['rho', 'vp', 'vt', 'vr', 'br', 'bt', 'bp', 't', 'p']:
        file = mas_helio_dir / f'{var}002.hdf'
        if file.exists():
            continue
        else:
            remote_file = base_url.format(var=var)
            dl.enqueue_file(remote_file, path=mas_helio_dir)

    # Download the files
    if dl.queued_downloads > 0:
        dl.download()
    return mas_helio_dir.resolve()
#%% derivatives
def ddx_fwd(f, dx, periodic=True, order=1):
    """return the first derivative of f in x using a first-order, second-order, or 3rd order forward difference"""
    if order == 1:
        A = diags([-1, 1], [0, 1], shape=(f.shape[0], f.shape[0])).toarray()
        if periodic:
            A[-1, 0] = 1
        else:
            A[-1, -1] = 1
            A[-1, -2] = -1
        A /= dx
    elif order == 2:
        A = diags([-3/2, 2, -1/2], [0, 1, 2], shape=(f.shape[0], f.shape[0])).toarray()
        if periodic:
            A[-1, 0] = 2
            A[-1, 1] = -1/2
            A[-2, 0] = -1/2
        else:
            A[-1, -1] = 1/2
            A[-1, -2] = -2
            A[-1, -3] = 3/2
            A[-2, -1] = 1/2
            A[-2, -2] = -1/2
    elif order == 3:
        A = diags([-11/6, 3, -3/2, 1/3], [0, 1, 2, 3], shape=(f.shape[0], f.shape[0])).toarray()
        if periodic:
            A[-1, 0] = 3
            A[-1, 1] = -3/2
            A[-1, 2] = 1/3
            A[-2, 0] = -3/2
            A[-2, 1] = 1/3
        else:
            return ArithmeticError
        A /= (dx)
    return A @ f


def ddx_bwd(f, dx, periodic=False, order=1): 
    """return the first derivative of f in x using a first-order  or second-order backward difference"""
    if order == 1:
        A = diags([-1, 1], [-1, 0], shape=(f.shape[0], f.shape[0])).toarray()
        if periodic:
            A[0, -1] = -1
            A /= dx
        else:
            A[0, 0] = -1
            A[0, 1] = 1
        A /= dx
    elif order ==  2:
        A = diags([3/2, -2, 1/2], [0, -1, -2], shape = (f.shape[0], f.shape[0])).toarray()
        if periodic:
            A[0, 0] = 2
            A[0, 1] = -1/2
            A[1, 0] = -1/2
        else:
            A[0, 0] = 1/2
            A[0, 1] = -2
            A[0, 2] = 3/2
            A[1, 0] = 1/2
            A[1, 1] = -1/2
        A /= dx
    return A @ f

def ddx_central(f, dx, periodic=True, order = 2):
    """ return the first derivative of f in x using a first-order or a second order central difference"""
    if order == 1:
        A = diags([-1, 1], [-1, 1], shape=(f.shape[0], f.shape[0])).toarray()
        if periodic:
            A[0, -1] = -1
            A[-1, 0] = 1
        else:
            A[0, 0] = -3
            A[0, 1] = 4
            A[0, 2] = -1
            A[-1, -1] = 3
            A[-1, -2] = -4
            A[-1, -3] = 1
        A /= (2 * dx)
    elif order == 2:
        A = diags([1, -2, 1], [-1, 0, 1], shape = (f.shape[0], f.shape[0])).toarray()
        if periodic:
            A[0,-1] = 1
            A[-1,0] = 1
        A /= (dx**2)
            
    return A @ f
#%% diffusion function 
def diffusive(Cx, u, dx, dt, nx, nt):
    """this function returns the predictor/corrector of a diffusive term
    approximated by a second-order central difference.
    
    essentially creates a diffusion matrix for second derivatives approximated by central difference
    
    ------
    #parameters:
        u = solution vector of velocity 
        dx = spatial step
        dt = temporal step
        nx = number of spatial points
        nt = number of temporal points
    """
    Diff_matrix = Cx*(1/dx**2)*(2*np.diag(np.ones(nx)) - np.diag(np.ones(nx-1),-1)- np.diag(np.ones(nx-1),1))
    diff = Diff_matrix @ u
    return diff
#%% omni data reading 
def read_ascii_file(filename, index):
    """This function opens and reads an ascii file, storing the parameters in a data dictionary
    ----- 
    parameters:
        inputs: 'filename', index = -1 for the last dataset in the file
   data_dictionary = time, year, day, hour, minute, flow speed, proton density, flow pressure  
    
    """
    
    with open(filename) as f:
        data_dictionary = {'time':[], #initializing dictionary key 
                           'year':[],
                           'day':[],
                           'hour':[],
                           'minute':[],
                           'Flow Speed':[],
                           'Vx': [],
                           'Vy': [],
                           'Vz': [], 
                           'Proton Density': [],
                           'Flow Pressure':[],
                           'SC x': [],
                           'SC y': [],
                           'SC z': []
                          }  #creating a dictionary for our data      
        for line in f:
            tmp = line.split()
            data_dictionary["year"].append(int(tmp[0])) #appending dictionary keys with omni data
            data_dictionary["day"].append(int(tmp[1]))
            data_dictionary["hour"].append(int(tmp[2]))
            data_dictionary["minute"].append(int(tmp[3]))
            data_dictionary["Flow Speed"].append(float(tmp[4]))
            data_dictionary["Vx"].append(float(tmp[5]))
            data_dictionary["Vy"].append(float(tmp[6]))
            data_dictionary["Vz"].append(float(tmp[7]))
            data_dictionary["Proton Density"].append(float(tmp[8]))
            data_dictionary["Flow Pressure"].append(float(tmp[9]))
            data_dictionary["SC x"].append(float(tmp[10]))
            data_dictionary["SC y"].append(float(tmp[11]))
            data_dictionary["SC z"].append(float(tmp[index]))            
            #create datetime in each line
            time0 = dt.datetime(int(tmp[0]),1,1,int(tmp[2]),int(tmp[3]),0) 
            + dt.timedelta(days=int(tmp[1])-1)
            data_dictionary["time"].append(time0) #putting time in datetime format. can call isoformat()
     
    return data_dictionary
#%% ACE file reading
def read_ace_file(filename, index, nlines):
    """This function opens and reads an ascii file, storing the parameters in a data dictionary
    ----- 
    parameters:
        inputs: 'filename', index = -1 for the last dataset in the file
   data_dictionary = time, year, day, hour, minute, flow speed, proton density, flow pressure  
    
    """
    
    with open(filename) as f:
        data_dictionary = {'Vx': [],
                           'Vy': [],
                           'Vz': [], 
                           'SC x': [],
                           'SC y': [],
                           'SC z': []
                          }  #creating a dictionary for our data   
        # skip the header lines
        # nlines = 31
        for iLine in range(nlines):
            tmp = f.readline() # just read these lines
            print(tmp) # checking if all header lines are skipped/read in only 
        for line in f:
            tmp = line.split()
            data_dictionary["Vx"].append(float(tmp[0]))
            data_dictionary["Vy"].append(float(tmp[1]))
            data_dictionary["Vz"].append(float(tmp[2]))
            data_dictionary["SC x"].append(float(tmp[3]))
            data_dictionary["SC y"].append(float(tmp[4]))
            data_dictionary["SC z"].append(float(tmp[index]))            
     
    return data_dictionary
#%% cartesian to spherical transformation 
def cart_to_spher(x,y,z):
    """this function takes in cartesian coordinates x,y,z and 
    returns spherical coordinates radius,theta (latitude), and 
    phi (longtidue). in km, rad, rad 
    ------
    parameters:
        x, y, z 
        r, theta, phi
    
        """
    r = (x**2 + y**2 + z**2)**0.5 #radius 
    theta = np.arctan(y/x)
    phi = np.arctan((x**2 + y**2) / z**2)
    return r, theta, phi 
#%% conversion from gse to hg coordinates
def GSE_to_HG_vel(x, y, z, xdot, ydot, zdot):
    """this function takes in 6 parameters in GSE coordinates corresponding to ...
    ------
    parameters:
        [x,y,z] position in GSE (cartesian) coordinates 
        [xdot, ydot, zdot] velocity in GSE coordinates
    returns: 
        radial velocity vr, longitudinal velocity vt, and latitudinal velocity vp
    """
    vr = (x*xdot + y*ydot + z*zdot)/((x**2 + y**2 + z**2)**0.5)
    
    vt = (xdot*y - x*ydot)/(x**2 + y**2)
    
    vp = (z*(x*xdot + y*ydot) - (x**2 + y**2)*zdot)/((x**2 + y**2 + z**2)*((x**2 + y**2)**0.5))
    
    return vr, vt, vp 
#%% smoothing function - triangular moving average
def smooth_tma(data, degree):
    """this function computes a triangular moving average of data in a dataset, 
    effectively smoothing the data to rid of discontinuities, noise, etc.
    -----
    parameters:
        data - dataset as numpy array
        degree - degree of "smoothness" 
        
    returns:
        smoothed data 
    
    adapted from plotly.com
        """
    triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))
    smoothed = []
    
    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # boundary conditions
    smoothed = [smoothed[0]]*int(degree + degree/2) + smoothed
    
    while  len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed 
#%% theta and phi - omni 
"creating a function to extract phi and theta from velocities"
def theta_phi(Vx,Vy,Vz,V):
    """this function returns theta and phi given cartesian velocities and flow speed
    -----
    parameters:
        x velocity, Vx
        y velocity, Vy
        z velocity, Vz
        flow speed, V
    returns 
    phi [deg]
    theta [deg]
    """
    a_theta = Vz/V
    theta = 180 * np.arcsin(a_theta) / np.pi
    a_phi = Vy/-Vx
    phi = 180 * np.arctan(a_phi) / np.pi 
    return theta, phi
#%% psihdf
import pyhdf.SD as h4
import h5py as h5

def rdh5(h5_filename):
    x = np.array([])
    y = np.array([])
    z = np.array([])
    f = np.array([])

    h5file = h5.File(h5_filename, 'r')
    f = h5file['Data']
    dims = f.shape
    ndims = np.ndim(f)

    #Get the scales if they exist:
    for i in range(0,ndims):
        if i == 0:
            if (len(h5file['Data'].dims[0].keys())!=0):
                x = h5file['Data'].dims[0][0]
        elif i == 1:
            if (len(h5file['Data'].dims[1].keys())!=0):
                y = h5file['Data'].dims[1][0]
        elif i == 2:
            if (len(h5file['Data'].dims[2].keys())!=0):
                z = h5file['Data'].dims[2][0]

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    f = np.array(f)

    h5file.close()

    return (x,y,z,f)

def rdhdf(hdf_filename):

    if (hdf_filename.endswith('h5')):
        x,y,z,f = rdh5(hdf_filename)
        return (x,y,z,f)

    x = np.array([])
    y = np.array([])
    z = np.array([])
    f = np.array([])

    # Open the HDF file
    sd_id = h4.SD(hdf_filename)

    #Read dataset.  In all PSI hdf4 files, the
    #data is stored in "Data-Set-2":
    sds_id = sd_id.select('Data-Set-2')
    f = sds_id.get()

    #Get number of dimensions:
    ndims = np.ndim(f)

    # Get the scales. Check if theys exist by looking at the 3rd
    # element of dim.info(). 0 = none, 5 = float32, 6 = float64.
    # see http://pysclint.sourceforge.net/pyhdf/pyhdf.SD.html#SD
    # and http://pysclint.sourceforge.net/pyhdf/pyhdf.SD.html#SDC
    for i in range(0,ndims):
        dim = sds_id.dim(i)
        if dim.info()[2] != 0:
            if i == 0:
                x = dim.getscale()
            elif i == 1:
                y = dim.getscale()
            elif i == 2:
                z = dim.getscale()

    sd_id.end()

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    f = np.array(f)

    return (x,y,z,f)

def rdhdf_1d(hdf_filename):

    x,y,z,f = rdhdf(hdf_filename)

    return (x,f)

def rdhdf_2d(hdf_filename):

    x,y,z,f = rdhdf(hdf_filename)

    if (hdf_filename.endswith('h5')):
        return(x,y,f)
    return (y,x,f)

def rdhdf_3d(hdf_filename):

    x,y,z,f = rdhdf(hdf_filename)
    if (hdf_filename.endswith('h5')):
        return(x,y,z,f)

    return (z,y,x,f)

def wrh5(h5_filename, x, y, z, f):

    h5file = h5.File(h5_filename, 'w')

    # Create the dataset (Data is the name used by the psi data)).
    h5file.create_dataset("Data", data=f)

    # Make sure the scales are desired by checking x type, which can
    # be None or None converted by np.asarray (have to trap seperately)
    if x is None: 
        x = np.array([], dtype=f.dtype)
        y = np.array([], dtype=f.dtype)
        z = np.array([], dtype=f.dtype)
    if x.any() == None:
        x = np.array([], dtype=f.dtype)
        y = np.array([], dtype=f.dtype)
        z = np.array([], dtype=f.dtype)

    # Make sure scales are the same precision as data.
    x=x.astype(f.dtype)
    y=y.astype(f.dtype)
    z=z.astype(f.dtype)

    #Get number of dimensions:
    ndims = np.ndim(f)

    #Set the scales:
    for i in range(0,ndims):
        if i == 0 and len(x) != 0:
            dim = h5file.create_dataset("dim1", data=x)
            h5file['Data'].dims.create_scale(dim,'dim1')
            h5file['Data'].dims[0].attach_scale(dim)
            h5file['Data'].dims[0].label = 'dim1'
        if i == 1 and len(y) != 0:
            dim = h5file.create_dataset("dim2", data=y)
            h5file['Data'].dims.create_scale(dim,'dim2')
            h5file['Data'].dims[1].attach_scale(dim)
            h5file['Data'].dims[1].label = 'dim2'
        elif i == 2 and len(z) != 0:
            dim = h5file.create_dataset("dim3", data=z)
            h5file['Data'].dims.create_scale(dim,'dim3')
            h5file['Data'].dims[2].attach_scale(dim)
            h5file['Data'].dims[2].label = 'dim3'

    # Close the file:
    h5file.close()

def wrhdf(hdf_filename, x, y, z, f):

    if (hdf_filename.endswith('h5')):
        wrh5(hdf_filename, x, y, z, f)
        return

    # Create an HDF file
    sd_id = h4.SD(hdf_filename, h4.SDC.WRITE | h4.SDC.CREATE | h4.SDC.TRUNC)

    if f.dtype == np.float32:
        ftype = h4.SDC.FLOAT32
    elif f.dtype == np.float64:
        ftype = h4.SDC.FLOAT64

    # Create the dataset (Data-Set-2 is the name used by the psi data)).
    sds_id = sd_id.create("Data-Set-2", ftype, f.shape)

    #Get number of dimensions:
    ndims = np.ndim(f)

    # Make sure the scales are desired by checking x type, which can
    # be None or None converted by np.asarray (have to trap seperately)
    if x is None: 
        x = np.array([], dtype=f.dtype)
        y = np.array([], dtype=f.dtype)
        z = np.array([], dtype=f.dtype)
    if x.any() == None:
        x = np.array([], dtype=f.dtype)
        y = np.array([], dtype=f.dtype)
        z = np.array([], dtype=f.dtype)

    #Set the scales (or don't if x is none or length zero)
    for i in range(0,ndims):
        dim = sds_id.dim(i)
        if i == 0 and len(x) != 0:
            if x.dtype == np.float32:
                stype = h4.SDC.FLOAT32
            elif x.dtype == np.float64:
                stype = h4.SDC.FLOAT64
            dim.setscale(stype,x)
        elif i == 1  and len(y) != 0:
            if y.dtype == np.float32:
                stype = h4.SDC.FLOAT32
            elif y.dtype == np.float64:
                stype = h4.SDC.FLOAT64
            dim.setscale(stype,y)
        elif i == 2 and len(z) != 0:
            if z.dtype == np.float32:
                stype = h4.SDC.FLOAT32
            elif z.dtype == np.float64:
                stype = h4.SDC.FLOAT64
            dim.setscale(stype,z)

    # Write the data:
    sds_id.set(f)

    # Close the dataset:
    sds_id.endaccess()

    # Flush and close the HDF file:
    sd_id.end()


def wrhdf_1d(hdf_filename,x,f):

    x = np.asarray(x)
    y = np.array([])
    z = np.array([])
    f = np.asarray(f)
    wrhdf(hdf_filename,x,y,z,f)


def wrhdf_2d(hdf_filename,x,y,f):

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.array([])
    f = np.asarray(f)
    if (hdf_filename.endswith('h5')):
        wrhdf(hdf_filename,x,y,z,f)
        return
    wrhdf(hdf_filename,y,x,z,f)


def wrhdf_3d(hdf_filename,x,y,z,f):

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    f = np.asarray(f)
    if (hdf_filename.endswith('h5')):
        wrhdf(hdf_filename,x,y,z,f)
        return
    wrhdf(hdf_filename,z,y,x,f)
#%%psihdf4
from pyhdf.SD import *

def rdhdf(hdf_filename):
    """
    Read an HDF4 file and return the scales and data values.

    str: hdf_filename
        HDF4 filename.

    tuple:
        List of scale and data values.
    """
    x = np.array([])
    y = np.array([])
    z = np.array([])
    f = np.array([])

    # Open the HDF file
    sd_id = SD(hdf_filename)

    #Read dataset.  In all PSI hdf4 files, the 
    #data is stored in "Data-Set-2":
    sds_id = sd_id.select('Data-Set-2')
    f = sds_id.get()
    
    #Get number of dimensions:
    ndims = np.ndim(f)
    
    #Get the scales:
    for i in range(0,ndims):
        dim = sds_id.dim(i)
        if i == 0:
            x = dim.getscale()
        elif i == 1:
            y = dim.getscale()
        elif i == 2: 
            z = dim.getscale()

    sd_id.end()

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    f = np.array(f) 

    return (x,y,z,f)

def rdhdf_1d(hdf_filename):

    x,y,z,f = rdhdf(hdf_filename)

    return (x,f)

def rdhdf_2d(hdf_filename):

    x,y,z,f = rdhdf(hdf_filename)

    return (y,x,f)

def rdhdf_3d(hdf_filename):

    x,y,z,f = rdhdf(hdf_filename)

    return (z,y,x,f)


def wrhdf(hdf_filename, x, y, z, f):
    """
    Write an HDF4 file. x, y, and z are the scales. f is the data.

    str: hdf_filename
        HDF4 filename.

    """

    # Create an HDF file
    sd_id = SD(hdf_filename, SDC.WRITE | SDC.CREATE | SDC.TRUNC)

    if f.dtype == np.float32:
        ftype = SDC.FLOAT32
    elif f.dtype == np.float64:
        ftype = SDC.FLOAT64

    # Create the dataset (Data-Set-2 is the name used by the psi data)).
    sds_id = sd_id.create("Data-Set-2", ftype, f.shape)

    #Get number of dimensions:
    ndims = np.ndim(f)

    #Set the scales:
    for i in range(0,ndims):
        dim = sds_id.dim(i)
        if i == 0:
            if x.dtype == np.float32:
                stype = SDC.FLOAT32
            elif x.dtype == np.float64:
                stype = SDC.FLOAT64
            dim.setscale(stype,x)
        elif i == 1:
            if y.dtype == np.float32:
                stype = SDC.FLOAT32
            elif y.dtype == np.float64:
                stype = SDC.FLOAT64
            dim.setscale(stype,y)
        elif i == 2: 
            if z.dtype == np.float32:
                stype = SDC.FLOAT32
            elif z.dtype == np.float64:
                stype = SDC.FLOAT64
            dim.setscale(stype,z)

    # Write the data:
    sds_id.set(f)

    # Close the dataset:
    sds_id.endaccess()

    # Flush and close the HDF file:
    sd_id.end()


def wrhdf_1d(hdf_filename,x,f):

    x = np.asarray(x)
    y = np.array([])
    z = np.array([])
    f = np.asarray(f)
    wrhdf(hdf_filename,x,y,z,f)


def wrhdf_2d(hdf_filename,x,y,f):

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.array([])
    f = np.asarray(f)
    wrhdf(hdf_filename,y,x,z,f)


def wrhdf_3d(hdf_filename,x,y,z,f):

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    f = np.asarray(f)
    wrhdf(hdf_filename,z,y,x,f)
#%% 



