# --- It is actually an ipynb file but made into a .py file because of JSON error
# aims: implement the GMM and the KDE to planar snapshots 
# calls: kde_plus_gmm_gagan 
# modefication history: gmalik, 31 May, 2021; 

# --------------------------------
# import libraries 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import time 
from scipy.interpolate import griddata 
from pylab import *
from kde_plus_gmm import kde_plus_gmm 
# --------------------------------
# Read data from file
path = '/gpfs/fs0/scratch/j/jphickey/jphickey/Boundary_Layer_PNAS_2017/'

time_stamp = '13'
fname = path + 'restart_010' + time_stamp + '_ydelta_adrian_scalar_omega_uvw_08240_10565.dat'
testing_file = '/gpfs/fs0/scratch/j/jphickey/g2malik/working_code/grid_interpolation/testf.dat' # temporary file for debugging
t = open(testing_file, "w")
# --------------------------------
# choose the starting position/parameters 
btfti = 0.04 # see Wu, Wallace and Hickey 2019, PoF 
binbin = 50 #50

start_z = 1 # start from the position at Z-label = ? {1...513}
xnb = 1     # start from the position at X-label = ? {1...2326}

# --------------------------------
#for aa in range(50):
#    staz = aa * 10 + start_z  # get a slide of data every 10 points in z
#    for x_plane in range(5):  # we can have at most 5 planes in our dataset, each 2000 wall units long.
staz = 41
x_plane = 4

f   = open (fname, mode = 'r')
jy = x_plane+xnb # ! 
# --------------------------------
stat = 0   # if you like to skip some beginning points 
nox  = 440 # 440 points is the length of 2000 wall units
wall_units = 50/11*nox
stax = stat + nox * (jy-1) # [1,2326] + 440*(jy-1)
skpx = 2326 - nox 
endx = stax + nox 

stay = 105   # [1,400] skip the buffer layer -- 100 wall units 
endy = 400   # 
noy = endy - stay + 1

# --------------------------------
# skip / go to certain index (I,J,K)
for i in range(3):
    data = f.readline()
for ii in range(stax-1):
    data = f.readline()
for jj in range(stay-1):
    for ii in range(2326):
        data = f.readline()
for kk in range(staz-1):
    for jj in range(400):
        for ii in range(2326):
            data = f.readline()
# --------------------------------
# get velocity 
xy    = [[],[]] # computational grid
uu    = []      # computational grid
tt    = []
small = 1e-15

for j in range(noy):
    ylb = stay + j + 1 # y-label grid 
    for i in range(nox):
        data = f.readline()
        lst = data.split()
        x = float(lst[0]) - 10842.4  # minus the starting position at re_theta = 1800 
        y = float(lst[1])
        yod = float(lst[3])
        xod = x / (y+small) * yod
        tl = float(lst[5]) # passive scalar, index of btfti 
        u = float(lst[7])
        xy[0].append(xod)
        xy[1].append(yod)
        uu.append(u)
        tt.append(tl)
    for i in range(skpx): # skip the x that we don't need 
        data = f.readline()

# --------------------------------
# Initialize gaussian mean array
gaussian_means = np.zeros((4,0))
grid_array = np.arange(100,300,100)

for grid_size in grid_array:
    # --------------------------------
    # interpolation to uniform grid    
    xmax = xy[0][-1]
    xmin = xy[0][0]
    ymax = xy[1][-1]
    ymin = xy[1][0]

    interpx = grid_size  # number of pts 
    interpy = grid_size
    xi=np.linspace(xmin,xmax,interpx)
    yi=np.linspace(ymin,ymax,interpy)

    XY  = np.meshgrid(xi,yi)
    UU  = griddata((xy[0],xy[1]), uu, (XY[0],XY[1]), method =  'cubic')
    TT  = griddata((xy[0],xy[1]), tt, (XY[0],XY[1]), method =  'cubic')

    # ----------------------------------
    # prepare the data for the histogram 
    uhis = []

    for i in range(interpx):
        for j in range(interpy):
            # detection of the turbultent region; BTFTI 
            lll = TT[i][j]
            if lll > btfti :
                uhis.append(UU[i][j]) #uhis doesnt include turbulent region but UU does
    #np.savetxt(testing_file,xy)
    # --------------------------------
    # GMM - main 
    GMM = kde_plus_gmm(XY,UU,uhis,staz,binbin,jy,time_stamp,testing_file,wall_units,interpx)
    gaussian_means = np.append(gaussian_means, np.reshape(GMM.means_g, (4,1)) , axis=1)
        
#---------------------

for i in range(np.shape(gaussian_means[:,0])[0]):
    plt.plot (grid_array,gaussian_means[i,:])
    plt.xlabel("No. of grid points")
    plt.ylabel("Gaussian component mean value")
    plt.title("Gaussian component #%d"%(i+1))
    plt.show()
    plt.close()

t.close()
print ('--- End of all ---')
