# sub function called by read_data 
# last modifed: gmalik, 03 June, 2021 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from matplotlib.pyplot import figure

class kde_plus_gmm():
    def __init__(self,XY,U_grid,U_hist,staz,binbin,jy,time_stamp,testing_file,wall_units, grid_points):      
        rsl = 500
        bins = binbin
        figure(figsize=(6.0, 4.0), dpi=rsl)
        
        data = np.reshape(U_hist,(1,-1))
        xx = np.linspace(np.min(data),np.max(data),500) #Evenly distributed velocity for the distribution curves only
                
        # ============ kde ============  
        kernel = gaussian_kde(data,bw_method='scott') # Create an object with methods and atrributes of the distribution

        pks = find_peaks(kernel.evaluate(xx),height = 0.5)[0] #Evaluate Method gives the y values for given x values so can plot the x vs evaluted values to plot the function
        xpk = [xx[jj] for jj in pks] #Finds x-cordinates of the peaks. The indexes of the peaks were found above
        
        nop = len(xpk) #Finds number of peaks and adds more if needed
        ypk = [kernel.evaluate(xx[jj]) for jj in pks] #Finds the the y value of the peaks
        xpk_sorted = -np.sort(-np.array(xpk)) # in decsending order   
        
        
        plt.hist(data.T,bins,density=True, edgecolor='grey',facecolor='none')# PLots the velocity pdf
        plt.plot(xx,kernel.evaluate(xx),'k-',label='kernal density estimation(KDE)') #Plots the calculated pdf curve
        plt.scatter(xpk,ypk,c="r",s=80,marker="o",label='peaks of KDE')  #Plots the peaks from the KDE
        
        # ============ gmm ============  
        N = np.arange(1, nop+1) #An array of possible number of UMZs based on no.of peaks from KDE 
        models = [None for i in range(len(N))]
        X = np.array(U_hist).T.reshape(-1,1) #Reshapes the velocity array for GMM algorithm
        for i in range(len(N)):
            mini = np.reshape(xpk_sorted[:N[i]],(-1,1)) #column array of x-cordinates of first N[i] peaks for initilization
            wini = np.array([1/(i+1) for ik in range(i+1)]) # equal weights initially 
            models[i] = GaussianMixture( n_components = N[i], means_init = mini,  weights_init = wini, tol = 1e-4 ).fit(X) #Creates an array of model objects each with different no of peaks

        # compute the AIC and the BIC criterion 
        AIC = [m.aic(X) for m in models]
        BIC = [m.bic(X) for m in models]

        M_best = models[np.argmin(BIC)] # or AIC (Chooses the model with has the lowest BIC)
        
        wall_points = int(80/400*grid_points)
        U_wall = U_grid[:wall_points][:]
        labels = M_best.predict(np.reshape(U_wall,(-1,1))) #Labels for the velcoities based on the best model
        labels = np.reshape(labels,U_wall.shape)
        
        #self.probabilities = M_best.predict_proba(U.reshape(-1,1)) #Gives the probability of being in the clusters rather than a label
        #self.probabilities= np.reshape(self.probabilities,(U.shape[0],U.shape[1],3))
                
        N_best = N[np.argmin(BIC)]  ## number of kernels/gaussian   components in best model
       
        self.means_g = []
        cov_g = [] #covariances or width of gaussians
       
        for ij in range(N_best):
            cov_g.append(M_best.covariances_[ij][0][0]) #Converting the resulting multidimensional array with empty dimensions to 1D
            self.means_g.append(M_best.means_[ij][0]) #THe result is a multidimensional array but only has values in one of the dimensions
        
        print('best #. of component',N_best)
        print('g_means',self.means_g)
        
        logprob = M_best.score_samples(xx.reshape(-1, 1)) #
        responsibilities = M_best.predict_proba(xx.reshape(-1, 1))
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis] #The newaxis just makes the pdf it a column vector because you need to multiply it with the probabilities which are arranged in a column fashion
             
        plt.plot(xx,pdf,'-',label='Gaussian Mixture Estimation')
        plt.plot(xx,pdf_individual,'--',label='individual Gaussian component')
       
        
        # ============ Graphing ============
        my_x_ticks = np.arange(0.4, 1.2, 0.1)
        my_y_ticks = np.arange(0, 13, 1)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
    
        plt.xlabel("Streamwise Velocity",fontdict={'family' : 'Calibri', 'size':12})
        plt.ylabel("Frequncy",fontdict={'family' : 'Calibri', 'size':12})
        plt.title("restart 010%s p.d.f. X#%d at Zlabel = %d with grid points = %d"%(time_stamp,jy,staz,grid_points),fontdict={'family' : 'Calibri', 'size':12})
        plt.legend(loc='upper left', prop={'family':'Calibri', 'size':10},frameon=False)
        
        ax=plt.gca();# get the handle of the axis 
        ax.spines['bottom'].set_linewidth(0.5);
        ax.spines['left'].set_linewidth(0.5);
        ax.spines['right'].set_linewidth(0.5);
        ax.spines['top'].set_linewidth(0.5);
        
        
        plt.savefig('/gpfs/fs0/scratch/j/jphickey/g2malik/working_code/grid_interpolation/Results/gridsize_convergence/010%s p.d.f. grid_points = %d.png'%(time_stamp,grid_points))
        plt.show()
        plt.close()
        
        # ===== Contour snapshot of flow ======
        figure(figsize=(20.0, 3.0), dpi=1000)
        plt.contourf(XY[0][:wall_points][:],XY[1][:wall_points][:],U_wall,40,cmap='hot')
        plt.contour(XY[0][:wall_points][:],XY[1][:wall_points][:wall_points],labels,2,colors='k')
        plt.title("010%s stream snapshot X#%d at Zlabel = %d with grid points = %d"%(time_stamp,jy,staz,grid_points),fontdict={'family' : 'Calibri', 'size':12})
        plt.savefig('/gpfs/fs0/scratch/j/jphickey/g2malik/working_code/grid_interpolation/Results/gridsize_convergence/010%s snapshot grid_points = %d.png'%(time_stamp,grid_points))
        plt.show()
        plt.close()
"""       
        # ============ plot BIC/AIC ============
        figure(num=10, figsize=(7.0, 4.0), dpi=rsl)        
        plt.plot(AIC)
        plt.plot(BIC)
        plt.title("AIC/BIC plot")
        plt.show()
        plt.close()
        
    

              
         ini = np.reshape(xpk,(-1,1))
         gmmModel = GaussianMixture(n_components=k, reg_covar=cov_th,tol=1e-6,max_iter=1000)
                                    ,means_init=ini)
                                    ,covariance_type='diag', reg_covar=3e-3)
                                    , means_init=ini)
         gmmModel.fit((np.array(U).T).reshape(-1,1))        
         wgts=gmmModel.weights_
         mu=gmmModel.means_
         cor=gmmModel.covariances_ 
         indx = np.arange(np.min(data),np.max(data),(np.max(data)-np.min(data))/500)
         yy   = 0 * indx  
         for ik in range(k):        
             zz  = wgts[ik]/np.sqrt(2*3.1416)/cor[ik][0]*np.exp(-(indx-mu[ik])**2/2/cor[ik][0]**2)
             yy += zz
             plt.plot(indx,zz,label='Kernel (%d in %d)'%(ik+1,k))          
         plt.plot(indx,yy,label='Synthetic p.d.f. ')
"""