# Import libraries
import numpy as np
import bisect
import matplotlib.pyplot as plt
import math
import os

n_col = 2 # number of columns in plot
pop = 'ret_tcr' # pop name
g_list = [1,2,3,4] # list of g values according to file names
idc_list = [10] # list of idc values according to file name

# Spike raster plots
for ix,i_run in enumerate(idc_list):
    fig, ax = plt.subplots(math.ceil((len(idc_list)*len(g_list))/n_col),n_col,sharex=True,sharey=True)
    custom_xlim = (0,2000)
    plt.setp(ax, xlim=custom_xlim)
    fig.suptitle('Raster: {}, g = 0.1-0.4, idc = 10'.format(pop), fontsize=16) # title of plot

    for jx,j_run in enumerate(g_list):
        PATH = './snn_csv/TCR_spikes_' + str(j_run) + '_' + str(i_run) + '.csv'
        print(PATH)
        data = np.genfromtxt(PATH,delimiter=',') # load csv into numpy array
        i = int(jx / n_col)
        j = int(jx % n_col)
        print("idc:{},g:{}".format(i_run, j_run)) 
        ax[i,j].set_title('idc:{}, g:{}'.format(i_run,j_run))
        ax[i,j].scatter(data[:,1],data[:,0],s=1) # plot raster for run

plt.show()
