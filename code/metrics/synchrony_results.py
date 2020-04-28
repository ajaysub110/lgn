## RASTER PLOT AND SYNCHRONY MEASURE

# Import libraries
import numpy as np
import bisect
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import math
import os


def find_le(a,x):
    # Find rightmost value less than or equal to x
    i = bisect.bisect_right(a, x)
    if i:
        return i
    return 0

def compute_phi_t(t,sp_times_i,isp_low,T):
    # computing instantaneous phase (Eqn. 6 from Khoshkhou paper)
    t_low, t_hi = 0,0
    if isp_low > 0:
        t_low = sp_times_i[isp_low-1]
        if isp_low + 1 < len(sp_times_i):
            t_hi = sp_times_i[isp_low]
        else:
            t_hi = T
    else:
        t_hi = sp_times_i[isp_low]
        t_low = 0.0
    return 2.*np.pi*((t-t_low)/(t_hi-t_low))

def synchrony_measure(csv_path,T=1000,t_res=0.1):
    """
    # Synchrony Measure
    ## Task:
    Implement the synchrony measure used in the paper for each population

    ## Input:
    Separate CSV file for each population with 2 columns; neuron index and spike time

    ## Algorithm
    1. Read CSV file into list of N (neurons in pop) lists each with corresponding spike times
    2. initialize S as array with dimension T/res
    3. For timestep t (with res of 0.1 since Izhikevich):
      - S(t) = 0
      - for each neuron i
        - calculate phi of i at t
        - for each neuron j not including i
          - calculate phi of j at t
          - Add to S(t)
      - append S(t) to S
    4. Average S
    5. Multiply by 2/N(N-1)
    6. Report result
    """
    data = np.genfromtxt(csv_path,delimiter=',')
    sp_times = [[]]
    n_neurons = int(data[-1,0]) + 1
    cur_i = 0
    for i in range(len(data)):
        if int(data[i,0]) != cur_i:
            sp_times.append([])
            cur_i = len(sp_times) - 1
        sp_times[cur_i].append(data[i,1])

    S = []
    for t in np.linspace(0,T,T/t_res):
        S_t = 0.0
        for i in range(n_neurons):
            isp_low_i = find_le(sp_times[i],t)
            phi_it = compute_phi_t(t,sp_times[i],isp_low_i,T)
            for j in range(n_neurons):
                if j == i:
                    continue
                else:
                    isp_low_j = find_le(sp_times[j],t)
                    phi_jt = compute_phi_t(t,sp_times[j],isp_low_j,T)
                    S_t += np.cos((phi_it-phi_jt)/2.0)**2
        S.append(S_t)
    return (2/(n_neurons*(n_neurons-1))) * np.mean(S)

N = 3
M = 10
c = 2
pop = 'ret_tcr'
g_list = [0.1, 0.5, 1]
idc_list = list(range(5,15))

'''
if os.path.exists('sync_scores.txt'):
    os.remove('sync_scores.txt')
f=open('./sync_scores.txt','a')
# Synchrony measure
scores = []
for i_run,g in enumerate(g_list):
    for j_run in range(M):
        PATH = '../../experiments/sync/'+pop+'/TCR_spikes_3d/TCR_spikes_' + str(i_run) + str(j_run) + '.csv'
        print("Starting run {}-{}".format(i_run,j_run))
        score = synchrony_measure(PATH) # compute score for run
        print("Completed run {}-{} with score {}".format(i_run,j_run, score))
        scores.append(score)
        f.write('%f\n'% score)
'''

# plot score vs run plot
f = open("sync_scores.txt","r")
scores = np.loadtxt(f)
plt.plot(idc_list,scores[:10])
plt.plot(idc_list,scores[10:20])
plt.plot(idc_list,scores[20:30])
plt.legend(['g = {}'.format(g_list[0]),'g = {}'.format(g_list[1]),'g = {}'.format(g_list[2])])
plt.title(pop + ': synchrony measure plot')
plt.xlabel('idc poisson lambda')
plt.ylabel('Synchrony measure')
plt.show()


# Spike raster plots
for i_run,g in enumerate(g_list):
    fig, ax = plt.subplots(math.ceil(M/c),c,sharex=True,sharey=True)
    custom_xlim = (500,2000)
    plt.setp(ax, xlim=custom_xlim)
    fig.suptitle('Raster: {}, g = {}, idc = 5 to 14'.format(pop,g), fontsize=16)

    for j_run in range(M):
        PATH = '../../experiments/sync/'+pop+'/TCR_spikes_3d/TCR_spikes_' + str(i_run) + str(j_run) + '.csv'
        data = np.genfromtxt(PATH,delimiter=',') # load csv into numpy array
        i = int(j_run / c)
        j = int(j_run % c)  
        print(i,j,i_run,j_run,M,N)
        ax[i,j].scatter(data[:,1],data[:,0],s=1) # plot raster for run

plt.show()
