# import libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import bisect

def isi_viz(csv_path):
    """
    # Spike ISI Visualizer
    ## Task:
    Build a histogram style neuron level inter-spike interval visualizer

    ## Input:
    Separate CSV file for each population with 2 columns; neuron index and spike time

    ## Algorithm:
    1. Read CSV file into list of n(neurons in pop) lists each with corresponding spike times
    2. replace each col with the diff between it and previous col
    3. barplot with error bars these differences for each row separately
    """
    data = np.genfromtxt(csv_path,delimiter=',')
    sp_times = [[]]
    n_neurons = int(data[-1,0])
    cur_i = 0
    for i in range(len(data)):
        if int(data[i,0]) != cur_i:
            sp_times.append([])
            cur_i = int(data[i,0])
        if len(sp_times[cur_i]) > 0:
            sp_times[cur_i].append(data[i,1] - data[i-1,1])
        else:
            sp_times[cur_i].append(data[i,1])

    stats = np.empty((len(sp_times),4))
    for i in range(len(sp_times)):
        min = np.min(sp_times[i][1:])
        max = np.max(sp_times[i][1:])
        mean = np.mean(sp_times[i][1:])
        std = np.std(sp_times[i][1:])
        stats[i,:] = np.array([min,max,mean,std])
    stats = stats.T

    plt.errorbar(np.arange(n_neurons+1),stats[2],stats[3],fmt='ok',lw=3)
    plt.errorbar(np.arange(n_neurons+1),stats[2],[stats[2]-stats[0],stats[1]-stats[2]],fmt='.k',ecolor='gray',lw=1)
    plt.xlabel('Neuron Index')
    plt.ylabel('Inter-Spike Interval(ISI)')
    plt.title('ISI Barplot for ' + csv_path)
    plt.show()

def find_le(a,x):
    # Find rightmost value less than or equal to x
    i = bisect.bisect_right(a, x)
    if i:
        return i
    return 0

def compute_phi_t(t,sp_times_i,isp_low,T):
    # computing instantaneous phase (Eqn. 6 from paper)
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
    n_neurons = int(data[-1,0])
    cur_i = 0
    for i in range(len(data)):
        if int(data[i,0]) != cur_i:
            sp_times.append([])
            cur_i = int(data[i,0])
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
        print(t,S_t)
        S.append(S_t)
    return (2/(n_neurons*(n_neurons-1))) * np.mean(S)

if __name__ == '__main__':
    isi_viz('TRN_spikes.csv')
    print(synchrony_measure('TRN_spikes.csv'))
