import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
fname = 'TCR_spikes_2.csv'
df = pd.read_csv(fname)
spike_data = df.values
spike_data = sorted(spike_data[:,1])
bw = 5
psth_vals = []

for i in range(0,1000,bw):
    psth_vals.append(len([e for e in spike_data if e>=i and e<i+bw])/bw)

plt.figure(1)
plt.plot(list(range(0,1000,bw)),psth_vals)
plt.title('PSTH for ' + fname)
plt.xlabel('time in ms')
plt.ylabel('Spike rate')
plt.show()

df = pd.DataFrame(psth_vals)
df.to_csv('psth_'+fname,index=False,header=False)
print(df.head())
