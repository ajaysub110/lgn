# !/usr/bin/python

import numpy as np
import os

import matplotlib.pylab as plt
import pylab
from pylab import *
from pyNN.random import NumpyRNG, RandomDistribution

import spynnaker8 as p
import pyNN.utility.plotting as plot
from pyNN.utility.plotting import Figure, Panel


import time
start_time = time.time()

class lgn_microcol():

    def __init__(self, p, scale_fact, DCbase):
        self.NumCellsTCR = int(8*scale_fact)
        self.NumCellsIN = int(2*scale_fact)
        self.NumCellsTRN = int(4*scale_fact)

        ''' Initialising Model connectivity parameters'''
        self.intra_pop_delay = 1
        self.intra_nucleus_delay = 2
        self.inter_node_delay = 3

        ''' THE STRENGTH OF CONNECTION TO THE
        RELAY CELLS ARE KNOWN TO BE VERY STRONG. HENCE THEY ARE SET TO 5.0. ON THE OTHER
        HAND THE CONNECTION TO INTERNEURONS ARE KNOWN TO BE FAR AWAY FROM THE SOMA. THEREFORE
        THEY ARE SET TO 1.0. THOSE FOR THE TRN ARE SET TO 2.0. THESE ARE OFCOURSE ARBITRARY
        FIGURES.'''
        self.p_in2tcr = 0.15 ##0.3 ## change to test effect of IN
        self.p_in2in = 0.236


        self.w_in2tcr= 2.0 ## 4.0##change to test effect of IN
        self.w_in2in = 2.0

        self.w_tcr2trn_internode = 2.0
        self.p_tcr2trn_internode = 0.35
        self.w_trn2trn_internode = 2.0
        self.p_trn2trn_internode = 0.2
        self.w_trn2tcr_internode = 2.0
        self.p_trn2tcr_internode = 0.15


        ''' Initialising Izhikevich spiking neuron model parameters.
        We have used the current-based model here.'''

        # Tonic mode parameters
        self.tcr_a_tonic = 0.02
        self.tcr_b_tonic = 0.2
        self.tcr_c_tonic = -65.0
        self.tcr_d_tonic = 6.0
        self.tcr_v_init_tonic = -65.0

        self.in_a_tonic = 0.1
        self.in_b_tonic = 0.2
        self.in_c_tonic = -65.0
        self.in_d_tonic = 6.0
        self.in_v_init_tonic = -70.0

        self.trn_a_tonic = 0.02
        self.trn_b_tonic = 0.2
        self.trn_c_tonic = -65.0
        self.trn_d_tonic = 6.0
        self.trn_v_init_tonic = -75.0

        self.tcr_u_init_tonic = self.tcr_b_tonic * self.tcr_v_init_tonic
        self.in_u_init_tonic = self.in_b_tonic * self.in_v_init_tonic
        self.trn_u_init_tonic = self.trn_b_tonic * self.trn_v_init_tonic

        self.current_Pulse = DCbase ##a constant dc bias current;
        self.tau_ex = 1.7 ###6 ##  excitatory input time constant
        self.tau_inh = 2.5 ###4 ## inhibitory input time constant

        '''Defining each cell type as dictionary'''

        # THALAMOCORTICAL RELAY CELLS (TCR)

        self.TCR_cell_params = {'a': self.tcr_a_tonic, 'b': self.tcr_b_tonic, 'c': self.tcr_c_tonic,
                                'd': self.tcr_d_tonic, 'v': self.tcr_v_init_tonic, 'u': self.tcr_u_init_tonic, 								'tau_syn_E': self.tau_ex, 'tau_syn_I': self.tau_inh,
                                'i_offset': self.current_Pulse/10.
                                }

        # THALAMIC INTERNEURONS (IN)

        self.IN_cell_params = {'a': self.in_a_tonic, 'b': self.in_b_tonic, 'c': self.in_c_tonic,
                                'd': self.in_d_tonic, 'v': self.in_v_init_tonic, 'u': self.in_u_init_tonic, 								'tau_syn_E': self.tau_ex, 'tau_syn_I': self.tau_inh,
                                'i_offset': self.current_Pulse/10.
                                }

        # THALAMIC RETICULAR NUCLEUS (TRN)

        self.TRN_cell_params = {'a': self.trn_a_tonic, 'b': self.trn_b_tonic, 'c': self.trn_c_tonic,
                                'd': self.trn_d_tonic, 'v': self.trn_v_init_tonic, 'u': self.trn_u_init_tonic,
                                'tau_syn_E': self.tau_ex, 'tau_syn_I': self.tau_inh,
                                'i_offset': self.current_Pulse/10.
                           }

        '''Creating populations of each cell type'''
        self.TCR_pop = p.Population(self.NumCellsTCR, p.Izhikevich(**self.TCR_cell_params), label='TCR_pop')
        self.IN_pop = p.Population(self.NumCellsIN, p.Izhikevich(**self.IN_cell_params), label='IN_pop')
        self.TRN_pop = p.Population(self.NumCellsTRN, p.Izhikevich(**self.TRN_cell_params), label='TRN_pop')

        '''IN2TCR'''
        self.Proj4 = p.Projection(self.IN_pop, self.TCR_pop,
                                  p.FixedProbabilityConnector(p_connect=self.p_in2tcr),
                                  synapse_type=p.StaticSynapse(weight=self.w_in2tcr, delay=self.intra_nucleus_delay), receptor_type='inhibitory')


        '''IN2IN'''
        self.Proj5  = p.Projection(self.IN_pop, self.IN_pop,
                                   p.FixedProbabilityConnector(p_connect=self.p_in2in),
                                   synapse_type=p.StaticSynapse(weight=self.w_in2in, delay=self.intra_pop_delay), receptor_type='inhibitory')

        '''TCR2TRN'''
        self.Proj6 = p.Projection(self.TCR_pop, self.TRN_pop,
                     p.FixedProbabilityConnector(p_connect=self.p_tcr2trn_internode),
                     p.StaticSynapse(weight=self.w_tcr2trn_internode, delay=self.inter_node_delay),
                     receptor_type='excitatory')

        '''TRN2TCR'''
        self.Proj7 = p.Projection(self.TRN_pop, self.TCR_pop,
                     p.FixedProbabilityConnector(p_connect=self.p_trn2tcr_internode),
                     p.StaticSynapse(weight=self.w_trn2tcr_internode, delay=self.inter_node_delay),
                     receptor_type='inhibitory')

        '''TRN2TRN'''

        self.Proj8 = p.Projection(self.TRN_pop, self.TRN_pop,
                     p.FixedProbabilityConnector(p_connect=self.p_trn2trn_internode),
                     p.StaticSynapse(weight=self.w_trn2trn_internode, delay=self.inter_node_delay),
                     receptor_type='inhibitory')

    def recordSpikes(self):

        self.TCR_pop.record(['spikes','v'])
        self.IN_pop.record(['spikes','v'])
        self.TRN_pop.record(['spikes','v'])

def getDisplaySpikes(lgn_module,v):
    print('within getdisplayspikes function')

    TCR_spikes = lgn_module.TCR_pop.spinnaker_get_data("spikes")
    IN_spikes = lgn_module.IN_pop.spinnaker_get_data("spikes")
    TRN_spikes = lgn_module.TRN_pop.spinnaker_get_data("spikes")

    plt.figure(2)
    plt.subplot(3,1,1)
    plt.plot(TCR_spikes[:, 1], TCR_spikes[:, 0], '.r',markersize=1)
    plt.subplot(3,1,2)
    plt.plot(IN_spikes[:, 1], IN_spikes[:, 0], '.b',markersize=1)
    
    plt.subplot(3,1,3)
    plt.plot(TRN_spikes[:, 1], TRN_spikes[:, 0], '.g',markersize=1)
    
    plt.show()

def getDisplayVoltages(lgn_module):
    print("within getdisplayvoltages function")
    
    TCR_voltages = lgn_module.TCR_pop.spinnaker_get_data('v')
    TRN_voltages = lgn_module.TRN_pop.spinnaker_get_data('v')
    IN_voltages = lgn_module.IN_pop.spinnaker_get_data('v')
    
    plt.figure(3)
    plt.subplot(3,1,1)
    plt.plot(TCR_voltages[:,1],TCR_voltages[:,2],'r',markersize=1)
    plt.subplot(3,1,2)
    plt.plot(TRN_voltages[:,1],TRN_voltages[:,2],'b',markersize=1)
    plt.subplot(3,1,3)
    plt.plot(IN_voltages[:,1],IN_voltages[:,2],'g',markersize=1)
    
    plt.show()
    

#     filenameTCR = './Data_Case3/TCRvoltages.csv'
#     np.savetxt(filenameTCR, TCR_voltages)

#     filenameTRN = './Data_Case3/TRNvoltages.csv'
#     np.savetxt(filenameTRN, TRN_voltages)

#     filenameIN = './Data_Case3/INvoltages.csv'
#     np.savetxt(filenameIN, IN_voltages)



if __name__ == "__main__":
    totalDuration=2000 ####2000 ##10000  total duration of simulation
    time_resol = 0.1 ## time-step of model equation solver
    TimeInt = 1/time_resol
    TotalDataPoints = totalDuration * TimeInt ##for a solution time-step of 0.1

    '''Pulse time lengths'''
    duration = 2000

    '''Define the population scale and the base DC value'''
    scale_fact = 10
    DCbase = 4

    '''DEFINE THE STRING SIZE'''
    stringSize = 5 ###5#1000

    '''SET SOME OF THE NOISE PARAMETERS'''
    NumPoissonInputs2TCR = int(8*scale_fact / 2)
    NumPoissonInputs2IN = int(2*scale_fact / 2)
    NumPoissonInputs2TRN = int(4*scale_fact / 2)
    p_noise2tcr = 0.07
    w_noise2tcr = 5.0
    p_noise2in = 0.47
    w_noise2in = 2.0
    

    inter_node_delay=3.0

    ''' SET UP SPINNAKER AND BEGIN SIMULATION'''

    p.setup(timestep=time_resol, min_delay=1, max_delay=14.0)
    #p.set_number_of_neurons_per_core(p.Izhikevich, 25)

    '''DEFINE THE NOISY INPUT COMMON TO ALL NODES'''
    ratePoissonInput_1 = 30
    ratePoissonInput_2 = 30
    ratePoissonInput_3 = 30
    startPoissonInput = p.RandomDistribution("uniform", [10, 50])  ###0[500, 700]
    durationPoissonInput =9950 ##1950  ##  ####TotalDuration

    noiseSource2TCR = p.Population(NumPoissonInputs2TCR, p.SpikeSourcePoisson,
                                        {'rate': ratePoissonInput_2,
                                         'duration': durationPoissonInput,
                                         'start': startPoissonInput},
                                        label='noiseSource2TCR')

    noiseSource2IN = p.Population(NumPoissonInputs2IN, p.SpikeSourcePoisson,
                                       {'rate': ratePoissonInput_1,
                                        'duration': durationPoissonInput,
                                        'start': startPoissonInput},
                                       label='noiseSource2IN')

    noiseSource2TRN = p.Population(NumPoissonInputs2TRN, p.SpikeSourcePoisson,
                                        {'rate': ratePoissonInput_3,
                                         'duration': durationPoissonInput,
                                         'start': startPoissonInput},
                                        label='noiseSource2TRN')

    '''FORM THE LGN string CONSISTING OF XXX NODES, WHERE EACH NODE IS AN LGN INSTANCE.
    EACH NODE RECEIVES PROJECTION FROM THE NOISE SOURCE DEFINED ABOVE, PARAMETERS OF CONNECTION
    ARE ALSO STATED ABOVE.'''
    lgn_string = []
    for i in range(0, stringSize):
        lgn_nodeX = lgn_microcol(p, scale_fact, DCbase)
        '''NOISE2TCR'''

        Proj1 = p.Projection(noiseSource2TCR, lgn_nodeX.TCR_pop,
                                  p.FixedProbabilityConnector(p_connect=p_noise2tcr),
                                  synapse_type=p.StaticSynapse(weight=w_noise2tcr, delay=inter_node_delay),
                                  receptor_type='excitatory')

        '''NOISE2IN'''
        Proj2 = p.Projection(noiseSource2IN, lgn_nodeX.IN_pop,
                                  p.FixedProbabilityConnector(p_connect=p_noise2in),
                                  synapse_type=p.StaticSynapse(weight=w_noise2in, delay=inter_node_delay),
                                  receptor_type='excitatory')

        # '''NOISE2TRN'''
        # Proj3 = p.Projection(noiseSource2TRN, lgn_nodeX.TRN_pop,
        #                             p.FixedProbabilityConnector(p_connect=p_noise2trn),
        #                             synapse_type=p.StaticSynapse(weight=w_noise2trn, delay=inter_node_delay),
        #                             receptor_type='excitatory')

        lgn_string.append(lgn_nodeX)
#         print('size of LGN string is %d' % size(lgn_string))
        print('length of LGN string is %d' % len(lgn_string))


    '''NEXT, WE CREATE THE STRING CONNECTIONS'''
    n1vect = []
    n0vect = []
    for numcon in range(0, stringSize):
        n0 = numcon
        if numcon == stringSize - 1:
            n1 = 0
        else:
            n1 = numcon + 1

        lgn_nodeN = lgn_string[n0]
        lgn_nodeN1 = lgn_string[n1]

        n1vect.append(n1)
        n0vect.append(n0)

        '''TRN TO TRN LATERAL INTER-NODE CONNECTION'''
        p.Projection(lgn_nodeN.TRN_pop, lgn_nodeN1.TRN_pop,
                     p.FixedProbabilityConnector(p_connect=lgn_nodeN.p_trn2trn_internode),
                     p.StaticSynapse(weight=lgn_nodeN.w_trn2trn_internode, delay=lgn_nodeN.inter_node_delay),
                     receptor_type='inhibitory')


        p.Projection(lgn_nodeN1.TRN_pop, lgn_nodeN.TRN_pop,
                     p.FixedProbabilityConnector(p_connect=lgn_nodeN.p_trn2trn_internode),
                     p.StaticSynapse(weight=lgn_nodeN.w_trn2trn_internode, delay=lgn_nodeN.inter_node_delay),
                     receptor_type='inhibitory')

        '''TCR TO TRN LATERAL INTER-NODE CONNECTION'''
        p.Projection(lgn_nodeN.TCR_pop, lgn_nodeN1.TRN_pop,
                    p.FixedProbabilityConnector(p_connect=lgn_nodeN.p_tcr2trn_internode),
                    p.StaticSynapse(weight=lgn_nodeN.w_tcr2trn_internode, delay=lgn_nodeN.inter_node_delay),
                    receptor_type='excitatory')

        p.Projection(lgn_nodeN1.TCR_pop, lgn_nodeN.TRN_pop,
                     p.FixedProbabilityConnector(p_connect=lgn_nodeN.p_tcr2trn_internode),
                     p.StaticSynapse(weight=lgn_nodeN.w_tcr2trn_internode, delay=lgn_nodeN.inter_node_delay),
                     receptor_type='excitatory')

        '''TRN TO TCR LATERAL INTER-NODE CONNECTION'''

        p.Projection(lgn_nodeN.TRN_pop, lgn_nodeN1.TCR_pop,
                     p.FixedProbabilityConnector(p_connect=lgn_nodeN.p_trn2tcr_internode),
                     p.StaticSynapse(weight=lgn_nodeN.w_trn2tcr_internode, delay=lgn_nodeN.inter_node_delay),
                     receptor_type='inhibitory')


        p.Projection(lgn_nodeN1.TRN_pop, lgn_nodeN.TCR_pop,
                     p.FixedProbabilityConnector(p_connect=lgn_nodeN.p_trn2tcr_internode),
                     p.StaticSynapse(weight=lgn_nodeN.w_trn2tcr_internode, delay=lgn_nodeN.inter_node_delay),
                     receptor_type='inhibitory')

    for j in range(len(lgn_string)):
        lgn_nodeY = lgn_string[j]
        lgn_nodeY.recordSpikes()

    noiseSource2TCR.record(['spikes'])
    noiseSource2IN.record(['spikes'])
    noiseSource2TRN.record(['spikes'])

    leftspread= -3 ###-100
    rightspread = 2 ###100
    if stringSize >= 5:
        for k in range(leftspread, rightspread):
            centralNode = int(ceil(len(lgn_string)/2))+k
            lgn_nodeZ = lgn_string[centralNode]
            lgn_nodeZ.TCR_pop.set(i_offset=DCbase)
            lgn_nodeZ.IN_pop.set(i_offset=DCbase)
    else:
        lgn_nodeZ = lgn_string[1]
        lgn_nodeZ.TCR_pop.set(i_offset=DCbase)
        lgn_nodeZ.IN_pop.set(i_offset=DCbase)
    
    # manual run code
    i = 3

    p.run(duration)

    noiseSource2TCR_spikes = noiseSource2TCR.spinnaker_get_data("spikes")
    noiseSource2IN_spikes = noiseSource2IN.spinnaker_get_data("spikes")
    noiseSource2TRN_spikes = noiseSource2TRN.spinnaker_get_data("spikes")

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(noiseSource2TCR_spikes[:, 1], noiseSource2TCR_spikes[:, 0], '.r',markersize=1)
    plt.subplot(3, 1, 2)
    plt.plot(noiseSource2IN_spikes[:, 1], noiseSource2IN_spikes[:, 0], '.b',markersize=1)
    plt.subplot(3, 1, 3)
    plt.plot(noiseSource2TRN_spikes[:, 1], noiseSource2TRN_spikes[:, 0], '.g',markersize=1)
    plt.show()
        
    filenameTCRnoise = './Data_Case3/noiseSource2TCR_spikeraster.csv'
    np.savetxt(filenameTCRnoise, noiseSource2TCR_spikes)

    filenameINnoise = './Data_Case3/noiseSource2IN_spikeraster.csv'
    np.savetxt(filenameINnoise, noiseSource2IN_spikes)

    filenameTRNnoise = './Data_Case3/noiseSource2TRN_spikeraster.csv'
    np.savetxt(filenameTRNnoise, noiseSource2TRN_spikes)

    filename_n1 = './Data_Case3/n1vect.csv'
    np.savetxt(filename_n1, n1vect)

    filename_n0 = './Data_Case3/n0vect.csv'
    np.savetxt(filename_n0, n0vect)

    for w in range(len(lgn_string)):
        lgn_nodeW = lgn_string[w]
        getDisplaySpikes(lgn_nodeW, w)
        print ("getting the %d th spikes" %w)
        
    lgn_nodeW = lgn_string[0]
    getDisplayVoltages(lgn_nodeW)
    print ("getting the %d th voltages" %w)

    p.end()
    print("--- %s SECONDS ELAPSED ---\n \n \n" % (time.time() - start_time))