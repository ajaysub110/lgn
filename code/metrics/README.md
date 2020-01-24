# Spike ISI Visualizer
## Task:
Build a histogram style neuron level inter-spike interval visualizer

## Input:
Separate CSV file for each population with 2 columns; neuron index and spike time

## Algorithm:
1. Read CSV file into list of n(neurons in pop) lists each with corresponding spike times
2. replace each col with the diff between it and previous col
3. barplot with error bars these differences for each row separately

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
