import numpy as np
import matplotlib.pyplot as plt
from functions_and_sims.spiking_network_functions import neuron_population_output_avg, generate_connectivity_mat
from scipy import signal
import scipy


# This creates all the data for the dots on figure 2, the hopf bifurcation figure

def compute_mean_centered_l2_norm(signal, tvals):
    mean_centered_signal = signal - np.mean(signal)
    result = np.sqrt(scipy.integrate.simpson(mean_centered_signal ** 2,x=tvals)),  # L2 Norm
    return result

num_of_points =4#J=8, others=10
tstop = 100 #300
dt = 0.01
N = 1000

# Make sure these param values match which BIF-TOOL data you're comparing against
J_choice = 1
param_min = 1.2#J-5 #E1.05 #delay=.9
param_max = 1.6#J-3.25 #delay=2
# go change varied param down in loop
delay = 2
g = -3.24
tau = 0.5
E = 3
initial = 1.6

num_oscillations = 10 # the number of oscillations at the end of simulation to average over for period,min-max,etc
param_list = np.linspace(param_min, param_max, num_of_points)
per = np.zeros(len(param_list))
average_min_amplitude = np.zeros(len(param_list))
average_max_amplitude= np.zeros(len(param_list))
average_l2_norm = np.zeros(len(param_list))
num_oscillations += 1
g_bar = generate_connectivity_mat(g,0.5,N)

#plt.figure(figsize=(12, 5))
#plt.plot(np.linspace(0,tstop,int(tstop/dt)),np.sin(np.linspace(0,tstop,int(tstop/dt))))


for i in range(0,len(param_list)):
    initial = param_list[i] #!!!g not J

    # Simulate
    print('b4')
    v_avg, spktimes, J = neuron_population_output_avg(J_choice, tstop, dt, N, g, tau, E, delay, initial, g_bar)
    #v_avg = np.sin(np.linspace(0,tstop,int(tstop/dt)))
    print('after')
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(0, tstop, dt),v_avg)

    # Compute Power Spectrum
    #arr = int(len(v_avg)/8)
    #nperseg = 2 ** (arr.bit_length() - 1)
    #print(nperseg)
    nperseg = 2 ** (17)
    f, Pxx_den = signal.welch(np.transpose(v_avg), fs=1 / dt, nperseg=nperseg)
    #plt.figure(figsize=(12, 5))
    #plt.plot(f, Pxx_den)

    # Compute Period:
    ind = np.argmax(Pxx_den)  # Grab index of highest point
    freq = f[ind]  # Grab associated frequency

    if freq!=0:
        per[i] = 1/freq  # Compute Period from freq:
    else:
        per[i]=5

    # Calculate the time step and the number of oscillations
    oscillation_period = int(per[i] / dt)  # Estimate number of time points in one oscillation
    last_oscillation_start = len(v_avg) - (oscillation_period * num_oscillations) #index at which your group of osc at end start
    last_oscillation_end = len(v_avg) - oscillation_period # index of when penultimate oscillation ends

    # Initialize lists for storing amplitude max/min and L2 norms
    min_amplitudes = []
    max_amplitudes = []
    l2_norms = []

    # Loop over the last 5 oscillations
    for j in range(num_oscillations-1):
        start_idx = last_oscillation_start + j * oscillation_period
        end_idx = start_idx + oscillation_period
        if end_idx > len(v_avg):  # Check to prevent index overflow
            break

        oscillation_segment = v_avg[start_idx:end_idx]
        tvals = np.linspace(start_idx*dt, end_idx*dt, len(oscillation_segment))

        # Calculate min and max amplitude
        min_amplitudes.append(np.min(oscillation_segment))
        max_amplitudes.append(np.max(oscillation_segment))

        # Compute L2 norm for this oscillation
        l2_norms.append(compute_mean_centered_l2_norm(oscillation_segment, tvals))

    # Calculate average min and max amplitudes
    average_min_amplitude[i] = np.mean(min_amplitudes)
    average_max_amplitude[i] = np.mean(max_amplitudes)

    # Average L2 norm
    average_l2_norm[i] = np.mean(l2_norms)

    # Compute the power spectrum of the last 5 oscillations
    #last_5_oscillations = v_avg[last_oscillation_start:last_oscillation_end]
    #f_last, Pxx_den_last = signal.welch(np.transpose(last_5_oscillations), fs=1 / dt, nperseg=nperseg)
    #ind = np.argmax(Pxx_den_last)  # Grab index of highest point
    #freq = f_last[ind]  # Grab associated frequency
    #print(1 / freq)  # Compute Period from freq:

    # Output results
    print(f"Average Min Amplitude of last 5 oscillations: {average_min_amplitude[i]}")
    print(f"Average Max Amplitude of last 5 oscillations: {average_max_amplitude[i]}")
    print(f"Average L2 Norm of last 5 oscillations: {average_l2_norm[i]}")
    print(f"Average period of last 5 oscillations: {per[i]}")

# Plotting
plt.figure(figsize=(12, 5))

# First plot: per vs param_list
plt.subplot(1, 2, 1)
plt.plot(param_list, per, marker='o', label='Period')
plt.xlabel('Parameter (E)')
plt.ylabel('Period')
plt.title('Period vs Parameter')
plt.grid()
plt.legend()

# Second plot: average min/max amplitude vs param_list
plt.subplot(1, 2, 2)
plt.plot(param_list, average_min_amplitude, marker='o', label='Average Min Amplitude', color='blue')
plt.plot(param_list, average_max_amplitude, marker='o', label='Average Max Amplitude', color='red')
plt.xlabel('Parameter (E)')
plt.ylabel('Amplitude')
plt.title('Average Min/Max Amplitude vs Parameter')
plt.grid()
plt.legend()


#save
#np.save('alphaFuncCaseFigures/fig2data/alpha_func_varied_delay.npy', param_list)
#np.save('alphaFuncCaseFigures/fig2data/alpha_func_period_varied_delay.npy', per)
#np.save('alphaFuncCaseFigures/fig2data/alpha_func_min_amp_varied_delay.npy', average_min_amplitude)
#np.save('alphaFuncCaseFigures/fig2data/alpha_func_max_amp_varied_delay.npy', average_max_amplitude)
#np.save('alphaFuncCaseFigures/fig2data/alpha_func_l2_norm_varied_delay.npy', average_l2_norm)

# Show plots
plt.tight_layout()
plt.show()
