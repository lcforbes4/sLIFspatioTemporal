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

num_of_points = 5
tstop = 500 #100 #500
dt = 0.01
N = 1000
J_choice = 1 #delta fucntion

# Make sure these param values match which BIF-TOOL data you're comparing against
param_name_to_vary = 'J0' #'J0','E', or 'delay'
param_min = -5 #J=-5 #E=1.05 #delay=.9
param_max = -3.25 #J=-3.25  #E=4.1 #delay=2
delay = 2
g = -4
tau = 1
E = 3
initial = 1.6
save = 0


num_oscillations = 10 # the number of oscillations at the end of simulation to average over for period,min-max,etc
param_list = np.linspace(param_min, param_max, num_of_points)
per = np.zeros(len(param_list))
average_min_amplitude = np.zeros(len(param_list))
average_max_amplitude= np.zeros(len(param_list))
average_l2_norm = np.zeros(len(param_list))

if param_name_to_vary == 'J0': #varying J0
    g_bar = generate_connectivity_mat(param_list[0],0.5,N)
else:
    g_bar = generate_connectivity_mat(g, 0.5, N)


for i in range(0,len(param_list)):
    #simulate
    if param_name_to_vary == 'J0':
        if i > 0:
            g_bar = (g_bar / param_list[i - 1]) * param_list[i]
        v_avg, spktimes, J = neuron_population_output_avg(J_choice, tstop, dt, N, param_list[i], tau, E, delay, initial,
                                                          g_bar)
    elif param_name_to_vary =='E':
        v_avg, spktimes, J = neuron_population_output_avg(J_choice, tstop, dt, N, g, tau, param_list[i], delay, initial,
                                                          g_bar)
    elif param_name_to_vary =='delay':
        v_avg, spktimes, J = neuron_population_output_avg(J_choice, tstop, dt, N, g, tau, E, param_list[i], initial,
                                                          g_bar)


    v_avg, spktimes, J = neuron_population_output_avg(J_choice, tstop, dt, N, param_list[i], tau, E, delay, initial, g_bar)
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(0, tstop, dt),v_avg)

    # Compute Power Spectrum
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
    last_oscillation_start = len(v_avg) - (oscillation_period * (num_oscillations+1)) #index at which your group of osc at end start, add 1 to ignore last oscillation data for cleaner data
    last_oscillation_end = len(v_avg) - oscillation_period # index of when penultimate oscillation ends

    # Initialize lists for storing amplitude max/min and L2 norms
    min_amplitudes = []
    max_amplitudes = []
    l2_norms = []

    # Loop over the last oscillations
    for j in range(num_oscillations):
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
plt.subplot(1, 3, 1)
plt.plot(param_list, per, marker='o', label='Period')
plt.xlabel('Parameter')
plt.ylabel('Period')
plt.title('Period vs Parameter')
plt.grid()
plt.legend()

# Second plot: average min/max amplitude vs param_list
plt.subplot(1, 3, 2)
plt.plot(param_list, average_min_amplitude, marker='o', label='Average Min Amplitude', color='blue')
plt.plot(param_list, average_max_amplitude, marker='o', label='Average Max Amplitude', color='red')
plt.xlabel('Parameter')
plt.ylabel('Amplitude')
plt.title('Average Min/Max Amplitude vs Parameter')
plt.grid()
plt.legend()

# Third plot: L2 norm
plt.subplot(1, 3, 3)
plt.plot(param_list, average_l2_norm, marker='o', label='Average Min Amplitude', color='blue')
plt.xlabel('Parameter')
plt.ylabel('Amplitude')
plt.title('Average Min/Max Amplitude vs Parameter')
plt.grid()


if save==1:
    np.save(f'../current_fig_scripts/figHopf_data/saves_sim_data/delta_varied_{param_name_to_vary}.npy', param_list)
    np.save(f'../current_fig_scripts/figHopf_data/saves_sim_data/delta_period_varied_{param_name_to_vary}.npy', per)
    np.save(f'../current_fig_scripts/figHopf_data/saves_sim_data/delta_min_amp_varied_{param_name_to_vary}.npy', average_min_amplitude)
    np.save(f'../current_fig_scripts/figHopf_data/saves_sim_data/delta_max_amp_varied_{param_name_to_vary}.npy', average_max_amplitude)
    np.save(f'../current_fig_scripts/figHopf_data/saves_sim_data/delta_l2_norm_varied_{param_name_to_vary}.npy', average_l2_norm)

# Show plots
plt.tight_layout()
plt.show()
