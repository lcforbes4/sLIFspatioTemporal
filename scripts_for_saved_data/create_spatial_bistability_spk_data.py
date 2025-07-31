import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from functions_and_sims.spiking_network_functions import plot_pop, synch_2pulses_spatial_sim, generate_spatial_connectivity_mat

# script to create data for spatial bistable

if __name__ == '__main__':
    #Parameters
    J_choice = 1 # 1- delta, 2-delayed exp, 3-alpha function
    J = 3
    J1 = 10
    E = .75
    tau = 0
    delay = 0
    N = 1000 # cant use saved matrix if N neq 1000
    dt = .01
    tstop = 40
    save = 0

    # The first pulse:
    pulse_time = 10
    pulse_amp = -2
    duration = 1

    # The second pulse:
    which_portion = 1 #0- middle third of net, 1-outer two thirds net
    pulse2_time = 30
    pulse2_amp = -2
    duration2 = 1

    v_ax = [0, 2]
    v_dot_ax = [-1.5, 1.5]


    # Calculate Equilibrium
    v0 = J + np.sqrt(J ** 2 + 4 * (E - J))
    v0 /= 2
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    initial2 = v0*0 #+ 5 * np.cos(x)

    #vmean2, v0 = calculate_mean_field(J_choice, E, J, delay, tstop, dt, initial2, tau)
    connection_prob = 0.5
    g_bar = generate_spatial_connectivity_mat(J, J1, N)
    vpop2, spktimes2, conduc2, ramp_vals = synch_2pulses_spatial_sim(1, tstop, dt, N, J, tau, E, delay, pulse_time,
                                                                     duration, pulse_amp, pulse2_time,
                                                                     duration2, pulse2_amp, initial2, g_bar, which_portion)

    if save == 1:
        np.save('bistableoscspiketimes.npy',spktimes2)

    fig = plt.figure(layout='constrained')
    gs = GridSpec(3, 1, figure=fig)

    ax4 = fig.add_subplot(gs[0, :])
    plot_pop(spktimes2, dt)
    plt.xlim([0, tstop])

    ax5 = fig.add_subplot(gs[1, :])

    # Plot the heatmap using the averaged voltages
    cax = ax5.imshow(vpop2.T, aspect='auto', cmap='viridis', origin='lower', extent = [0, tstop, 0, N])
    ax5.set_xlabel("Time (ms)")
    ax5.set_ylabel("Neuron Index")

    # Plot voltages at a given time point
    time_index = int((tstop-1) / dt)  # Choose the time index corresponding to t = 30 ms
    voltages_at_time = vpop2[time_index,:]  # All neuron voltages at that time point

    ax6 = fig.add_subplot(gs[2, :])
    ax6.plot(np.arange(N), voltages_at_time, 'bo')
    ax6.set_xlabel("Neuron Index")
    ax6.set_ylabel("Voltage (mV)")

    # Add color bar to represent the voltage scale
    fig.colorbar(cax, ax=ax5, label="Voltage (mV)")

    plt.show()