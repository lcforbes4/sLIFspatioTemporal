# This creates the data for figure 3: ramping the parameter of both the mean field and the averaged spiking network as
# it passes through the bifurcation in both directions in order to show the region of stability

#temporal bistab

import numpy as np
from matplotlib import pyplot as plt
from functions_and_sims.mean_field_equation_functions import ramping_mean_field, plot_ramped_mean_field_vs_param, MFsimulation_2_unif_pulses
from functions_and_sims.matlab_import_functions import get_periodic_branch_points
from matplotlib.gridspec import GridSpec
from functions_and_sims.spiking_network_functions import plot_pop, plot_avg, generate_spatial_connectivity_mat, ramp_param_sim, synch_2pulses_sim

def plot_bif_curve_J_vs_v(E, file_name):
    linewidth = 1
    p_sol_color='black'
    equil_sol_color = 'black'
    param_string = 'J_value'

    val_list = get_periodic_branch_points(file_name, param_string, 0, 1)

    # Get J value of Hopf Point
    Hopf_J_val = val_list[0][param_string]
    J_min = np.min(val_list[:][param_string])
    J_max = 4  # np.max(val_list[:]['E_value'])
    J_unst = np.linspace(J_min, Hopf_J_val)
    J_stab = np.linspace(Hopf_J_val, J_max)

    plt.plot(J_unst, (J_unst + np.sqrt(J_unst ** 2 + 4 * (E - J_unst))) / 2, color=equil_sol_color, linestyle='--', linewidth=linewidth)  # equil unstable
    plt.plot(J_stab, (J_stab + np.sqrt(J_stab ** 2 + 4 * (E - J_stab))) / 2, color=equil_sol_color, linestyle='-', linewidth=linewidth)  # equil stable

    # Get the index where the periodic solutions branch becomes stable
    j = 1
    while val_list[j]['Stability_color'] == 'b*':
        j += 1


    plt.plot(val_list[:j]['J_value'], val_list[:j]['Max'], color=p_sol_color, linestyle='--', linewidth=linewidth)  # max unstable
    plt.plot(val_list[j:]['J_value'], val_list[j:]['Max'], color=p_sol_color, linestyle='-', linewidth=linewidth)  # max stable
    plt.plot(val_list[:j]['J_value'], val_list[:j]['Min'], color=p_sol_color, linestyle='--', linewidth=linewidth)  # min unstable
    plt.plot(val_list[j:]['J_value'], val_list[j:]['Min'], color=p_sol_color, linestyle='-', linewidth=linewidth)  # min stable

    return

def MF_bistable_pulse_data_save():
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')  # Create two subplots
    t_max = 100
    Nx = 500
    D = 2
    J0 = -3.04
    J_1 = 0
    E = 3
    dt = 0.01
    v0 = (J0 + np.sqrt(J0 ** 2 + 4 * (E - J0))) / 2
    x = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    m_history = np.zeros((int(D / dt), Nx))  # The history
    center = v0
    for tt in range(0, int(D / dt)):
        m_history[tt, :] = center
    m_initial = center

    m = MFsimulation_2_unif_pulses(m_initial, m_history, t_max, Nx, D, dt, J0, J_1, E)
    # Plot the image of the simulation in the first subplot
    axs[0].imshow(m.T, aspect='auto', cmap='viridis', origin='lower', interpolation='none',
                  extent=[0, len(m) * .01, -np.pi, np.pi])
    axs[0].set_ylabel('Space (x)')
    axs[0].set_xlabel('Time (t)')
    axs[0].set_xlim([0, t_max])

    # Plot the amplitude at x=0 over time in the second subplot
    x_0_index = Nx // 2  # The index corresponding to x=0
    amplitude_at_0 = m[:, x_0_index]  # Extract the amplitude at x=0 for each time step

    axs[1].plot(np.arange(0, t_max, dt), amplitude_at_0, color='tab:blue')
    axs[1].set_ylabel('Amplitude at x=0')
    axs[1].set_xlabel('Time (t)')
    axs[1].set_title('Amplitude at x=0 over Time')

    plt.tight_layout()
    plt.show()
    return

def spiking_bistable_pulse_data_save():
    save = 0

    # Parameters
    J_choice = 1  # 1- delta, 2-delayed exp, 3-alpha function
    J = -3.22
    E = 3
    tau = 0
    delay = 2
    N = 1000  # cant use saved matrix if N neq 1000

    tstop = 100
    pulse_time = 25
    pulse_amp = 1
    duration = 2
    ramp_time = 1

    # Graphing/Euler Options
    dt = .001

    v_ax = [0, 2]
    v_dot_ax = [-1.5, 1.5]

    N_pulse_time = int(pulse_time / dt)
    N_duration = int(duration / dt)

    # Calculate Equilibrium
    v0 = J + np.sqrt(J ** 2 + 4 * (E - J))
    v0 /= 2

    initial2 = v0  # 1.1 #v0+.01

    # vmean2, v0 = calculate_mean_field(J_choice, E, J, delay, tstop, dt, initial2, tau)
    connection_prob = 0.5
    g_bar = np.load('../current_fig_scripts/figHopf_bistab_data/g_bar_bistab_temp.npy')
    # g_bar =  generate_connectivity_mat(J, connection_prob, N)
    vpop2, spktimes2, conduc2, ramp_vals = synch_2pulses_sim(1, tstop, dt, N, J, tau, E, delay, pulse_time, duration,
                                                             pulse_amp, initial2, g_bar)

    if save ==1:
        np.save(
            '../current_fig_scripts/figHopf_bistab_data/single_param_bistab_data/spikingpopvoltages_J-3.22_E3_D2_N1000_T100_dt.001.npy',
            vpop2)
        np.save(
            '../current_fig_scripts/figHopf_bistab_data/single_param_bistab_data/spiktimes_J-3.22_E3_D2_N1000_T100_dt.001.npy',
            spktimes2)
        np.save('../current_fig_scripts/figHopf_bistab_data/single_param_bistab_data/E_pulses_T100_dt_.001.npy', ramp_vals)
        # np.save('bistableoscspiketimes.npy',spktimes2)

    fig = plt.figure(layout='constrained')
    gs = GridSpec(3, 1, figure=fig)

    ax4 = fig.add_subplot(gs[0, :])
    plot_pop(spktimes2, dt)
    plt.xlim([0, tstop])

    ax5 = fig.add_subplot(gs[1, :])
    # plot_single_neuron(tstop, dt, vpop2, N, spktimes2, J, v0, 0, delay, 'k')
    # plot_mean_field(vmean2, tstop, dt, v0)
    plot_avg(tstop, dt, vpop2)  # t_max, dt, v, N, J, color1='k'
    plt.xlim([0, tstop])

    ax6 = fig.add_subplot(gs[2, :])
    plt.plot(range(0, int(tstop / dt)), ramp_vals[:-1])
    plt.xlim([0, int(tstop / dt)])

    plt.show()

def MF_and_spk_ramp_data_save():
    save = 1

    # Parameters
    J_choice = 1  # 1- delta, 2-delayed exp, 3-alpha function
    ramping_param_ind = 0  # J=0, E=1, delay??
    J = -4
    E = 3
    tau = 0
    delay = 2
    N = 100
    tstop = 100
    dt = .01
    v0 = J + np.sqrt(J ** 2 + 4 * (E - J))
    v0 /= 2
    initial = v0  # v0+.01
    connection_prob = 0.5
    g_bar = generate_spatial_connectivity_mat(J, 0, N, p=connection_prob)

    # spiking sim ramp forward:
    param_start3 = -3.5
    param_end3 = -2.5
    vpop2, spktimes2 = ramp_param_sim(J_choice, tstop, dt, N, J, tau, E, delay, initial, g_bar,  ramping_param_ind, param_start3, param_end3)
    print('spiking forward complete')
    #vpop2 = np.load('../current_fig_scripts/figHopf_bistab_data/fwdbwdv1/spiking_J-3.5to-2.5_N2000_dt001.npy')


    # spiking sim ramp backward:
    param_start4 = -2.5
    param_end4 = -3.5
    vpop3, spktimes3 = ramp_param_sim(J_choice, tstop, dt, N, J, tau, E, delay, initial, g_bar,  ramping_param_ind, param_start4, param_end4)
    #vpop3 = np.load('../current_fig_scripts/figHopf_bistab_data/fwdbwdv1/spiking_J-2.5to-3.5_N2000_dt001.npy')
    print('spiking backward complete')

    # mean field ramp forward:
    param_start1 = -3.08
    param_end1 = -3
    vmean1 = ramping_mean_field(J_choice, E, J, delay, tstop, dt, 1.7, tau, ramping_param_ind, param_start1, param_end1)
    # vmean1 = np.load('deltaCaseFigures/figHopf_bistab_data/fwdbwdv1/mfield_J-3.1to-3_dt001.npy')
    print('mean field forward complete')

    # mean field ramp backward:
    param_start2 = -3.04
    param_end2 = -3.08
    v0 = J + np.sqrt(param_start2 ** 2 + 4 * (E - param_start2))
    vmean2 = ramping_mean_field(J_choice, E, J, delay, tstop, dt, 1.4, tau, ramping_param_ind, param_start2, param_end2)
    # vmean2 = np.load('deltaCaseFigures/figHopf_bistab_data/fwdbwdv1/mfield_J-3.04to-3.1_dt001.npy')
    print('mean field backward complete')


    if ramping_param_ind ==0:
        ramp_param = 'E'
    if ramping_param_ind ==1:
        ramp_param ='J'
    if save ==1:
        np.save(f'../current_fig_scripts/figHopf_bistab_data/fwdbwdv1/spktimesfwd_N:{N}_dt:{dt}_J:{J}_E:{E}_D:{delay}_rampingIn{ramp_param}from{param_start3}to{param_end3}.npy',vpop2)
        np.save(f'../current_fig_scripts/figHopf_bistab_data/fwdbwdv1/spktimesbwd_N:{N}_dt:{dt}_J:{J}_E:{E}_D:{delay}_rampingIn{ramp_param}from{param_start4}to{param_end4}.npy', vpop3)
        np.save(f'../current_fig_scripts/figHopf_bistab_data/fwdbwdv1/mfieldFwd_N:{N}_dt:{dt}_J:{J}_E:{E}_D:{delay}_rampingIn{ramp_param}from{param_start1}to{param_end1}.npy', vmean1)
        np.save(f'../current_fig_scripts/figHopf_bistab_data/fwdbwdv1/mfieldBwd_N:{N}_dt:{dt}_J:{J}_E:{E}_D:{delay}_rampingIn{ramp_param}from{param_start2}to{param_end2}.npy', vmean2)

    fig = plt.figure(layout='constrained')
    gs = GridSpec(2, 2, figure=fig)

    ax5 = fig.add_subplot(gs[0, 1])
    plot_ramped_mean_field_vs_param(vmean1, tstop, dt, param_start1, param_end1, color='grey')
    plot_bif_curve_J_vs_v(E, '../current_fig_scripts/figHopf_data/matlab_data/branch4_Jsave.mat')
    plt.xlim([param_start1, param_end1])
    plt.ylim([0.8, 1.75])

    ax6 = fig.add_subplot(gs[1, 1])
    plot_ramped_mean_field_vs_param(vmean2, tstop, dt, param_start2, param_end2, color='grey')
    plot_bif_curve_J_vs_v(E, '../current_fig_scripts/figHopf_data/matlab_data/branch4_Jsave.mat')
    plt.xlim([param_end2, param_end1])
    plt.ylim([0.8, 1.75])

    ax7 = fig.add_subplot(gs[0, 0])
    plot_ramped_mean_field_vs_param(vpop2, tstop, dt, param_start3, param_end3, color='grey')
    #plot_bif_curve_J_vs_v(E, '../current_fig_scripts/figHopf_data/matlab_data/branch4_Jsave.mat')
    plt.xlim([param_start3, param_end3])
    plt.ylim([0.5, 1.75])

    ax8 = fig.add_subplot(gs[1, 0])
    plot_ramped_mean_field_vs_param(vpop3, tstop, dt, param_start4, param_end4, color='grey')
    #plot_bif_curve_J_vs_v(E, '../current_fig_scripts/figHopf_data/matlab_data/branch4_Jsave.mat')
    plt.xlim([param_end4, param_start4])
    plt.ylim([0.5, 1.75])

    # fig = plt.figure(layout='constrained')
    # plot_bif_curve(E, 'deltaCaseFigures/figHopf_data/matlab_data/branch4_Jsave.mat')

    plt.show()


    return

if __name__ == '__main__':
    #MF_and_spk_ramp_data_save()
    #spiking_bistable_pulse_data_save()
    MF_bistable_pulse_data_save()

