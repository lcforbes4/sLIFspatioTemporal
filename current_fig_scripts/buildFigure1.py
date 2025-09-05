import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from functions_and_sims.spiking_network_functions import plot_single_neuron, plot_spike_times
from functions_and_sims.spiking_network_functions import plot_pop, plot_avg, neuron_population
import seaborn as sns
from functions_and_sims.visualization_functions import plot_network_graph_diagram, add_subplot_label

fontsize = 8
labelsize = 6
letterlabelsize =10

def two_neuron_delay_visualization_save():
    # run a 2 neuron spiking simulation and save

    J_choice = 0 # 0 for delta function, 1 for exponential, and 2 for alpha function

    t_max2 = 1.5
    g_2pop = -1
    dt = .001
    E = 3
    tauD = .25
    tau=0.25
    initial = 1.5
    v0 = (g_2pop + np.sqrt(g_2pop ** 2 + 4 * (E - g_2pop))) / 2

    vpop2, spktimes2, J2 = neuron_population(J_choice, t_max2, dt, 2, tau, E, tauD, initial, g_2pop,
                                             0)  # data for ax1,2
    spike_y = 2.8

    fig = plt.figure(layout='constrained')
    plot_single_neuron(vpop2, t_max2, dt, 0, 'r')
    plot_spike_times(spike_y, spktimes2, tauD, dt, 1, 0, 'r|')
    plot_single_neuron(vpop2, t_max2, dt, 1, 'b')
    plot_spike_times(spike_y, spktimes2, tauD, dt, 0, 1, 'b|')
    plt.xlim([0, t_max2])
    plt.ylim([0, 3])

    fig.set_size_inches(2, 2)  # width by height in inches
    sns.despine(fig)
    plt.tight_layout()
    #plt.savefig('fig1v2.png')
    plt.show()

    if J_choice==0: #delta func
        np.save('fig1_data/vpop_delay_visual_data_J_-1_E_3_D_.2.npy', vpop2)
        np.save('fig1_data/spikes_delay_visual_data_J_-1_E_3_D_.2.npy', spktimes2)
    if J_choice==1: #exponential func
        np.save('more_realistic_cases_data/vpop_exp_delay_visual_data_J_-1_E_3_D_.2.npy', vpop2)
        np.save('more_realistic_cases_data/spikes_exp_delay_visual_data_J_-1_E_3_D_.2.npy', spktimes2)
    if J_choice==2:
        np.save('more_realistic_cases_data/vpop_alpha_delay_visual_data_J_-1_E_3_D_.2.npy', vpop2)
        np.save('more_realistic_cases_data/spikes_alpha_delay_visual_data_J_-1_E_3_D_.2.npy', spktimes2)

    return


def build_fig_1_v1():
    t_max = 10
    t_max2 = 1.5
    dt = 0.001
    N = 1000
    g = -4  # height of conductance spike
    tau = 0  # width of conductance spike
    tauD = 0.2  # the time delay of the synaptic filter
    E = 3  # Resting Voltage
    initial = 1.5

    # tauD = int(tauD / dt)
    v0 = (g + np.sqrt(g ** 2 + 4 * (E - g))) / 2

    # for meanField
    Nt = int(t_max / dt)
    Ndelay = int(tauD / dt)
    tplot = np.arange(0, tauD, dt)

    # data for N=2 population
    g_2pop = -1
    vpop2, spktimes2, J2 = neuron_population(1, t_max2, dt, 2, tau, E, tauD, initial, g_2pop, 0)  # data for ax1,2

    # data for oscillatory large population
    vpop, spktimes, J = neuron_population(1, t_max, dt, N, tau, E, 1, initial, -5, 0)  # data for ax5,6

    # data for non-oscillatory population
    vpop3, spktimes3, J3 = neuron_population(1, t_max, dt, N, tau, E, tauD, initial, -2, 0)  # data for ax3,4

    # Figure 1:
    fig = plt.figure(layout='constrained')
    gs = GridSpec(2, 6, figure=fig)

    ax1 = fig.add_subplot(gs[0, :2])
    spike_y = 2.8
    plot_single_neuron(vpop2, t_max2, dt, 0, 'r')
    plot_spike_times(spike_y, spktimes2, tauD, dt, 1, 0, 'r|')
    plot_single_neuron(vpop2, t_max2, dt, 1, 'b')
    plot_spike_times(spike_y, spktimes2, tauD, dt, 0, 1, 'b|')
    plt.xlim([0, t_max2])
    plt.ylim([0, 3])
    plt.ylabel("Neuron Voltage")
    plt.xlabel("Time")

    ax2 = fig.add_subplot(gs[1, :2])
    connection_prob = 0.5
    NN = 30
    gg = 4
    g_bar = np.random.binomial(n=1, p=connection_prob, size=(NN, NN)) * gg / connection_prob / NN
    np.fill_diagonal(g_bar, 0)  # make sure connection =0 if i=j
    plot_network_graph_diagram(g_bar)

    ax3 = fig.add_subplot(gs[0, 2:4])
    plot_pop(spktimes3, dt)
    plt.xlim([0, t_max])
    plt.ylim([0, N])
    # plt.title('J = -2')
    plt.ylabel("Neuron Index")

    ax4 = fig.add_subplot(gs[1, 2:4])
    plot_avg(t_max, dt, vpop3)
    plt.xlim([0, t_max])
    plt.ylim([0, 1.75])
    plt.ylabel("Avg Population Voltage")

    ax5 = fig.add_subplot(gs[0, 4:])
    plot_pop(spktimes, dt)
    plt.xlim([0, t_max])
    plt.ylim([0, N])
    # plt.title('J = -18')

    ax6 = fig.add_subplot(gs[1, 4:])
    plot_avg(t_max, dt, vpop)
    plt.xlim([0, t_max])
    plt.ylim([0, 1.75])
    plt.xlabel("Time")


    #############
    fig.set_size_inches(7.5, 5)  # width by height in inches
    sns.despine(fig)
    plt.tight_layout()
    plt.savefig('figs_saved/fig1_v1.png')
    plt.show()
    return

def build_fig_1_v2():
    fig = plt.figure()
    fig.set_size_inches(6.5, 4.25)  # width by height in inches

    ## Setting locations in the grid:
    gs = GridSpec(2, 3, figure=fig)
    gs_spikedelay = gs[0,1]
    gs_network = gs[0,0]
    gs_homograster = gs[0,2]
    gs_oscraster = gs[1,0]
    gs_bumpraster = gs[1,1]
    gs_spatiotempraster = gs[1,2]

    t_max = 30
    dt = 0.001
    N = 1000
    g = -4  # height of conductance spike
    tau = 0  # width of conductance spike
    tauD = 0.2  # the time delay of the synaptic filter
    E = 3  # Resting Voltage
    initial = 1.5

    # tauD = int(tauD / dt)
    #v0 = (g + np.sqrt(g ** 2 + 4 * (E - g))) / 2

    # data for N=2 population
    #g_2pop = -1
    #vpop2, spktimes2, J2 = neuron_population(1, t_max2, dt, 2, g_2pop, tau, E, tauD, initial)  # data for ax1,2

    # data for oscillatory large population
    #vpop, spktimes, J = neuron_population(1, t_max, dt, N, tau, E, 1, initial, -8, 0)  # data for ax5,6

    # data for non-oscillatory population
    vpop3, spktimes3, J3 = neuron_population(1, t_max, dt, N, tau, E, tauD, initial, -2, 0)  # data for ax3,4


    #delay
    ax_spikedelay = fig.add_subplot(gs_spikedelay)
    spike_y = 2.8
    t_max2 = 1.5
    vpop2 = np.load('fig1_data/vpop_delay_visual_data_J_-1_E_3_D_.2.npy')
    spktimes2 = np.load('fig1_data/spikes_delay_visual_data_J_-1_E_3_D_.2.npy')
    plot_single_neuron(vpop2, t_max2, .01, 0, 'r')
    plot_spike_times(spike_y, spktimes2, tauD, .01, 1, 0, 'r|')
    plot_single_neuron(vpop2, t_max2, .01, 1, 'b')
    plot_spike_times(spike_y, spktimes2, tauD, .01, 0, 1, 'b|')
    plt.xlim([0, t_max2])
    plt.ylim([0, 3])
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel("Time (t)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    #network
    ax_network = fig.add_subplot(gs_network)
    connection_prob = 0.5
    NN = 15
    gg = 4
    g_bar = np.random.binomial(n=1, p=connection_prob, size=(NN, NN)) * gg / connection_prob / NN
    np.fill_diagonal(g_bar, 0)  # make sure connection =0 if i=j
    plot_network_graph_diagram(g_bar)

    #homog
    ax_homograster = fig.add_subplot(gs_homograster)
    plot_pop(spktimes3, dt, xlim=[0, t_max], ylim=[0, N], fontsize=fontsize, labelsize=labelsize)

    #spatiotemp
    ax_spatiotempraster = fig.add_subplot(gs_spatiotempraster)
    #vpop = np.load('../spiking_network_patterns_fig/pop_voltages_N_1000_dt_.001_D_1_E_2_J0_-2_J1_-20.npy')
    spktimes = np.load('spiking_network_patterns_fig/spktimes_N_1000_dt_.001_D_1_E_2_J0_-2_J1_-20.npy')
    N = 1000
    dt = .001
    plot_pop(spktimes, dt, xlim=[0, t_max], ylim=[0, N], fontsize=fontsize, labelsize=labelsize)

    #bump
    ax_bumpraster = fig.add_subplot(gs_bumpraster)
    #vpop = np.load('../spiking_network_patterns_fig/pop_voltages_N_1000_dt_.001_D_1_E_2_J0_-2_J1_20.npy')
    spktimes = np.load('spiking_network_patterns_fig/spktimes_N_1000_dt_.001_D_1_E_2_J0_-2_J1_20.npy')
    #plt.gca().set_yticklabels([])
    N = 1000
    plot_pop(spktimes, .001, xlim=[0, t_max], ylim=[0, N], fontsize=fontsize, labelsize=labelsize)

    #uniform oscillation
    ax_oscraster = fig.add_subplot(gs_oscraster)
    #vpop = np.load('../spiking_network_patterns_fig/pop_voltages_N_1000_dt_.001_D_1_E_2_J0_-15_J1_0.npy')
    spktimes = np.load('spiking_network_patterns_fig/spktimes_N_1000_dt_.001_D_1_E_2_J0_-15_J1_0.npy')
    N = 1000
    plot_pop(spktimes, .001, xlim=[0, t_max], ylim=[0, N], fontsize=fontsize, labelsize=labelsize)

    #############
    plt.tight_layout(rect=[0, 0, 1, 0.95]) #change to tight layout before adding labels
    letter_xoffset = -0.07
    letter_yoffset = 0.02

    add_subplot_label(fig, ax_network, 'A',x_offset=letter_xoffset, y_offset= letter_yoffset, fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikedelay, 'B', x_offset=letter_xoffset, y_offset=letter_yoffset, fontsize=letterlabelsize)
    add_subplot_label(fig, ax_homograster, 'C', x_offset=letter_xoffset, y_offset=letter_yoffset, fontsize=letterlabelsize)
    add_subplot_label(fig, ax_oscraster, 'D', x_offset=letter_xoffset, y_offset=letter_yoffset, fontsize=letterlabelsize)
    add_subplot_label(fig, ax_bumpraster, 'E', x_offset=letter_xoffset, y_offset=letter_yoffset, fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spatiotempraster, 'F', x_offset=letter_xoffset, y_offset=letter_yoffset, fontsize=letterlabelsize)


    #####

    sns.despine(fig)
    #plt.tight_layout()
    #plt.savefig('fig1v2.png')
    plt.savefig('figs_saved/fig1v2.pdf', format='pdf')
    plt.show()
    return


if __name__ == '__main__':
    build_fig_1_v2()
    #two_neuron_delay_visualization_save()

