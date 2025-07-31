import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from publish.functions_and_sims.spiking_network_functions import plot_single_neuron, plot_spike_times
import seaborn as sns
from publish.functions_and_sims.plot_bifurcation_curves_and_phase_diag import plot_alph_func_instabilties_E_gthan_1, plot_alph_func_instabilties_E_lessthan_1
from publish.functions_and_sims.plot_bifurcation_curves_and_phase_diag import plot_del_exp_instabilties_E_gthan_1, plot_del_exp_instabilties_E_lessthan_1
from publish.functions_and_sims.visualization_functions import plot_delayed_exp_function, plot_alpha_function, add_subplot_label

fontsize = 8
labelsize = 6
letterlabelsize =10
linewidth = 1

def build_fig_syn_respose():
    fig = plt.figure(layout='constrained')
    gs = GridSpec(2, 4, figure=fig)
    fig.set_size_inches(6.5, 3)  # width by height in inches
    gs_delExp = gs[0,0]
    gs_delExpSpike = gs[0,1]
    gs_delExpAbvThres = gs[0,2]
    gs_delExpBelowThres = gs[0,3]
    gs_alphFunc = gs[1,0]
    gs_alphFuncSpike = gs[1, 1]
    gs_alphFuncAbvThres = gs[1, 2]
    gs_alphFuncBelowThres = gs[1, 3]


    t_max = 20
    dt = 0.001
    N = 1000
    g = -3  # height of conductance spike
    tau = 1  # width of conductance spike
    tauD = .2  # the time delay of the conductivity
    E = 3  # Resting Voltage
    initial = 1.5

    # tauD = int(tauD / dt)
    v0 = (g + np.sqrt(g ** 2 + 4 * (E - g))) / 2

    J2 = 0  # placeholder
    t_max2 = 1.5
    vpop2 = np.load(
        'more_realistic_cases_data/vpop_exp_delay_visual_data_J_-1_E_3_D_.2.npy')
    spktimes2 = np.load(
        'more_realistic_cases_data/spikes_exp_delay_visual_data_J_-1_E_3_D_.2.npy')

    # Figure 1:


    ax_delExp = fig.add_subplot(gs_delExp)
    plot_delayed_exp_function(linewidth=linewidth)
    plt.tick_params(labelsize=labelsize)
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel("Time (t)", fontsize=fontsize)

    ax_delExpSpike = fig.add_subplot(gs_delExpSpike)
    spike_y = 3
    plot_single_neuron(vpop2, t_max2, dt, 0, 'r', linewidth=linewidth)
    plot_spike_times(spike_y, spktimes2, tauD, dt, 1, 0, 'r|', labelsize=labelsize)
    plot_single_neuron(vpop2, t_max2, dt, 1, 'b', linewidth=linewidth)
    plot_spike_times(spike_y, spktimes2, tauD, dt, 0, 1, 'b|', labelsize=labelsize)
    plt.xlim([0, t_max2])
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel("Time (t)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ax_delExpAbvThres = fig.add_subplot(gs_delExpAbvThres)
    plot_del_exp_instabilties_E_gthan_1(linewidth=linewidth)
    plt.ylabel(r"$J_1$", fontsize=fontsize)
    plt.xlabel(r"$J_0$", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ax_delExpBelowThres = fig.add_subplot(gs_delExpBelowThres)
    plot_del_exp_instabilties_E_lessthan_1(linewidth=linewidth)
    plt.ylabel(r"$J_1$", fontsize=fontsize)
    plt.xlabel(r"$J_0$", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)


    ## ALPHA
    # data for N=2 population
    vpop2 = np.load(
        'more_realistic_cases_data/vpop_alpha_delay_visual_data_J_-1_E_3_D_.2.npy')
    spktimes2 = np.load(
        'more_realistic_cases_data/spikes_alpha_delay_visual_data_J_-1_E_3_D_.2.npy')

    ax_alphFunc = fig.add_subplot(gs_alphFunc)
    plot_alpha_function(linewidth=linewidth)
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel("Time (t)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ax_alphFuncSpike = fig.add_subplot(gs_alphFuncSpike)
    spike_y = 3
    plot_single_neuron(vpop2, t_max2, dt, 0, 'r', linewidth=linewidth)
    plot_spike_times(spike_y, spktimes2, tauD, dt, 1, 0, 'r|', labelsize=labelsize)
    plot_single_neuron(vpop2, t_max2, dt, 1, 'b', linewidth=linewidth)
    plot_spike_times(spike_y, spktimes2, tauD, dt, 0, 1, 'b|', labelsize=labelsize)
    plt.xlim([0, t_max2])
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel("Time (t)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ax_alphFuncAbvThres = fig.add_subplot(gs_alphFuncAbvThres)
    plot_alph_func_instabilties_E_gthan_1()
    plt.ylabel(r"$J_1$", fontsize=fontsize)
    plt.xlabel(r"$J_0$", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ax_alphFuncBelowThres = fig.add_subplot(gs_alphFuncBelowThres)
    plot_alph_func_instabilties_E_lessthan_1(linewidth=linewidth)
    plt.ylabel(r"$J_1$", fontsize=fontsize)
    plt.xlabel(r"$J_0$", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    #############
    plt.tight_layout(rect=[0.02, 0.02, .98, 0.98])
    letter_xoffset = -0.05
    letter_yoffset = 0.01
    add_subplot_label(fig, ax_delExp, 'A', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_delExpSpike, 'B', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_delExpAbvThres, 'C', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_delExpBelowThres, 'D', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_alphFunc, 'E', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_alphFuncSpike, 'F', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_alphFuncAbvThres, 'G', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_alphFuncBelowThres, 'H', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)


    #########
    sns.despine(fig)
    plt.savefig('figs_saved/more_realistic_cases.pdf')

    plt.show()
    return

if __name__ == '__main__':
    build_fig_syn_respose()