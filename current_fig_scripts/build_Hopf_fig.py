import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from functions_and_sims.mean_field_equation_functions import calculate_mean_field, plot_mean_field, plot_spatial_MF
from functions_and_sims.matlab_import_functions import get_periodic_branch_points
#from calculateMeanFieldProperties import period_varied_J, period_varied_E, period_varied_tau
from functions_and_sims.spiking_network_functions import neuron_population, plot_avg
import seaborn as sns
from functions_and_sims.plot_bifurcation_curves_and_phase_diag import plot_Hopf_curves_varied_D, plot_Hopf_phase_diag
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from functions_and_sims.visualization_functions import add_subplot_label
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


fontsize = 8
labelsize = 6
letterlabelsize =10
linewidth = 1
datadotsize = 4

def build_Hopffig_v1():
    t_max = 40
    t_max2 = 5
    dt = 0.001
    N = 500
    g = -4  # height of conductance spike
    tau = 0  # width of conductance spike
    tauD = 2  # the time delay of the conductivity
    E = 3  # Resting Voltage

    v0 = (g + np.sqrt(g ** 2 + 4 * (E - g))) / 2

    # for meanField
    Nt = int(t_max / dt)
    Ndelay = int(tauD / dt)
    tplot = np.arange(0, tauD, dt)

    data_dot_color = 'grey'
    non_osc_color = '#1f77b4'  # blue
    osc_color = '#ff7f0e'  # orange

    ####################################################################################################################
    # Figure 2:
    fig2 = plt.figure(layout='constrained')
    gs = GridSpec(12, 8, figure=fig2)
    initial = 1.5

    # 1. The Non-Oscillatory Mean-Field and Pop-avg:
    J_non_osc = -2.5
    ax7 = fig2.add_subplot(gs[0:2, :2])

    # Calculate Data
    vmean7, v07 = calculate_mean_field(1, E, J_non_osc, tauD, t_max, dt, initial, tau)  # data for ax 7
    vpop, spktimes, J1 = neuron_population(1, t_max, dt, N, tau, E, tauD, initial, J_non_osc, 0)

    # Plot
    plt.plot(np.arange(0, t_max, dt), v07 * np.ones(Nt), 'r-')
    plot_mean_field(vmean7, t_max, dt, v07, color='k')
    plot_avg(t_max, dt, vpop, color=non_osc_color)
    plt.ylabel("Voltage")
    # plt.text(20,1.475, f'J = {J_non_osc}')

    ax7.text(0, 2.1, '\u25A0', fontsize=10, color=non_osc_color, verticalalignment='bottom',
             horizontalalignment='right')  # Circle for Non-Oscillatory

    plt.ylim([0, 2])
    plt.xlim([0, t_max])
    # plt.xlabel("Time")

    # Customize legend: remove border and add labels
    # plt.legend(handles=[plt.Line2D([0], [0], color=non_osc_color, linestyle='-', label='Simulation Pop Avg (Non-Osc)')]+
    #    [plt.Line2D([0], [0], color='k', linestyle='-', label='Mean Field')] +
    #    [plt.Line2D([0], [0], color='r', linestyle='-', label=r'Equilibrium $(v_+)$')],
    #           frameon=False, fontsize='small')

    # 2. The Oscillatory Mean-Field and Pop-avg:
    J_osc = -4
    ax8 = fig2.add_subplot(gs[2:4, :2])

    # Calculate Data
    vmean8, v08 = calculate_mean_field(1, E, J_osc, tauD, t_max, dt, initial, tau)  # data for ax 8
    vpop2, spktimes2, J2 = neuron_population(1, t_max, dt, N, tau, E, tauD, initial, J_osc,0)

    # Plot
    plt.plot(np.arange(0, t_max, dt), v08 * np.ones(Nt), 'r--')
    plot_mean_field(vmean8, t_max, dt, v08, color='k')
    plot_avg(t_max, dt, vpop2, color=osc_color)

    ax8.text(0, 2.1, '\u25CF', fontsize=10, color=osc_color, verticalalignment='bottom',
             horizontalalignment='right')  # Circle for Non-Oscillatory

    plt.ylabel("Voltage")
    # plt.text(20, 1.8, f'J = {J_osc}')
    plt.ylim([0, 2])
    plt.xlim([0, t_max])
    # plt.xlabel("Time")
    # plt.legend(handles=[plt.Line2D([0], [0], color=osc_color, linestyle='-', label='Simulation Pop Avg (Osc)')],
    #           frameon=False, fontsize='small')

    # 3.
    ax9 = fig2.add_subplot(gs[4:8, :2])
    plot_Hopf_phase_diag(showSimloc=1)
    plt.ylabel("E")
    plt.xlabel("J")
    plt.xlim([-5, 5])
    plt.ylim([-1, 4])

    # 3.
    ax99 = fig2.add_subplot(gs[8:12, :2])
    plot_Hopf_curves_varied_D()
    plt.ylabel("E")
    plt.xlabel("J")
    plt.xlim([-10, 0])
    plt.ylim([0, 3])

    # 4. Varied J - Bifurcation Diagram
    ax10 = fig2.add_subplot(gs[0:4, 2:4])

    # Import and prep data from matlab file
    val_list = get_periodic_branch_points('figHopf_data/matlab_data_v2/branch4_Jsave.mat', 'J_value', 0, 1)

    # Get J value of Hopf Point
    Hopf_J_val = val_list[0]['J_value']
    J_min = np.min(val_list[:]['J_value'])
    J_max = 3  # np.max(val_list[:]['J_value'])
    J_unst = np.linspace(J_min, Hopf_J_val)
    J_stab = np.linspace(Hopf_J_val, J_max)

    plt.plot(J_unst, (J_unst + np.sqrt(J_unst ** 2 + 4 * (E - J_unst))) / 2, 'r--')  # equil unstable
    plt.plot(J_stab, (J_stab + np.sqrt(J_stab ** 2 + 4 * (E - J_stab))) / 2, 'r-')  # equil stable

    # Get the index where the periodic solutions branch becomes stable
    j = 1
    while val_list[j]['Stability_color'] == 'b*':
        j += 1

    plt.plot(val_list[:j]['J_value'], val_list[:j]['Max'], color='k', linestyle='--')  # max unstable
    plt.plot(val_list[j:]['J_value'], val_list[j:]['Max'], 'k-')  # max stable
    plt.plot(val_list[:j]['J_value'], val_list[:j]['Min'], 'k--')  # min unstable
    plt.plot(val_list[j:]['J_value'], val_list[j:]['Min'], 'k-')  # min stable

    # Plot max min from population spiking simulation
    param_list = np.load('figHopf_data/saves_sim_data_v2/delta_varied_J.npy')
    average_min_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_min_amp_varied_J.npy')
    average_max_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_max_amp_varied_J.npy')
    plt.plot(param_list, average_min_amplitude, color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude')
    plt.plot(param_list, average_max_amplitude, color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude')

    k = 2  # index of osc plot 2
    plt.plot(param_list[k], average_min_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    plt.plot(param_list[k], average_max_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    plt.plot(J_non_osc, 1.3, color=non_osc_color, marker='s')

    plt.ylabel("Voltage")
    plt.ylim([-0.5, 1.8])
    plt.xlim([-5, -2])
    # plt.xlabel("J")

    # 5. Varied J - L2 norm:
    ax11 = fig2.add_subplot(gs[4:8, 2:4])
    plt.plot(val_list[:j]['J_value'], val_list[:j]['L2'], 'k--', label='DDE-BIFTOOL')  # l2 norm
    plt.plot(val_list[j:]['J_value'], val_list[j:]['L2'], 'k-', label='DDE-BIFTOOL')  # l2 norm

    l2_sim = np.load('figHopf_data/saves_sim_data_v2/delta_l2_norm_varied_J.npy')
    plt.plot(param_list, l2_sim, color=data_dot_color, marker='o', linestyle=' ', label='Period')
    print(l2_sim)

    plt.ylabel("L2 Norm")
    plt.xlim([-5, -2])
    plt.ylim([0, 1.6])
    # plt.xlabel("J")

    # 6. Varied J - Period
    ax12 = fig2.add_subplot(gs[8:12, 2:4])
    # Period from BifTool:
    plt.plot(val_list[:j]['J_value'], val_list[:j]['Period'], 'k--')  # period
    plt.plot(val_list[j:]['J_value'], val_list[j:]['Period'], 'k-')  # period

    per_sim = np.load('figHopf_data/saves_sim_data_v2/delta_period_varied_J.npy')
    plt.plot(param_list, per_sim, color=data_dot_color, marker='o', linestyle=' ', label='Period')

    plt.ylabel("Period")
    plt.xlabel("Coupling Strength J")
    plt.xlim([-5, -2])

    # 7. Varied E Bifurcation Diagram
    ax13 = fig2.add_subplot(gs[0:4, 4:6])
    val_list = get_periodic_branch_points('figHopf_data/matlab_data_v2/branch4_Esave.mat', 'E_value', 1, 1)
    # Get E value of Hopf Point
    Hopf_E_val = val_list[0]['E_value']
    E_min = np.min(val_list[:]['E_value'])
    E_max = 8  # np.max(val_list[:]['E_value'])
    E_unst = np.linspace(E_min, Hopf_E_val)
    E_stab = np.linspace(Hopf_E_val, E_max)

    # For plotting equil line not given by biftool
    g = -4  # This should match the J value used to compute the branch
    plt.plot(E_unst, (g + np.sqrt(g ** 2 + 4 * (E_unst - g))) / 2, 'r--')  # equil unstable
    plt.plot(E_stab, (g + np.sqrt(g ** 2 + 4 * (E_stab - g))) / 2, 'r-')  # equil stable
    plt.plot([0, 1], [0, 1], 'r-')

    # Get the index where the periodic solutions branch becomes stable
    j = 1
    while val_list[j]['Stability_color'] == 'b*':
        j += 1

    plt.plot(val_list[:j]['E_value'], val_list[:j]['Max'], 'k--')  # max unstable
    plt.plot(val_list[j:]['E_value'], val_list[j:]['Max'], 'k-')  # max stable
    plt.plot(val_list[:j]['E_value'], val_list[:j]['Min'], 'k--')  # min unstable
    plt.plot(val_list[j:]['E_value'], val_list[j:]['Min'], 'k-')  # min stable
    # plt.ylabel("Voltage")
    # plt.xlabel("E")

    # Plot max min from population spiking simulation
    param_list = np.load('figHopf_data/saves_sim_data_v2/delta_varied_E.npy')
    average_min_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_min_amp_varied_E.npy')
    average_max_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_max_amp_varied_E.npy')
    plt.plot(param_list[:-1], average_min_amplitude[:-1], color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude')
    plt.plot(param_list[:-1], average_max_amplitude[:-1], color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude')

    k = 5  # index of osc plot 2
    plt.plot(param_list[k], average_min_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    plt.plot(param_list[k], average_max_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    plt.plot(J_non_osc, 1.3, color=non_osc_color, marker='s')

    plt.ylim([0, 3])
    plt.xlim([0, E_max])

    # 8. Varied E l2 norm
    ax14 = fig2.add_subplot(gs[4:8, 4:6])
    plt.plot(val_list[:j]['E_value'], val_list[:j]['L2'], 'k--', label='DDE-BIFTOOL')  # l2 norm
    plt.plot(val_list[j:]['E_value'], val_list[j:]['L2'], 'k-', label='DDE-BIFTOOL')  # l2 norm
    # plt.ylabel("L2 Norm")
    # plt.xlabel("E")
    l2_sim = np.load('figHopf_data/saves_sim_data_v2/delta_l2_norm_varied_E.npy')
    plt.plot(param_list, l2_sim, color=data_dot_color, marker='o', linestyle=' ', label='Period')

    plt.xlim([0, E_max])
    plt.ylim([0, 1.6])

    # 9. Varied E Period
    ax15 = fig2.add_subplot(gs[8:12, 4:6])
    plt.plot(val_list[:j]['E_value'], val_list[:j]['Period'], 'k--')  # period
    plt.plot(val_list[j:]['E_value'], val_list[j:]['Period'], 'k-')  # period
    # plt.ylabel("Period")

    per_sim = np.load('figHopf_data/saves_sim_data_v2/delta_period_varied_E.npy')
    plt.plot(param_list[:-1], per_sim[:-1], color=data_dot_color, marker='o', linestyle=' ', label='Period')

    plt.xlabel("Reversal Potential E")
    plt.xlim([0, E_max])
    plt.ylim([4, 6])

    # 10. Varied delay Bifurcation Diagram
    ax16 = fig2.add_subplot(gs[0:4, 6:8])
    val_list = get_periodic_branch_points('figHopf_data/matlab_data_v2/branch4_delaysave.mat', 'tau_vals', 2, 1)
    # Get E value of Hopf Point
    Hopf_tau_val = val_list[0]['tau_vals']
    tau_min = 0  # np.min(val_list[:]['tau_vals'])
    tau_max = 2  # np.max(val_list[:]['tau_vals'])
    tau_stab = np.linspace(tau_min, Hopf_tau_val)
    tau_unst = np.linspace(Hopf_tau_val, tau_max)

    v0 = (g + np.sqrt(g ** 2 + 4 * (E - g))) / 2

    plt.plot(tau_unst, v0 * np.ones(len(tau_unst)), 'r--')  # equil unstable
    plt.plot(tau_stab, v0 * np.ones(len(tau_stab)), 'r-')  # equil stable

    # Get the index where the periodic solutions branch becomes stable
    j = 1
    while val_list[j]['Stability_color'] == 'b*':
        j += 1

    plt.plot(val_list[:j]['tau_vals'], val_list[:j]['Max'], 'k--')  # max unstable
    plt.plot(val_list[j:]['tau_vals'], val_list[j:]['Max'], 'k-')  # max stable
    plt.plot(val_list[:j]['tau_vals'], val_list[:j]['Min'], 'k--')  # min unstable
    plt.plot(val_list[j:]['tau_vals'], val_list[j:]['Min'], 'k-')  # min stable

    # Plot max min from population spiking simulation
    param_list = np.load('figHopf_data/saves_sim_data_v2/delta_varied_delay.npy')
    average_min_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_min_amp_varied_delay.npy')
    average_max_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_max_amp_varied_delay.npy')
    plt.plot(param_list, average_min_amplitude, color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude')
    plt.plot(param_list, average_max_amplitude, color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude')

    k = 9  # index of osc plot 2
    plt.plot(param_list[k], average_min_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    plt.plot(param_list[k], average_max_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    plt.plot(J_non_osc, 1.3, color=non_osc_color, marker='s')

    # plt.ylabel("Voltage")
    # plt.xlabel("Tau")
    plt.ylim([0, 1.8])
    plt.xlim([tau_min, tau_max])

    # 11. Varied delay l2 norm
    ax17 = fig2.add_subplot(gs[4:8, 6:8])
    plt.plot(val_list[:j]['tau_vals'], val_list[:j]['L2'], 'k--', label='DDE-BIFTOOL')  # l2 norm
    plt.plot(val_list[j:]['tau_vals'], val_list[j:]['L2'], 'k-', label='DDE-BIFTOOL')  # l2 norm
    # plt.ylabel("L2 Norm")
    # plt.xlabel("Tau")
    l2_sim = np.load('figHopf_data/saves_sim_data_v2/delta_l2_norm_varied_delay.npy')
    plt.plot(param_list, l2_sim, color=data_dot_color, marker='o', linestyle=' ', label='Period')

    plt.xlim([tau_min, tau_max])
    plt.ylim([0, 1.25])

    # 12. Varied delay period
    ax18 = fig2.add_subplot(gs[8:12, 6:8])
    plt.plot(val_list[:j]['tau_vals'], val_list[:j]['Period'], 'k--')  # period
    plt.plot(val_list[j:]['tau_vals'], val_list[j:]['Period'], 'k-')  # period
    # plt.ylabel("Period")
    per_sim = np.load('figHopf_data/saves_sim_data_v2/delta_period_varied_delay.npy')
    plt.plot(param_list, per_sim, color=data_dot_color, marker='o', linestyle=' ', label='Period')

    plt.xlabel("Delay Time D")
    plt.xlim([tau_min, tau_max])

    #########################################################################################
    fig2.set_size_inches(14, 9)  # width by height in inches
    sns.despine(fig2)
    plt.tight_layout()
    plt.savefig('figs_saved/fig2_v1.png')
    plt.show()
    return


class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        # Extract the linestyles from the Line2D objects
        linestyle1 = orig_handle[0].get_linestyle()  # First Line2D
        linestyle2 = orig_handle[1].get_linestyle()  # Second Line2D

        linewidth1 = orig_handle[0].get_linewidth()  # First Line2D
        linewidth2 = orig_handle[1].get_linewidth()  # Second Line2D

        # Extract the colors from the Line2D objects
        color1 = orig_handle[0].get_color()  # First Line2D color
        color2 = orig_handle[1].get_color()  # Second Line2D color

        # Create two lines for the legend with different linestyles
        l1 = plt.Line2D([x0, x0 + width], [0.7 * height, 0.7 * height], linestyle=linestyle1, color=color1, linewidth=linewidth1)
        l2 = plt.Line2D([x0, x0 + width], [0.3 * height, 0.3 * height], linestyle=linestyle2, color=color2, linewidth=linewidth2)
        return [l1, l2]
def build_Hopffig_v2():
    fig2 = plt.figure()
    fig2.set_size_inches(6.5, 7)


    ## setting grid
    gs = GridSpec(4, 3, figure=fig2, height_ratios=[1, .75, .75, .75])
    gs_osc = gs[0, 1]
    #gs_oscCompare = gs[1:2,1]
    gs_phasePort = gs[0,0]
    gs_hopfCurveVaryD = gs[0,2]
    gs_J0BifDiag = gs[1,0]
    gs_EBifDiag = gs[1, 1]
    gs_DBifDiag = gs[1, 2]
    gs_J0L2norm = gs[2, 0]
    gs_EL2norm = gs[2, 1]
    gs_DL2norm = gs[2, 2]
    gs_J0period = gs[3, 0]
    gs_Eperiod = gs[3, 1]
    gs_Dperiod = gs[3, 2]

    ##################################################################################################################
    t_max = 25
    dt = 0.001
    N = 500
    g = -4  # height of conductance spike
    tau = 0  # width of conductance spike
    tauD = 1  # the time delay of the conductivity
    E = 2  # Resting Voltage
    J_osc = -6
    initial = 1.38

    # mean field sim:
    vmean8, v08 = calculate_mean_field(1, E, J_osc, tauD, t_max, dt, initial, tau)  # data for ax 8

    # spiking net sim:
    vpop2, spktimes2, J2 = neuron_population(1, t_max, dt, N, tau, E, tauD, initial, J_osc, 0)

    #######

    data_dot_color = 'grey'
    non_osc_color = '#1f77b4'  # blue
    osc_color = '#ff7f0e'  # orange


    # mean field osc plot:
    ax_osc = fig2.add_subplot(gs_osc)
    ax_osc.set_visible(False)
    ax_inset1 = inset_axes(ax_osc, width="90%", height="40%", loc='upper right')

    m = np.load('../SpatialMeanField/UniformOscildt.01_E2_D1_J0-6_J10.npy')
    plot_spatial_MF(m,.01,xlim=[0, t_max], fontsize=fontsize, labelsize=labelsize, color_bar=True, ax=ax_inset1)
    ax_inset1.get_xaxis().set_visible(False)

    # compare MF and spiking net osc:
    #ax_oscCompare = fig2.add_subplot(gs_oscCompare)
    ax_inset2 = inset_axes(ax_osc, width="90%", height="40%", loc='lower right')
    Nt = int(t_max / dt)
    plt.plot(np.arange(0, t_max, dt), v08 * np.ones(Nt), 'r--', linewidth=linewidth)
    plot_mean_field(vmean8, t_max, dt, v08, color='k', linewidth=linewidth)
    plot_avg(t_max, dt, vpop2, color=osc_color, linewidth=linewidth)
    plt.ylabel(r"$v$", fontsize=fontsize)
    plt.ylim([-.75, 1.7])
    plt.xlim([0, t_max])
    plt.xlabel(r"Time ($t$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    legend_line_width = 1
    # Create two line handles for each grouped legend item
    orange_line = Line2D([0], [0], color=osc_color, linestyle='-', linewidth=legend_line_width)
    black_line = Line2D([0], [0], color='k', linestyle='-', linewidth=legend_line_width)

    # Legend with grouped lines
    ax_inset2.legend(
        handles=[orange_line, black_line],
        labels=[r'$sLIF$', 'MF'],
        handler_map={tuple: AnyObjectHandler()},
        loc='lower center',
        fontsize=labelsize,
        frameon=False,
        ncol=2
    )

    # phase portrait:
    ax_phasePort = fig2.add_subplot(gs_phasePort)
    plot_Hopf_phase_diag(linewidth=linewidth)
    plt.ylabel(r"Resting Poten. ($E$)",fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Stren. ($J_0$)",fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([-5, 5.5])
    plt.ylim([-1, 4])

    # hopf curves for varying D:
    ax_hopfCurveVaryD = fig2.add_subplot(gs_hopfCurveVaryD)
    plot_Hopf_curves_varied_D(linewidth=linewidth)
    plt.ylabel(r"Resting Poten. ($E$)", fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Stren. ($J_0$)",fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([-10, 0])
    plt.ylim([0, 3])

    # 4. Varied J - Bifurcation Diagram
    ax_J0BifDiag = fig2.add_subplot(gs_J0BifDiag)

    # Import and prep data from matlab file
    val_list = get_periodic_branch_points('figHopf_data/matlab_data_v2/branch4_Jsave.mat', 'J_value', 0, 1)

    # Get J value of Hopf Point
    Hopf_J_val = val_list[0]['J_value']
    J_min = np.min(val_list[:]['J_value'])
    J_max = 3  # np.max(val_list[:]['J_value'])
    J_unst = np.linspace(J_min, Hopf_J_val)
    J_stab = np.linspace(Hopf_J_val, J_max)

    plt.plot(J_unst, (J_unst + np.sqrt(J_unst ** 2 + 4 * (E - J_unst))) / 2, 'r--', linewidth=linewidth)  # equil unstable
    plt.plot(J_stab, (J_stab + np.sqrt(J_stab ** 2 + 4 * (E - J_stab))) / 2, 'r-', linewidth=linewidth)  # equil stable

    # Get the index where the periodic solutions branch becomes stable
    j = 1
    while val_list[j]['Stability_color'] == 'b*':
        j += 1

    plt.plot(val_list[:j]['J_value'], val_list[:j]['Max'], color='k', linestyle='--', linewidth=linewidth)  # max unstable
    plt.plot(val_list[j:]['J_value'], val_list[j:]['Max'], 'k-', linewidth=linewidth)  # max stable
    plt.plot(val_list[:j]['J_value'], val_list[:j]['Min'], 'k--', linewidth=linewidth)  # min unstable
    plt.plot(val_list[j:]['J_value'], val_list[j:]['Min'], 'k-', linewidth=linewidth)  # min stable

    # Plot max min from population spiking simulation
    param_list = np.load('figHopf_data/saves_sim_data_v2/delta_varied_J.npy')
    average_min_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_min_amp_varied_J.npy')
    average_max_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_max_amp_varied_J.npy')
    plt.plot(param_list, average_min_amplitude, color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude', ms =datadotsize)
    plt.plot(param_list, average_max_amplitude, color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude', ms =datadotsize)

    #k = 2  # index of osc plot 2
    #plt.plot(param_list[k], average_min_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    #plt.plot(param_list[k], average_max_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    #plt.plot(J_non_osc, 1.3, color=non_osc_color, marker='s')

    legend_line_width = 1
    # Create two line handles for each grouped legend item
    red_lines = (
        Line2D([0], [0], color='r', linestyle='-', linewidth=legend_line_width),
        Line2D([0], [0], color='r', linestyle='--', linewidth=legend_line_width)
    )

    black_lines = (
        Line2D([0], [0], color='k', linestyle='-', linewidth=legend_line_width),
        Line2D([0], [0], color='k', linestyle='--', linewidth=legend_line_width)
    )

    grey_dot = Line2D([0], [0], color='gray', marker='o', linestyle='None', ms =datadotsize-1)

    # Legend with grouped lines
    ax_J0BifDiag.legend(
        handles=[red_lines, black_lines, grey_dot],
        labels=[r'$v_+$', 'MF $v_{osc}$', 'sLIF $v_{osc}$'],
        handler_map={tuple: AnyObjectHandler()},
        loc='lower right',
        fontsize=labelsize,
        frameon=False
    )

    plt.ylabel(r"Voltage ($v$)",fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Stren. ($J_0$)",fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.ylim([-0.5, 1.8])
    plt.xlim([-5, -2])



    # 5. Varied J - L2 norm:
    ax_J0L2norm = fig2.add_subplot(gs_J0L2norm)
    plt.plot(val_list[:j]['J_value'], val_list[:j]['L2'], 'k--', label='DDE-BIFTOOL', linewidth=linewidth)  # l2 norm
    plt.plot(val_list[j:]['J_value'], val_list[j:]['L2'], 'k-', label='DDE-BIFTOOL', linewidth=linewidth)  # l2 norm
    #J_vals, period, l2 = period_varied_J(1, E, -3.12, -3.03, tau, tauD, t_max, dt)

    l2_sim = np.load('figHopf_data/saves_sim_data_v2/delta_l2_norm_varied_J.npy')
    plt.plot(param_list, l2_sim, color=data_dot_color, marker='o', linestyle=' ', label='Period', ms =datadotsize)
    print(l2_sim)

    plt.ylabel(r"$||v-v_+||_{L^2}$", fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Stren. ($J_0$)",fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([-5, -2])
    plt.ylim([0, 1.6])
    # plt.xlabel("J")

    # 6. Varied J - Period
    ax_J0period = fig2.add_subplot(gs_J0period)

    # Period from BifTool:
    plt.plot(val_list[:j]['J_value'], val_list[:j]['Period'], 'k--', linewidth=linewidth)  # period
    plt.plot(val_list[j:]['J_value'], val_list[j:]['Period'], 'k-', linewidth=linewidth)  # period

    per_sim = np.load('figHopf_data/saves_sim_data_v2/delta_period_varied_J.npy')
    plt.plot(param_list, per_sim, color=data_dot_color, marker='o', linestyle=' ', label='Period', ms =datadotsize)

    plt.ylabel("Oscillation Period", fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Stren. ($J_0$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([-5, -2])

    # 7. Varied E Bifurcation Diagram
    ax_EBifDiag = fig2.add_subplot(gs_EBifDiag)
    val_list = get_periodic_branch_points('figHopf_data/matlab_data_v2/branch4_Esave.mat', 'E_value', 1, 1)
    # Get E value of Hopf Point
    Hopf_E_val = val_list[0]['E_value']
    E_min = np.min(val_list[:]['E_value'])
    E_max = 8  # np.max(val_list[:]['E_value'])
    E_unst = np.linspace(E_min, Hopf_E_val)
    E_stab = np.linspace(Hopf_E_val, E_max)

    # For plotting equil line not given by biftool
    g = -4  # This should match the J value used to compute the branch
    plt.plot(E_unst, (g + np.sqrt(g ** 2 + 4 * (E_unst - g))) / 2, 'r--', linewidth=linewidth)  # equil unstable
    plt.plot(E_stab, (g + np.sqrt(g ** 2 + 4 * (E_stab - g))) / 2, 'r-', linewidth=linewidth)  # equil stable
    plt.plot([0, 1], [0, 1], 'r-', linewidth=linewidth)

    # Get the index where the periodic solutions branch becomes stable
    j = 1
    while val_list[j]['Stability_color'] == 'b*':
        j += 1

    plt.plot(val_list[:j]['E_value'], val_list[:j]['Max'], 'k--', linewidth=linewidth)  # max unstable
    plt.plot(val_list[j:]['E_value'], val_list[j:]['Max'], 'k-', linewidth=linewidth)  # max stable
    plt.plot(val_list[:j]['E_value'], val_list[:j]['Min'], 'k--', linewidth=linewidth)  # min unstable
    plt.plot(val_list[j:]['E_value'], val_list[j:]['Min'], 'k-', linewidth=linewidth)  # min stable
    plt.ylabel(r"Voltage ($v$)", fontsize=fontsize)
    plt.xlabel(r"Resting Poten. ($E$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    # Plot max min from population spiking simulation
    param_list = np.load('figHopf_data/saves_sim_data_v2/delta_varied_E.npy')
    average_min_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_min_amp_varied_E.npy')
    average_max_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_max_amp_varied_E.npy')
    plt.plot(param_list[:-1], average_min_amplitude[:-1], color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude', ms =datadotsize)
    plt.plot(param_list[:-1], average_max_amplitude[:-1], color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude', ms =datadotsize)

    '''k = 5  # index of osc plot 2
    plt.plot(param_list[k], average_min_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    plt.plot(param_list[k], average_max_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    #plt.plot(J_non_osc, 1.3, color=non_osc_color, marker='s')'''

    plt.ylim([0, 3])
    plt.xlim([0, E_max])

    # 8. Varied E l2 norm
    ax_EL2norm = fig2.add_subplot(gs_EL2norm)
    plt.plot(val_list[:j]['E_value'], val_list[:j]['L2'], 'k--', label='DDE-BIFTOOL', linewidth=linewidth)  # l2 norm
    plt.plot(val_list[j:]['E_value'], val_list[j:]['L2'], 'k-', label='DDE-BIFTOOL', linewidth=linewidth)  # l2 norm
    plt.ylabel(r"$||v-v_+||_{L^2}$", fontsize=fontsize)
    plt.xlabel(r"Resting Poten. ($E$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    l2_sim = np.load('figHopf_data/saves_sim_data_v2/delta_l2_norm_varied_E.npy')
    plt.plot(param_list, l2_sim, color=data_dot_color, marker='o', linestyle=' ', label='Period', ms =datadotsize)

    plt.xlim([0, E_max])
    plt.ylim([0, 1.6])

    # 9. Varied E Period
    ax_Eperiod = fig2.add_subplot(gs_Eperiod)
    plt.plot(val_list[:j]['E_value'], val_list[:j]['Period'], 'k--', linewidth=linewidth)  # period
    plt.plot(val_list[j:]['E_value'], val_list[j:]['Period'], 'k-', linewidth=linewidth)  # period

    per_sim = np.load('figHopf_data/saves_sim_data_v2/delta_period_varied_E.npy')
    plt.plot(param_list[:-1], per_sim[:-1], color=data_dot_color, marker='o', linestyle=' ', label='Period', ms =datadotsize)

    plt.ylabel("Oscillation Period", fontsize=fontsize)
    plt.xlabel(r"Resting Poten. ($E$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([0, E_max])
    plt.ylim([4.4, 5.8])

    # 10. Varied delay Bifurcation Diagram
    ax_DBifDiag = fig2.add_subplot(gs_DBifDiag)
    val_list = get_periodic_branch_points('figHopf_data/matlab_data_v2/branch4_delaysave.mat', 'tau_vals', 2, 1)
    # Get E value of Hopf Point
    Hopf_tau_val = val_list[0]['tau_vals']
    tau_min = 0  # np.min(val_list[:]['tau_vals'])
    tau_max = 2  # np.max(val_list[:]['tau_vals'])
    tau_stab = np.linspace(tau_min, Hopf_tau_val)
    tau_unst = np.linspace(Hopf_tau_val, tau_max)

    v0 = (g + np.sqrt(g ** 2 + 4 * (E - g))) / 2

    plt.plot(tau_unst, v0 * np.ones(len(tau_unst)), 'r--', linewidth=linewidth)  # equil unstable
    plt.plot(tau_stab, v0 * np.ones(len(tau_stab)), 'r-', linewidth=linewidth)  # equil stable

    # Get the index where the periodic solutions branch becomes stable
    j = 1
    while val_list[j]['Stability_color'] == 'b*':
        j += 1

    plt.plot(val_list[:j]['tau_vals'], val_list[:j]['Max'], 'k--', linewidth=linewidth)  # max unstable
    plt.plot(val_list[j:]['tau_vals'], val_list[j:]['Max'], 'k-', linewidth=linewidth)  # max stable
    plt.plot(val_list[:j]['tau_vals'], val_list[:j]['Min'], 'k--', linewidth=linewidth)  # min unstable
    plt.plot(val_list[j:]['tau_vals'], val_list[j:]['Min'], 'k-', linewidth=linewidth)  # min stable

    # Plot max min from population spiking simulation
    param_list = np.load('figHopf_data/saves_sim_data_v2/delta_varied_delay.npy')
    average_min_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_min_amp_varied_delay.npy')
    average_max_amplitude = np.load('figHopf_data/saves_sim_data_v2/delta_max_amp_varied_delay.npy')
    plt.plot(param_list, average_min_amplitude, color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude', ms =datadotsize)
    plt.plot(param_list, average_max_amplitude, color=data_dot_color, marker='o', linestyle=' ',
             label='Average Min Amplitude', ms =datadotsize)

    '''k = 9  # index of osc plot 2
    plt.plot(param_list[k], average_min_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    plt.plot(param_list[k], average_max_amplitude[k], color=osc_color, marker='o', linestyle=' ')
    #plt.plot(J_non_osc, 1.3, color=non_osc_color, marker='s')'''

    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel("Delay (D)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.ylim([0, 1.8])
    plt.xlim([tau_min, tau_max])

    # 11. Varied delay l2 norm
    ax_DL2norm= fig2.add_subplot(gs_DL2norm)
    plt.plot(val_list[:j]['tau_vals'], val_list[:j]['L2'], 'k--', label='DDE-BIFTOOL', linewidth=linewidth)  # l2 norm
    plt.plot(val_list[j:]['tau_vals'], val_list[j:]['L2'], 'k-', label='DDE-BIFTOOL', linewidth=linewidth)  # l2 norm

    l2_sim = np.load('figHopf_data/saves_sim_data_v2/delta_l2_norm_varied_delay.npy')
    plt.plot(param_list, l2_sim, color=data_dot_color, marker='o', linestyle=' ', label='Period', ms =datadotsize)

    plt.ylabel(r"$||v-v_+||_{L^2}$", fontsize=fontsize)
    plt.xlabel(r"Delay ($D$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([tau_min, tau_max])
    plt.ylim([0, 1.25])

    # 12. Varied delay period
    ax_Dperiod = fig2.add_subplot(gs_Dperiod)
    plt.plot(val_list[:j]['tau_vals'], val_list[:j]['Period'], 'k--', linewidth=linewidth)  # period
    plt.plot(val_list[j:]['tau_vals'], val_list[j:]['Period'], 'k-', linewidth=linewidth)  # period

    per_sim = np.load('figHopf_data/saves_sim_data_v2/delta_period_varied_delay.npy')
    plt.plot(param_list, per_sim, color=data_dot_color, marker='o', linestyle=' ', label='Period', ms =datadotsize)

    plt.ylabel("Oscillation Period", fontsize=fontsize)
    plt.xlabel(r"Delay ($D$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([tau_min, tau_max])


    ####### Lettering the plots
    plt.tight_layout(h_pad=2, w_pad=2, rect=[0, 0, .98, 0.98]) #change to tight layout before adding labels
    letter_xoffset = -0.08
    letter_yoffset = 0.01

    add_subplot_label(fig2, ax_phasePort, 'A',x_offset=letter_xoffset, y_offset= letter_yoffset, fontsize=letterlabelsize)
    add_subplot_label(fig2, ax_osc, 'B', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig2, ax_hopfCurveVaryD, 'C', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig2, ax_J0BifDiag, 'D', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig2, ax_EBifDiag, 'E', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig2, ax_DBifDiag, 'F', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig2, ax_J0L2norm, 'G', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig2, ax_EL2norm, 'H', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig2, ax_DL2norm, 'I', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig2, ax_J0period, 'J', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig2, ax_Eperiod, 'K', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig2, ax_Dperiod, 'L', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)

    #########################################################################################
    sns.despine(fig2)
    #plt.tight_layout()

    plt.savefig('figs_saved/Hopffig_v2.pdf')
    plt.show()
    return

if __name__ == '__main__':
    build_Hopffig_v2()


