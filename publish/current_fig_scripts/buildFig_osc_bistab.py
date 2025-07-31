import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.gridspec import GridSpec
import seaborn as sns
from publish.functions_and_sims.mean_field_equation_functions import plot_ramped_mean_field_vs_param, plot_spatial_MF, MFsimulation_2_unif_pulses
import matplotlib.patches as patches
from publish.scripts_for_saved_data.create_data_for_Hopf_bistab_fig import plot_bif_curve_J_vs_v
from publish.functions_and_sims.spiking_network_functions import plot_pop
from publish.functions_and_sims.visualization_functions import add_subplot_label

fontsize = 8
labelsize = 6
letterlabelsize =10


def build_bistab_fig_one_col():
    fig = plt.figure()
    fig.set_size_inches(3.25, 7)  # width by height in inches
    gs = GridSpec(6, 2, figure=fig,
                  width_ratios=[2, .7],     # columns
                height_ratios=[1, 1, 1,1,1,1])   #rows
    gs_MFrampUp = gs[0, 0]
    gs_MFosc = gs[0,1]
    gs_spikingRampUp = gs[1,0]
    gs_spikingOsc = gs[1,1]
    gs_MFrampDown = gs[2,0]
    gs_MFhomg = gs[2,1]
    gs_spikingRampDown = gs[3,0]
    gs_spikingHomog = gs[3,1]
    gs_MFbistab = gs[4,:]
    gs_spikingBistab = gs[5,:]


    J_choice = 1
    J = -4
    E = 3
    tau = 1
    delay = 2
    dt = .001
    bistab_region_color = 'green'
    slice_color = 'blue'

    mean_zoom_start = -3.048
    mean_zoom_end = -3.047
    spknet_zoom_start = -3.223
    spknet_zoom_end = -3.205

    arrowxlocal=[-3.01,-3.1,-3.005,-3.075]
    arrowylocal=[1,0.9,1,0.9]
    arrowwidth=[.2,.2,.2,.2]
    arrowlength=[.01,.05,-.01,-.05]



    # 1. Mean Field osc->non osc
    ax_MFrampUp = fig.add_subplot(gs_MFrampUp)
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Str. ($J_0$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    param_start1 = -3.1
    param_end1 = -3
    tstop = 5000
    vmean1 = np.load('figHopf_bistab_data/fwdbwdv1/mfield_J-3.1to-3_dt001.npy')
    plot_ramped_mean_field_vs_param(vmean1, tstop, dt, param_start1, param_end1, color='grey')
    plot_bif_curve_J_vs_v(E, 'figHopf_data/matlab_data_v2/branch4_Jsave.mat')

    # Create a right-pointing arrow
    k = 0
    arrow = patches.Arrow(x=arrowxlocal[k], y=arrowylocal[k], dx=arrowlength[k], dy=0, width=arrowwidth[k], color='black')
    # Add the arrow to the plot
    ax_MFrampUp.add_patch(arrow)

    ax_MFrampUp.axvspan(-3.05, -3.03, facecolor=bistab_region_color, alpha=0.3)
    ax_MFrampUp.axvspan(mean_zoom_start, mean_zoom_end, facecolor=slice_color, alpha=0.3)
    plt.xlim([-3.08, -3])
    plt.ylim([0.8, 1.75])

    # 2. Spiking Net osc->non osc
    ax_spikingRampUp = fig.add_subplot(gs_spikingRampUp)
    vpop3 = np.load('figHopf_bistab_data/fwdbwdv1/spiking_J-3.5to-2.5_N2000_dt001.npy')
    param_start3 = -3.5
    param_end3 = -2.5
    plot_ramped_mean_field_vs_param(vpop3, 2000, dt, param_start3, param_end3, color='grey')
    # plot_bif_curve(E, 'figHopf_data/matlab_data_v2/branch4_Jsave.mat')

    #shaded regions
    ax_spikingRampUp.axvspan(-3.267, -3.2, facecolor=bistab_region_color, alpha=0.3)
    ax_spikingRampUp.axvspan(spknet_zoom_start, spknet_zoom_end, facecolor=slice_color, alpha=0.3)

    k = 1
    arrow = patches.Arrow(x=arrowxlocal[k], y=arrowylocal[k], dx=arrowlength[k], dy=0, width=arrowwidth[k],
                          color='black')
    ax_spikingRampUp.add_patch(arrow)

    plt.xlim([-3.4, -3.05])
    plt.ylim([0.7, 1.75])
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Str. ($J_0$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)


    # 3. Mean Field non osc->osc
    ax_MFrampDown = fig.add_subplot(gs_MFrampDown)
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Str. ($J_0$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    param_start2 = -3.04
    param_end2 = -3.1
    tstop = 5000
    vmean2 = np.load('figHopf_bistab_data/fwdbwdv1/mfield_J-3.04to-3.1_dt001.npy')
    plot_ramped_mean_field_vs_param(vmean2, tstop, dt, param_start2, param_end2, color='grey')
    plot_bif_curve_J_vs_v(E, 'figHopf_data/matlab_data_v2/branch4_Jsave.mat')
    ax_MFrampDown.axvspan(-3.05, -3.03, facecolor=bistab_region_color, alpha=0.3)
    ax_MFrampDown.axvspan(mean_zoom_start, mean_zoom_end, facecolor=slice_color, alpha=0.3)

    # Create a left-pointing arrow
    k = 2
    arrow = patches.Arrow(x=arrowxlocal[k], y=arrowylocal[k], dx=arrowlength[k], dy=0, width=arrowwidth[k],
                          color='black')
    ax_MFrampDown.add_patch(arrow)

    plt.xlim([-3.08, -3])
    plt.ylim([0.8, 1.75])

    # 4. Spiking Net non osc->osc
    ax_spikingRampDown = fig.add_subplot(gs_spikingRampDown)
    vpop4 = np.load('figHopf_bistab_data/fwdbwdv1/spiking_J-2.5to-3.5_N2000_dt001.npy')
    param_start4 = -2.5
    param_end4 = -3.5
    plot_ramped_mean_field_vs_param(vpop4, 2000, dt, param_start4, param_end4, color='grey')
    # plot_bif_curve(E, 'figHopf_data/matlab_data_v2/branch4_Jsave.mat')
    ax_spikingRampDown.axvspan(-3.267, -3.2, facecolor=bistab_region_color, alpha=0.3)
    ax_spikingRampDown.axvspan(spknet_zoom_start, spknet_zoom_end, facecolor=slice_color, alpha=0.3)

    #arrow
    k = 3
    arrow = patches.Arrow(x=arrowxlocal[k], y=arrowylocal[k], dx=arrowlength[k], dy=0, width=arrowwidth[k],
                          color='black')
    ax_spikingRampDown.add_patch(arrow)

    plt.xlim([-3.4, -3.05])
    plt.ylim([0.7, 1.75])
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Str. ($J_0$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    # 5
    ax_MFosc = fig.add_subplot(gs_MFosc)
    # ax5.get_xaxis().set_visible(False)
    plot_ramped_mean_field_vs_param(vmean1, tstop, dt, param_start1, param_end1, color='grey')
    ax_MFosc.axvspan(-3.05, -3.03, facecolor=slice_color, alpha=0.3)
    ax_MFosc.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax_MFosc.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.xlim([mean_zoom_start, mean_zoom_end])
    plt.ylim([0.8, 1.75])
    plt.xlabel(r'$J_0$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #ax5.set_yticklabels([])
    #plt.ylabel("Voltage (v)")

    # 6
    ax_MFhomg = fig.add_subplot(gs_MFhomg)
    # ax6.get_xaxis().set_visible(False)
    plot_ramped_mean_field_vs_param(vmean2, tstop, dt, param_start2, param_end2, color='grey')
    ax_MFhomg.axvspan(-3.05, -3.03, facecolor=slice_color, alpha=0.3)
    ax_MFhomg.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax_MFhomg.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.xlim([mean_zoom_start, mean_zoom_end])
    plt.ylim([0.8, 1.75])
    plt.xlabel(r'$J_0$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #ax6.set_yticklabels([])

    # 7
    ax_spikingOsc = fig.add_subplot(gs_spikingOsc)
    # ax7.get_xaxis().set_visible(False)
    plot_ramped_mean_field_vs_param(vpop3, 2000, dt, param_start3, param_end3, color='grey')
    ax_spikingOsc.axvspan(-3.26, -3.2, facecolor=slice_color, alpha=0.3)
    ax_spikingOsc.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    plt.xlim([spknet_zoom_start, spknet_zoom_end])
    plt.ylim([0.7, 1.75])
    plt.xlabel(r'$J_0$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    # 8
    ax_spikingHomog = fig.add_subplot(gs_spikingHomog)
    # ax8.get_xaxis().set_visible(False)
    plot_ramped_mean_field_vs_param(vpop4, 2000, dt, param_start4, param_end4, color='grey')
    ax_spikingHomog.axvspan(-3.26, -3.2, facecolor=slice_color, alpha=0.3)
    ax_spikingHomog.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    plt.xlim([spknet_zoom_start, spknet_zoom_end])
    plt.ylim([0.7, 1.75])
    plt.xlabel(r'$J_0$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ax_spikingBistab = fig.add_subplot(gs_spikingBistab)
    spktimes2 = np.load(
        'figHopf_bistab_data/single_param_bistab_data/spiktimes_J-3.22_E3_D2_N1000_T100_dt.001.npy')
    plot_pop(spktimes2, .001, plotevery=2)
    plt.xlim([0, 100])
    plt.ylim([0, 1000])
    plt.ylabel("Neuron (i)",fontsize=fontsize)
    plt.xlabel("Time (t)",fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ax_MFbistab = fig.add_subplot(gs_MFbistab)
    t_max = 100
    Nx = 500
    D = 2
    J0 = -3.04
    J_1 = 0
    dt = 0.01
    v0 = (J0 + np.sqrt(J0 ** 2 + 4 * (E - J0))) / 2
    x = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    m_history = np.zeros((int(D / dt), Nx))  # The history
    center = v0
    for tt in range(0, int(D / dt)):
        m_history[tt, :] = center
    m_initial = center

    m = MFsimulation_2_unif_pulses(m_initial, m_history, t_max, Nx, D, dt, J0, J_1, E)
    plt.imshow(m.T, aspect='auto', cmap='viridis', origin='lower', interpolation='none',
               extent=[0, len(m) * .01, -np.pi, np.pi])
    plt.ylabel('Space (x)', fontsize=fontsize)
    plt.xlabel('Time (t)',fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([0, t_max])

    #####################################
    plt.tight_layout(h_pad=2,rect=[0, 0, 1, .98])
    letter_xoffset = -0.04
    letter_yoffset = 0.01
    add_subplot_label(fig, ax_MFrampUp, 'A', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_MFosc, 'B', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikingRampUp, 'C', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikingOsc, 'D', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_MFrampDown, 'E', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_MFhomg, 'F', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikingRampDown, 'G', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikingHomog, 'H', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_MFbistab, 'I', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikingBistab, 'J', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)

    ####
    sns.despine(fig)
    #plt.savefig('fig_bistab_v2.png',dpi=fig.get_dpi())
    plt.show()

    return

def build_bistab_fig_v2():
    fig = plt.figure()
    fig.set_size_inches(6.5, 4)  # width by height in inches
    gs = GridSpec(3, 6, figure=fig,
                  width_ratios=[1, 1, 3, 1,1,3],     # columns
                height_ratios=[1, 1, 1])   #rows
    gs_MFrampUp = gs[0, 0:2]
    gs_MFosc = gs[0,2]
    gs_spikingRampUp = gs[1,0:2]
    gs_spikingOsc = gs[1,2]
    gs_MFrampDown = gs[0,3:5]
    gs_MFhomg = gs[0,5]
    gs_spikingRampDown = gs[1,3:5]
    gs_spikingHomog = gs[1,5]

    gs_MFbistab = gs[2,:3]
    gs_spikingBistab = gs[2,3:]


    J_choice = 1
    J = -4
    E = 3
    tau = 1
    delay = 2
    dt = .001
    bistab_region_color = 'green'
    slice_color = 'blue'

    mean_zoom_start = -3.048
    mean_zoom_end = -3.047
    spknet_zoom_start = -3.223
    spknet_zoom_end = -3.205

    arrowxlocal=[-3.01,-3.1,-3.005,-3.075]
    arrowylocal=[1,0.9,1,0.9]
    arrowwidth=[.2,.2,.2,.2]
    arrowlength=[.01,.05,-.01,-.05]



    # 1. Mean Field osc->non osc
    ax_MFrampUp = fig.add_subplot(gs_MFrampUp)
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Str. ($J_0$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    param_start1 = -3.1
    param_end1 = -3
    tstop = 5000
    vmean1 = np.load('figHopf_bistab_data/fwdbwdv1/mfield_J-3.1to-3_dt001.npy')
    plot_ramped_mean_field_vs_param(vmean1, tstop, dt, param_start1, param_end1, color='grey')
    plot_bif_curve_J_vs_v(E, 'figHopf_data/matlab_data_v2/branch4_Jsave.mat')

    # Create a right-pointing arrow
    k = 0
    arrow = patches.Arrow(x=arrowxlocal[k], y=arrowylocal[k], dx=arrowlength[k], dy=0, width=arrowwidth[k], color='black')
    # Add the arrow to the plot
    ax_MFrampUp.add_patch(arrow)

    ax_MFrampUp.axvspan(-3.05, -3.03, facecolor=bistab_region_color, alpha=0.3)
    ax_MFrampUp.axvspan(mean_zoom_start, mean_zoom_end, facecolor=slice_color, alpha=0.3)
    plt.xlim([-3.08, -3])
    plt.ylim([0.8, 1.75])

    # 2. Spiking Net osc->non osc
    ax_spikingRampUp = fig.add_subplot(gs_spikingRampUp)
    vpop3 = np.load('figHopf_bistab_data/fwdbwdv1/spiking_J-3.5to-2.5_N2000_dt001.npy')
    param_start3 = -3.5
    param_end3 = -2.5
    plot_ramped_mean_field_vs_param(vpop3, 2000, dt, param_start3, param_end3, color='grey')
    # plot_bif_curve(E, 'figHopf_data/matlab_data_v2/branch4_Jsave.mat')

    #shaded regions
    ax_spikingRampUp.axvspan(-3.267, -3.2, facecolor=bistab_region_color, alpha=0.3)
    ax_spikingRampUp.axvspan(spknet_zoom_start, spknet_zoom_end, facecolor=slice_color, alpha=0.3)

    k = 1
    arrow = patches.Arrow(x=arrowxlocal[k], y=arrowylocal[k], dx=arrowlength[k], dy=0, width=arrowwidth[k],
                          color='black')
    ax_spikingRampUp.add_patch(arrow)

    plt.xlim([-3.4, -3.05])
    plt.ylim([0.7, 1.75])
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Str. ($J_0$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)


    # 3. Mean Field non osc->osc
    ax_MFrampDown = fig.add_subplot(gs_MFrampDown)
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Str. ($J_0$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    param_start2 = -3.04
    param_end2 = -3.1
    tstop = 5000
    vmean2 = np.load('figHopf_bistab_data/fwdbwdv1/mfield_J-3.04to-3.1_dt001.npy')
    plot_ramped_mean_field_vs_param(vmean2, tstop, dt, param_start2, param_end2, color='grey')
    plot_bif_curve_J_vs_v(E, 'figHopf_data/matlab_data_v2/branch4_Jsave.mat')
    ax_MFrampDown.axvspan(-3.05, -3.03, facecolor=bistab_region_color, alpha=0.3)
    ax_MFrampDown.axvspan(mean_zoom_start, mean_zoom_end, facecolor=slice_color, alpha=0.3)

    # Create a left-pointing arrow
    k = 2
    arrow = patches.Arrow(x=arrowxlocal[k], y=arrowylocal[k], dx=arrowlength[k], dy=0, width=arrowwidth[k],
                          color='black')
    ax_MFrampDown.add_patch(arrow)

    plt.xlim([-3.08, -3])
    plt.ylim([0.8, 1.75])

    # 4. Spiking Net non osc->osc
    ax_spikingRampDown = fig.add_subplot(gs_spikingRampDown)
    vpop4 = np.load('figHopf_bistab_data/fwdbwdv1/spiking_J-2.5to-3.5_N2000_dt001.npy')
    param_start4 = -2.5
    param_end4 = -3.5
    plot_ramped_mean_field_vs_param(vpop4, 2000, dt, param_start4, param_end4, color='grey')
    # plot_bif_curve(E, 'figHopf_data/matlab_data_v2/branch4_Jsave.mat')
    ax_spikingRampDown.axvspan(-3.267, -3.2, facecolor=bistab_region_color, alpha=0.3)
    ax_spikingRampDown.axvspan(spknet_zoom_start, spknet_zoom_end, facecolor=slice_color, alpha=0.3)

    #arrow
    k = 3
    arrow = patches.Arrow(x=arrowxlocal[k], y=arrowylocal[k], dx=arrowlength[k], dy=0, width=arrowwidth[k],
                          color='black')
    ax_spikingRampDown.add_patch(arrow)

    plt.xlim([-3.4, -3.05])
    plt.ylim([0.7, 1.75])
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.xlabel(r"Avg. Syn. Str. ($J_0$)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    # 5
    ax_MFosc = fig.add_subplot(gs_MFosc)
    # ax5.get_xaxis().set_visible(False)
    plot_ramped_mean_field_vs_param(vmean1, tstop, dt, param_start1, param_end1, color='grey')
    ax_MFosc.axvspan(-3.05, -3.03, facecolor=slice_color, alpha=0.3)
    ax_MFosc.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax_MFosc.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.xlim([mean_zoom_start, mean_zoom_end])
    plt.ylim([0.8, 1.75])
    plt.xlabel(r'$J_0$', fontsize=fontsize)
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #ax5.set_yticklabels([])
    #plt.ylabel("Voltage (v)")

    # 6
    ax_MFhomg = fig.add_subplot(gs_MFhomg)
    # ax6.get_xaxis().set_visible(False)
    plot_ramped_mean_field_vs_param(vmean2, tstop, dt, param_start2, param_end2, color='grey')
    ax_MFhomg.axvspan(-3.05, -3.03, facecolor=slice_color, alpha=0.3)
    ax_MFhomg.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax_MFhomg.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.xlim([mean_zoom_start, mean_zoom_end])
    plt.ylim([0.8, 1.75])
    plt.xlabel(r'$J_0$', fontsize=fontsize)
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #ax6.set_yticklabels([])

    # 7
    ax_spikingOsc = fig.add_subplot(gs_spikingOsc)
    # ax7.get_xaxis().set_visible(False)
    plot_ramped_mean_field_vs_param(vpop3, 2000, dt, param_start3, param_end3, color='grey')
    ax_spikingOsc.axvspan(-3.26, -3.2, facecolor=slice_color, alpha=0.3)
    ax_spikingOsc.xaxis.set_major_locator(MaxNLocator(nbins=2, prune='both'))
    plt.xlim([spknet_zoom_start, spknet_zoom_end])
    plt.ylim([0.7, 1.75])
    plt.xlabel(r'$J_0$', fontsize=fontsize)
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    # 8
    ax_spikingHomog = fig.add_subplot(gs_spikingHomog)
    # ax8.get_xaxis().set_visible(False)
    plot_ramped_mean_field_vs_param(vpop4, 2000, dt, param_start4, param_end4, color='grey')
    ax_spikingHomog.axvspan(-3.26, -3.2, facecolor=slice_color, alpha=0.3)
    ax_spikingHomog.xaxis.set_major_locator(MaxNLocator(nbins=2, prune='both'))
    plt.xlim([spknet_zoom_start, spknet_zoom_end])
    plt.ylim([0.7, 1.75])
    plt.xlabel(r'$J_0$', fontsize=fontsize)
    plt.ylabel("Voltage (v)", fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ax_spikingBistab = fig.add_subplot(gs_spikingBistab)
    spktimes2 = np.load(
        'figHopf_bistab_data/single_param_bistab_data/spiktimes_J-3.22_E3_D2_N1000_T100_dt.001.npy')
    plot_pop(spktimes2, .001, plotevery=2, ms=0.5)
    plt.xlim([10, 80])
    plt.ylim([0, 1000])
    plt.ylabel("Neuron (i)",fontsize=fontsize)
    plt.xlabel("Time (t)",fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ax_MFbistab = fig.add_subplot(gs_MFbistab)
    t_max = 80
    Nx = 500
    D = 2
    J0 = -3.04
    J_1 = 0
    dt = 0.01
    v0 = (J0 + np.sqrt(J0 ** 2 + 4 * (E - J0))) / 2
    x = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    m_history = np.zeros((int(D / dt), Nx))  # The history
    center = v0
    for tt in range(0, int(D / dt)):
        m_history[tt, :] = center
    m_initial = center

    m = MFsimulation_2_unif_pulses(m_initial, m_history, t_max, Nx, D, dt, J0, J_1, E)
    #plt.imshow(m.T, aspect='auto', cmap='viridis', origin='lower', interpolation='none',extent=[0, len(m) * .01, -np.pi, np.pi])
    plot_spatial_MF(m, dt)
    plt.ylabel('Space (x)', fontsize=fontsize)
    plt.xlabel('Time (t)',fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([10, t_max])

    #####################################
    plt.tight_layout(rect=[0, 0, 1, .98])
    letter_xoffset = -0.07
    letter_yoffset = 0.01
    add_subplot_label(fig, ax_MFrampUp, 'A', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_MFosc, 'B', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikingRampUp, 'E', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikingOsc, 'F', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_MFrampDown, 'C', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_MFhomg, 'D', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikingRampDown, 'G', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikingHomog, 'H', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_MFbistab, 'I', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikingBistab, 'J', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)

    ####
    sns.despine(fig)
    plt.savefig('figs_saved/fig_bistab_v2.png')
    plt.savefig('figs_saved/fig_bistab_v2.pdf')
    plt.show()

    return

if __name__ == '__main__':
    #mean_bistable_data_save()
    build_bistab_fig_v2()
    #build_bistab_fig()
