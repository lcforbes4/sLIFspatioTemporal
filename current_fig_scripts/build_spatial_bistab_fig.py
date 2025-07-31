import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.gridspec import GridSpec
#from calculateMeanFieldProperties import period_varied_J, period_varied_E, period_varied_tau
from functions_and_sims.spiking_network_functions import plot_pop
import seaborn as sns
from functions_and_sims.visualization_functions import add_subplot_label
from build_spatial_fig import detect_upper_fold, detect_lower_fold
from functions_and_sims.plot_bifurcation_curves_and_phase_diag import Turing_phase_diag_zoom

fontsize = 8
labelsize = 6
letterlabelsize =10
linewidth = 1

def plot_E_continuation2(fig,gs,data,maxsteps=200, linewidth = 1):
    linewidth_skinny = linewidth-.2
    #maxsteps = 160
    # Load the .mat file
    #data = loadmat('spatial_fig_data/lif_data.mat')

    # Extract relevant variables from the MATLAB structure
    #l2_grid = data['l2grid'].flatten()  # This is the solution grid for the continuation
    E = data['Egrid'].flatten()
    mingrid = data['mingrid'].flatten()
    maxgrid = data['maxgrid'].flatten()
    J0 = data['J0'].flatten()

    # Add the plots for v = E if E < 1 and v = (J0 + sqrt(J0^2 + 4*(E-J0)))/2 if E > 1
    v_values = np.zeros_like(E[:maxsteps])
    for i in range(maxsteps):
        if E[i] <= 1:
            v_values[i] = E[i]  # If E < 1, v = E
        else:
            v_values[i] = (J0 + np.sqrt(J0 ** 2 + 4 * (E[i] - J0))) / 2  # If E > 1, v as per given formula

    E_for_E_and_v_minus = np.linspace(-10, 1, 500)
    # Plot the new curves
    plt.plot(E_for_E_and_v_minus, E_for_E_and_v_minus, color='m', linewidth=linewidth_skinny)
    plt.plot(E_for_E_and_v_minus, (J0 - np.sqrt(J0 ** 2 + 4 * (E_for_E_and_v_minus - J0))) / 2, linestyle ='--',color='m', linewidth=linewidth_skinny)

    Eturing=E[0]
    if J0 < 2:
        E_for_v_plus_unstab = np.linspace(1, Eturing, 500)
        E_for_v_plus_stab = np.linspace(Eturing, max(E), 500)
    elif J0 >= 2 and J0<5:
        E_for_v_plus_unstab = np.linspace(-10, Eturing, 500)
        E_for_v_plus_stab = np.linspace(Eturing, max(E), 500)
    else:
        E_for_v_plus_unstab = []
        E_for_v_plus_stab = np.linspace(-10, max(E), 500)
    plt.plot(E_for_v_plus_unstab, (J0 + np.sqrt(J0 ** 2 + 4 * (E_for_v_plus_unstab - J0))) / 2, color='m',linestyle='--', linewidth=linewidth_skinny)
    plt.plot(E_for_v_plus_stab, (J0 + np.sqrt(J0 ** 2 + 4 * (E_for_v_plus_stab - J0))) / 2, color='m',linestyle='-', linewidth=linewidth_skinny)

    # Concatenate the point (1,1) to the data
    E_new = np.append(E[:maxsteps], 1)  # Append 1 to the x data
    mingrid_new = np.append(np.real(mingrid[:maxsteps]), 1)  # Append 1 to the y data for mingrid
    maxgrid_new = np.append(np.real(maxgrid[:maxsteps]), 1)  # Append 1 to the y data for maxgrid


    # detect fold points
    upper_idx = detect_upper_fold(E[:maxsteps])
    lower_idx = detect_lower_fold(E[:maxsteps])


    # Plot the lines with the concatenated point (1, 1)
    #plt.plot(E_new, mingrid_new, color='k', linewidth=1)
    #plt.plot(E_new, maxgrid_new, color='k', linewidth=1)
    print(upper_idx)
    if not lower_idx:
        plt.plot(E_new, mingrid_new, color='grey',linestyle='--', linewidth=linewidth)
        plt.plot(E_new, maxgrid_new, 'k--', linewidth=1.2)
    else:
        upper_idx = upper_idx[0]
        lower_idx = lower_idx[0]
        plt.plot(E_new[:upper_idx], mingrid_new[:upper_idx], color='grey',linestyle='--', linewidth=linewidth_skinny)
        plt.plot(E_new[:upper_idx], maxgrid_new[:upper_idx], 'k--', linewidth=linewidth)
        plt.plot(E_new[upper_idx:lower_idx], mingrid_new[upper_idx:lower_idx], color='grey',linestyle='-', linewidth=linewidth_skinny)
        plt.plot(E_new[upper_idx:lower_idx], maxgrid_new[upper_idx:lower_idx], 'k-', linewidth=linewidth)
        plt.plot(E_new[lower_idx:], mingrid_new[lower_idx:], color='grey',linestyle='--', linewidth=linewidth_skinny)
        plt.plot(E_new[lower_idx:], maxgrid_new[lower_idx:], 'k--', linewidth=linewidth)

    #plot instability point
    if J0<5:
        plt.plot(E[0],(J0 + np.sqrt(J0 ** 2 + 4 * (E[0] - J0))) / 2,'ro', ms=3)
    else:
        plt.plot(E[0], (J0 - np.sqrt(J0 ** 2 + 4 * (E[0] - J0))) / 2, 'o', markerfacecolor='none', markeredgecolor='r', markersize=3)

    # plot E
    plt.plot(1, 1, color='orange', marker='o', ms=3)

    #plot fold points
    plt.plot(E[lower_idx], np.real(maxgrid[lower_idx]), 'bo', ms=3)
    plt.plot(E[lower_idx], np.real(mingrid[lower_idx]), 'bo', ms=2)
    plt.plot(E[upper_idx], np.real(maxgrid[upper_idx]), 'go', ms=3)
    plt.plot(E[upper_idx], np.real(mingrid[upper_idx]), 'go', ms=2)
    plt.plot(-(J0 / 2 - 1) ** 2 + 1, J0 / 2, 'ko', ms=3) # saddle

    return

def plot_J0_continuation2(fig,gs,data,maxsteps=200, flip=0, linewidth = 1):
    linewidth_skinny = linewidth - .2
    # this creates a figure
    #maxsteps = 200
    # Load the .mat file
    #data = loadmat('spatial_fig_data/lif_data_J0.mat')

    # Extract relevant variables from the MATLAB structure
    #l2_grid = data['l2grid'].flatten()  # This is the solution grid for the continuation
    E = data['E'].flatten()
    mingrid = data['mingrid'].flatten()
    maxgrid = data['maxgrid'].flatten()
    J0 = data['J0_grid'].flatten()

    if flip == 1:
        E = np.flip(E)  # Reverse E
        mingrid = np.flip(mingrid)  # Reverse mingrid
        maxgrid = np.flip(maxgrid)  # Reverse maxgrid
        J0 = np.flip(J0)  # Reverse J0

    #detect folds
    upper_idx = detect_upper_fold(J0[:maxsteps])
    lower_idx = detect_lower_fold(J0[:maxsteps])
    print(upper_idx)
    print(lower_idx)

    J0turing=J0[0]


    J0_for_v_plus_unstable = np.linspace(-10, J0turing, 500)
    J0_for_v_plus_stable = np.linspace(J0turing,20, 500)

    if E < 1:
        J0_for_v_plus_unstable1 = np.linspace(2 + 2 * np.sqrt(1 - E), J0turing, 500)
        plt.plot(J0_for_v_plus_unstable1,
                 (J0_for_v_plus_unstable1 + np.sqrt(J0_for_v_plus_unstable1 ** 2 + 4 * (E - J0_for_v_plus_unstable1))) / 2,
                 'm', linestyle='--', linewidth=linewidth_skinny)
        plt.plot(J0_for_v_plus_stable,
                 (J0_for_v_plus_stable + np.sqrt(J0_for_v_plus_stable ** 2 + 4 * (E - J0_for_v_plus_stable))) / 2,
                 'm', linestyle='-', linewidth=linewidth_skinny)

        J0_for_v_minus = np.linspace(2 + 2 * np.sqrt(1 - E), 20, 500)
        J0_for_E = np.linspace(-10, 20, 500)
        plt.plot(J0_for_v_minus, (J0_for_v_minus - np.sqrt(J0_for_v_minus ** 2 + 4 * (E - J0_for_v_minus))) / 2,
                 color='m', linestyle='--',linewidth=linewidth)
        plt.plot(J0_for_E, E * np.ones(len(J0_for_v_plus_stable)), 'm', linestyle='-',linewidth=linewidth_skinny)
    elif E == 1:
        J0_for_v_minus = np.linspace(2, 20, 500)
        J0_for_E = np.linspace(-10, 2, 500)

        plt.plot(J0_for_v_minus, (J0_for_v_minus - np.sqrt(J0_for_v_minus ** 2 + 4 * (E - J0_for_v_minus))) / 2,
                 color='m', linestyle='-.', linewidth=linewidth)
        plt.plot(J0_for_E, E * np.ones(len(J0_for_v_plus_stable)), 'm', linestyle='-', linewidth=linewidth_skinny)

        J0_for_v_plus_unstable = np.linspace(2, J0turing, 500)
        plt.plot(J0_for_v_plus_unstable,
                 (J0_for_v_plus_unstable + np.sqrt(J0_for_v_plus_unstable ** 2 + 4 * (E - J0_for_v_plus_unstable))) / 2,
                 'm', linestyle='--', linewidth=linewidth_skinny)
        plt.plot(J0_for_v_plus_stable,
                 (J0_for_v_plus_stable + np.sqrt(J0_for_v_plus_stable ** 2 + 4 * (E - J0_for_v_plus_stable))) / 2, 'm',
                 linestyle='-', linewidth=linewidth_skinny)
    else:
        plt.plot(J0_for_v_plus_unstable, (J0_for_v_plus_unstable + np.sqrt(J0_for_v_plus_unstable ** 2 + 4 * (E - J0_for_v_plus_unstable))) / 2, 'm', linestyle='--', linewidth=linewidth_skinny)
        plt.plot(J0_for_v_plus_stable, (J0_for_v_plus_stable + np.sqrt(J0_for_v_plus_stable ** 2 + 4 * (E - J0_for_v_plus_stable))) / 2, 'm', linestyle='-', linewidth=linewidth_skinny)

    if not upper_idx:
        plt.plot(J0[:maxsteps], np.real(mingrid[:maxsteps]), color='grey',linestyle='--',linewidth=linewidth_skinny)
        plt.plot(J0[:maxsteps], np.real(maxgrid[:maxsteps]), 'k--',linewidth=linewidth)
    elif not lower_idx:
        upper_idx = upper_idx[0]
        plt.plot(J0[:upper_idx], np.real(mingrid[:upper_idx]), color='grey',linestyle='--', linewidth=linewidth_skinny)
        plt.plot(J0[:upper_idx], np.real(maxgrid[:upper_idx]), 'k--', linewidth=linewidth)
        plt.plot(J0[upper_idx:maxsteps], np.real(mingrid[upper_idx:maxsteps]), color='grey',linestyle='-', linewidth=linewidth_skinny)
        plt.plot(J0[upper_idx:maxsteps], np.real(maxgrid[upper_idx:maxsteps]), 'k-', linewidth=linewidth)
    else:
        upper_idx = upper_idx[0]
        lower_idx1 = lower_idx[0]
        plt.plot(J0[:upper_idx], np.real(mingrid[:upper_idx]), color='grey',linestyle='--', linewidth=linewidth_skinny)
        plt.plot(J0[:upper_idx], np.real(maxgrid[:upper_idx]), 'k--', linewidth=linewidth)
        plt.plot(J0[upper_idx:lower_idx1], np.real(mingrid[upper_idx:lower_idx1]), color='grey',linestyle='-', linewidth=linewidth_skinny)
        plt.plot(J0[upper_idx:lower_idx1], np.real(maxgrid[upper_idx:lower_idx1]), 'k-', linewidth=linewidth)
        plt.plot(J0[lower_idx1:maxsteps], np.real(mingrid[lower_idx1:maxsteps]), color='grey',linestyle='--', linewidth=linewidth_skinny)
        plt.plot(J0[lower_idx1:maxsteps], np.real(maxgrid[lower_idx1:maxsteps]), 'k--', linewidth=linewidth)

    # plot E
    if E==1:
        plt.plot(-1.2, 1, color='orange', marker='o', ms=3)


    plt.plot(J0[upper_idx], np.real(maxgrid[upper_idx]), 'go', ms=3)
    plt.plot(J0[upper_idx], np.real(mingrid[upper_idx]), 'go', ms=2)

    plt.plot(J0[lower_idx], np.real(maxgrid[lower_idx]), 'bo', ms=3)
    plt.plot(J0[lower_idx], np.real(mingrid[lower_idx]), 'bo', ms=2)

    plt.plot(2+2*np.sqrt(1-E), 1+1*np.sqrt(1-E), 'ko', ms=3)

    if E>-1.24:
        plt.plot(J0turing, (J0turing + np.sqrt(J0turing ** 2 + 4 * (E - J0turing))) / 2, 'ro', ms=3)
    else:
        plt.plot(J0turing, (J0turing - np.sqrt(J0turing ** 2 + 4 * (E - J0turing))) / 2, 'o', markerfacecolor='none', markeredgecolor='r', markersize=3)


    return


def build_spatial_fig_v4():
    # a second version that focuses on the behavior around the codimension 2 point and regions of bistability
    fig = plt.figure()
    fig.set_size_inches(6.5, 6.5)  # width by height in inches
    gs = GridSpec(5, 4, figure=fig, height_ratios=[2/3, 2/3, 2/3, 1, 1])  # equal row height
    gs_HLraster = gs[0:1, 0]
    gs_QLraster = gs[1:2, 0]
    gs_HQLraster = gs[2:3,0]
    gs_phaseDiag = gs[0:3, 1:4]
    gs_Ediag1 = gs[3:4, 0:1]
    gs_Ediag2 = gs[3:4, 1:2]
    gs_Ediag3 = gs[3:4, 2:3]
    gs_Ediag4 = gs[3:4, 3:4]
    gs_J0diag1 = gs[4:5, 0:1]
    gs_J0diag2 = gs[4:5, 1:2]
    gs_J0diag3 = gs[4:5, 2:3]
    gs_J0diag4 = gs[4:5, 3:4]


    # Plot Phase Diagram (ax)
    ax_phaseDiag = fig.add_subplot(gs_phaseDiag)  # Span 3 rows by 3 columns
    Turing_phase_diag_zoom(labelsize=labelsize, linewidth=linewidth)
    plt.xlabel(r'Mean Coupling Strength ($J_0$)', fontsize=fontsize)
    plt.ylabel(r'Reversal Potential ($E$)', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([2, 6.5])
    plt.ylim([-2.5, 2])


    ax_Ediag1 = fig.add_subplot(gs_Ediag1)
    plt.title(r'$J_0=4$', loc="left", fontsize=fontsize, pad=5)
    data = loadmat('spatial_fig_data/lif_data_cont_in_E_J0_4.00_J1_10.mat')
    plot_E_continuation2(fig, gs, data, maxsteps=150)
    plt.ylim([-3, 4.5])
    plt.xlim([-1, 2])
    plt.xlabel('$E$', fontsize=fontsize)
    plt.ylabel('$v$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ax_Ediag2 = fig.add_subplot(gs_Ediag2)
    plt.title(r'$J_0=5$', loc="left", fontsize=fontsize, pad=5)
    data = loadmat('spatial_fig_data/lif_data_cont_in_E_J0_5.00_J1_10.mat')
    plot_E_continuation2(fig, gs, data, maxsteps=153)
    plt.ylim([-3, 4.5])
    plt.xlim([-1.5, 1.5])
    plt.xlabel('$E$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #plt.ylabel('$v$')
    #ax_Ediag2.set_yticklabels([])

    ax_Ediag3 = fig.add_subplot(gs_Ediag3)
    plt.title(r'$J_0=5.75$', loc="left", fontsize=fontsize, pad=5)
    data = loadmat('spatial_fig_data/lif_data_cont_in_E_J0_5.75_J1_10.mat')
    plot_E_continuation2(fig, gs, data, maxsteps=151)
    plt.ylim([-3, 4.5])
    plt.xlim([-3, 1.5])
    plt.xlabel('$E$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #plt.ylabel('$v$')
    #ax_Ediag3.set_yticklabels([])

    ax_Ediag4 = fig.add_subplot(gs_Ediag4)
    plt.title(r'$J_0=6.5$', loc="left", fontsize=fontsize, pad=5)
    data = loadmat('spatial_fig_data/lif_data_cont_in_E_J0_6.50_J1_10.mat')
    plot_E_continuation2(fig, gs, data, maxsteps=150)
    plt.ylim([-3, 4.5])
    plt.xlim([-5, 1.5])
    plt.xlabel('$E$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #plt.ylabel('$v$')
    #ax_Ediag4.set_yticklabels([])

    ax_J0diag1 = fig.add_subplot(gs_J0diag1)
    plt.title(r'$E=1.5$', loc="left", fontsize=fontsize, pad=0)
    data = loadmat('spatial_fig_data/lif_cont_in_J0_E_1.5_J1_10.mat')
    plot_J0_continuation2(fig, gs, data, maxsteps=150)
    plt.ylim([-2.5, 4])
    plt.xlim([-3, 5])
    plt.xlabel('$J_0$', fontsize=fontsize)
    plt.ylabel('$v$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ax_J0diag2 = fig.add_subplot(gs_J0diag2)
    plt.title(r'$E=1$', loc="left", fontsize=fontsize, pad=0)
    data = loadmat('spatial_fig_data/lif_cont_in_J0_E_1_J1_10.mat')
    plot_J0_continuation2(fig, gs, data, maxsteps=159)
    plt.ylim([-2.5, 4])
    plt.xlim([-1.5, 5])
    plt.xlabel('$J_0$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #plt.ylabel('$v$')
    #ax_J0diag2.set_yticklabels([])

    ax_J0diag3 = fig.add_subplot(gs_J0diag3)
    plt.title(r'$E=0.25$', loc="left", fontsize=fontsize, pad=0)
    data = loadmat('spatial_fig_data/lif_cont_in_J0_E_0.25_J1_10.mat')
    plot_J0_continuation2(fig, gs, data, maxsteps=500)
    plt.ylim([-2.5, 4])
    plt.xlim([2, 6])
    plt.xlabel('$J_0$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #plt.ylabel('$v$')
    #ax_J0diag3.set_yticklabels([])

    ax_J0diag4 = fig.add_subplot(gs_J0diag4)
    plt.title(r'$E=-1.5$',loc="left", fontsize=fontsize,pad=0)
    data = loadmat('spatial_fig_data/lif_data_J0_E_-1.5.mat')
    plot_J0_continuation2(fig, gs, data, maxsteps=120)
    plt.ylim([-3, 4])
    plt.xlim([5, 8])
    plt.xlabel('$J_0$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #plt.ylabel('$v$')
    ax_J0diag4.set_yticklabels([])

    ax_QLraster = fig.add_subplot(gs_QLraster)  # Changed to the 4th column
    plt.text(0, 1.05, '■', transform=ax_QLraster.transAxes, fontsize=fontsize, fontweight='bold', va='bottom', ha='left')
    #vpop2 = np.load('spatial_fig_data/Bistab_QL_spiking_pop_voltages_J03_J10_E.75.npy')
    spktimes2 = np.load('spatial_fig_data/Bistab_QL_spiking_pop_spk_times_J03_J10_E.75.npy')
    tstop = 40
    plot_pop(spktimes2, 0.01, plotevery=5, fontsize=fontsize, ms=0.5)
    plt.xlim([0,tstop])
    plt.ylim([0,1000])
    #ax_QLraster.set_xticklabels([])
    plt.ylabel('Neuron',fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #plt.yticks([])

    ax_HQLraster = fig.add_subplot(gs_HQLraster)  # Changed to the 4th column
    plt.text(0, 1.05, '▲', transform=ax_HQLraster.transAxes, fontsize=fontsize, fontweight='bold', va='bottom', ha='left')
    #vpop2 = np.load('spatial_fig_data/Bistab_HQL_spiking_pop_voltages_J4.2_J10_E.5.npy')
    spktimes2 = np.load('spatial_fig_data/Bistab_HQL_spiking_pop_spk_times_J4.2_J10_E.5.npy')
    tstop = 40
    plot_pop(spktimes2, 0.01, plotevery=5, fontsize=fontsize, ms=0.5)
    plt.xlim([0, tstop])
    plt.ylim([0, 1000])
    #ax_HQLraster.set_xticklabels([])
    plt.ylabel('Neuron',fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #plt.yticks([])

    ax_HLraster = fig.add_subplot(gs_HLraster)  # Moved ax9 to row 2, col 4
    plt.text(0, 1.05, '●', transform=ax_HLraster.transAxes, fontsize=fontsize, fontweight='bold', va='bottom', ha='left')
    vpop2 = np.load('spatial_fig_data/Bistab_HL_spiking_pop_voltages_J3.1_J10_E1.75.npy')
    spktimes2 = np.load('spatial_fig_data/Bistab_HL_spiking_pop_spk_times_J3.1_J10_E1.75.npy')
    tstop = 40
    plot_pop(spktimes2, 0.01, plotevery=5, fontsize=fontsize, ms=0.5)
    plt.xlim([0, tstop])
    plt.ylim([0, 1000])
    #plt.yticks([])
    plt.xlabel('Time ($t$)', fontsize=fontsize)
    plt.ylabel('Neuron',fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    #################

    plt.tight_layout(rect=[0, 0, .98, 0.98])  # change to tight layout before adding labels
    letter_xoffset = -0.07
    letter_yoffset = 0.01
    add_subplot_label(fig, ax_HLraster, 'A', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_HQLraster, 'C', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_QLraster, 'B', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_phaseDiag, 'D', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_Ediag1, 'E', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_Ediag2, 'F', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_Ediag3, 'G', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_Ediag4, 'H', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_J0diag1, 'I', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_J0diag2, 'J', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_J0diag3, 'K', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_J0diag4, 'L', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)


    #########################################################################################
    sns.despine(fig)
    plt.savefig('figs_saved/figspatial_v4.pdf')
    plt.show()

    return


if __name__=='__main__':
    build_spatial_fig_v4()
