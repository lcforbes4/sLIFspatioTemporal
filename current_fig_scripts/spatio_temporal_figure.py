import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from functions_and_sims.plot_bifurcation_curves_and_phase_diag import plot_delta_instabilties_E_lessthan_1, plot_delta_instabilties_E_gthan_1
from sympy import symbols, lambdify, acos, sqrt
from functions_and_sims.visualization_functions import add_subplot_label
from functions_and_sims.mean_field_equation_functions import plot_spatial_MF
from functions_and_sims.spiking_network_functions import plot_pop

fontsize = 8
labelsize = 6
letterlabelsize =10
def plot_turing_hopf_in_E_J0_varying_J1(J1_vals,ylim=None, linewidth = 1):
    # this plot the multiple Hopf Curves

    # Define the symbolic variables
    J_var, e = symbols('J e')
    dd = 1

    # Define constants
    #J1_vals = [0.2, 0.5, 1]
    #cmap3 = cm.get_cmap('tab20b')
    #d_colors = [cmap3(3/20), cmap3(2/20),cmap3(1/20)]#['#0000FF', '#3399FF', '#66CCFF']#['lightgrey', 'darkgrey', 'black']
    d_colors = ['k', 'k', 'k','k']
    # Create a list to hold legend handles
    legend_handles = []

    label_x_loc = [-6, -5, 2.5,7]
    label_y_loc = [3, 6.5, 5.7,5.5]
    rotation_loc=[-10,-35,-65,-80]

    for i in range(0, len(J1_vals)):
        J1 = J1_vals[i]

        # Define the function v
        vplus = (J_var + sqrt(J_var ** 2 + 4 * (e - J_var))) / 2

        # Define the implicit equation
        implicit_eq = (-1 / dd) * acos(4 * vplus / J1) - (J1 / 2) * sqrt(1 - (4 * vplus / J1) ** 2)

        # Define the function v
        vminus = (J_var - sqrt(J_var ** 2 + 4 * (e - J_var))) / 2

        # Define the implicit equation
        implicit_eq2 = (-1 / dd) * acos(4 * vminus / J1) - (J1 / 2) * sqrt(1 - (4 * vminus / J1) ** 2)

        # Convert the implicit equation to a numerical function
        implicit_function = lambdify((J_var, e), implicit_eq, 'numpy')
        implicit_function2 = lambdify((J_var, e), implicit_eq2, 'numpy')

        # Create a grid of J and e values
        J_vals = np.linspace(-10, 10, 500)
        e_vals = np.linspace(-10, 10, 1000)
        J_grid, e_grid = np.meshgrid(J_vals, e_vals)

        # Compute the implicit function on the grid
        F_grid = implicit_function(J_grid, e_grid)
        F_grid2 = implicit_function2(J_grid, e_grid)

        # Plotting the contour
        contour = plt.contour(J_grid, e_grid, F_grid, levels=[0], colors=[d_colors[i]], linewidths = linewidth)
        contour2 = plt.contour(J_grid, e_grid, F_grid2, levels=[0], linestyles='dashed', colors=[d_colors[i]], linewidths = linewidth)

        plt.text(label_x_loc[i], label_y_loc[i], r'$J_1={}$'.format(J1_vals[i]), rotation=rotation_loc[i], fontsize=5, ha='center',va='center')

    # Add a handle for the legend using a proxy artist
        # Add labels along the contour line
        #plt.clabel(contour, inline=False, fontsize=10, fmt=f'$D={dd}$',colors='black', inline_spacing=30)
        #legend_handles.append(plt.Line2D([0], [0], color=d_colors[i], label=f'$D={dd}$'))


    plt.plot([-10, 10], [1, 1], 'b')
    j_vals2 = np.linspace(2, 10, 20)
    plt.plot(j_vals2, 1 - ((j_vals2 - 2) / 2) ** 2, 'b')

    if ylim is not None:
        plt.ylim(ylim)

    plt.ylabel('$E$', fontsize=fontsize)
    plt.xlabel('$J_0$',fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    # Customize legend: remove border and add labels
    #plt.legend(handles=legend_handles + [plt.Line2D([0], [0], color='#0000FF', linestyle='--', label=r'$D \rightarrow \infty$')],
    #           frameon=False, fontsize='small',loc='lower left')

    return


def plot_turing_hopf_in_E_J0_varying_D(D_vals,ylim=None, linewidth = 1):
    # this plot the multiple Hopf Curves

    # Define the symbolic variables
    J_var, e = symbols('J e')
    J1 = -12

    # Define constants
    #J1_vals = [0.2, 0.5, 1]
    #cmap3 = cm.get_cmap('tab20b')
    #d_colors = [cmap3(3/20), cmap3(2/20),cmap3(1/20)]#['#0000FF', '#3399FF', '#66CCFF']#['lightgrey', 'darkgrey', 'black']
    d_colors = ['k', 'k', 'k']
    # Create a list to hold legend handles
    legend_handles = []

    label_x_loc = [-6.5, -5, -.8]
    label_y_loc = [2.8, 6, 7.5]
    rotation_loc = [-10, -30, -58]

    for i in range(0, len(D_vals)):
        dd = D_vals[i]

        # Define the function v
        vplus = (J_var + sqrt(J_var ** 2 + 4 * (e - J_var))) / 2

        # Define the implicit equation
        implicit_eq = (-1 / dd) * acos(4 * vplus / J1) - (J1/2) * sqrt(1 - (4* vplus / J1) ** 2)

        # Define the function v
        vminus = (J_var - sqrt(J_var ** 2 + 4 * (e - J_var))) / 2

        # Define the implicit equation
        implicit_eq2 = (-1 / dd) * acos(4 * vminus / J1) - (J1 / 2) * sqrt(1 - (4 * vminus / J1) ** 2)
        implicit_limit = 4*vplus/J1 +1

        # Convert the implicit equation to a numerical function
        implicit_function = lambdify((J_var, e), implicit_eq, 'numpy')
        implicit_function2 = lambdify((J_var, e), implicit_eq2, 'numpy')
        implicit_function3 = lambdify((J_var, e), implicit_limit, 'numpy')

        # Create a grid of J and e values
        J_vals = np.linspace(-10, 10, 500)
        e_vals = np.linspace(-10, 10, 500)
        J_grid, e_grid = np.meshgrid(J_vals, e_vals)

        # Compute the implicit function on the grid
        F_grid = implicit_function(J_grid, e_grid)
        F_grid2 = implicit_function2(J_grid, e_grid)
        F_grid3 = implicit_function3(J_grid, e_grid)

        # Plotting the contour
        contour = plt.contour(J_grid, e_grid, F_grid, levels=[0], colors=[d_colors[i]], linewidths=linewidth)
        contour2 = plt.contour(J_grid, e_grid, F_grid2, levels=[0], linestyles = 'dashed', colors=[d_colors[i]], linewidths=linewidth)
        contour2 = plt.contour(J_grid, e_grid, F_grid3, levels=[0], linestyles='dotted', colors='k', linewidths=linewidth)

        plt.text(label_x_loc[i], label_y_loc[i], r'$D={}$'.format(D_vals[i]), rotation=rotation_loc[i], fontsize=5,ha='center',va='center')


        # Add a handle for the legend using a proxy artist
        # Add labels along the contour line
        #plt.clabel(contour, inline=False, fontsize=10, fmt=f'$D={dd}$',colors='black', inline_spacing=30)
        #legend_handles.append(plt.Line2D([0], [0], color=d_colors[i], label=f'$D={dd}$'))


    plt.plot([-10, 10], [1, 1], 'b')
    j_vals2 = np.linspace(2, 10, 20)
    plt.plot(j_vals2, 1 - ((j_vals2 - 2) / 2) ** 2, 'b')

    plt.text(1.75, 7.5, r'$D\gg 1$', rotation=-60, fontsize=5,
             ha='center',
             va='center')

    if ylim is not None:
        plt.ylim(ylim)

    plt.ylabel('$E$',fontsize=fontsize)
    plt.xlabel('$J_0$',fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    # Customize legend: remove border and add labels
    #plt.legend(handles=legend_handles + [plt.Line2D([0], [0], color='#0000FF', linestyle='--', label=r'$D \rightarrow \infty$')],
    #           frameon=False, fontsize='small',loc='lower left')

    return



def build_spatio_tempo_fig():
    fig = plt.figure()
    fig.set_size_inches(6.5, 8)  # width by height in inches

    ## Setting locations in the grid:
    gs = GridSpec(7, 4, figure=fig)
    #gs_instab_above_thresh = gs[0, 0]
    #gs_instab_below_thresh = gs[0, 1]
    #gs_THcurve_vary_J1 = gs[0, 2]
    #gs_THcurve_vary_D = gs[0, 3]

    gs_traveling_wave = gs[0, 1]
    gs_standing_wave = gs[0, 0]
    gs_SW_TW_spike = gs[0,2:4]


    gs_osc_bumpMF = gs[1, :2]
    gs_osc_bumpspiking = gs[1, 2:4]

    gs_zigzag_MF = gs[2, :2]
    gs_zigzag_spiking = gs[2, 2:4]

    gs_spotsMF = gs[5, 0:2]
    gs_spotsspiking = gs[5, 2:4]

    gs_lurching_MF = gs[3, 0:2]
    gs_lurching_spiking = gs[3, 2:4]

    gs_chaoticzigzagMF = gs[6, 0:2]
    gs_chaoticzigzagspiking = gs[6, 2:4]

    gs_travelingMixedSW1MF = gs[4, :2]
    gs_travelingMixedSW1spiking = gs[4, 2:4]



    ## Plotting:


    markersz = 0.2

    ax_traveling_wave = fig.add_subplot(gs_traveling_wave)
    m = np.load('spiking_network_patterns_fig/MF_TW_D_1_E_2_J0=-2, J1=-8, dt=0.005, N=1000.npy')
    plot_spatial_MF(m,.005,xlim=[60,70], fontsize=fontsize, labelsize=labelsize)
    ax_traveling_wave.set_xlabel('')
    ax_traveling_wave.set_ylabel('')
    ax_traveling_wave.set_xticklabels([])
    ax_traveling_wave.set_yticklabels([])

    ax_standing_wave = fig.add_subplot(gs_standing_wave)
    m = np.load('spiking_network_patterns_fig/MF_SW_D_1_E_2_J0=-2, J1=-8, dt=0.005, N=1000.npy')
    plot_spatial_MF(m,.005,xlim=[5,15], fontsize=fontsize, labelsize=labelsize)
    ax_standing_wave.set_xlabel('')
    ax_standing_wave.set_yticklabels([])
    ax_standing_wave.set_xticklabels([])

    ax_SW_TW_spike = fig.add_subplot(gs_SW_TW_spike)
    spktimes = np.load('spiking_network_patterns_fig/spktimes_SW,D=1,E=2,J0=-2, J1=-8, dt=0.005, N=1000.npy')
    N = 1000
    dt = .005
    plot_pop(spktimes, dt, xlim=[0, 20], ylim=[0, N], fontsize=fontsize, labelsize=labelsize)
    ax_SW_TW_spike.set_xlabel('')
    ax_SW_TW_spike.set_ylabel('')
    ax_SW_TW_spike.set_xticklabels([])
    ax_SW_TW_spike.set_yticklabels([])


    ax_osc_bumpMF = fig.add_subplot(gs_osc_bumpMF)
    m = np.load('spiking_network_patterns_fig/MF_breather_D_0.2_E_10_J0=-60, J1=20, dt=0.005, N=1000.npy')
    plot_spatial_MF(m, .005, xlim=[20, 40], fontsize=fontsize, labelsize=labelsize)
    ax_osc_bumpMF.set_xlabel('')
    #ax_osc_bumpMF.set_ylabel('')
    ax_osc_bumpMF.set_xticklabels([])
    ax_osc_bumpMF.set_yticklabels([])

    ax_osc_bumpSpiking = fig.add_subplot(gs_osc_bumpspiking)
    spiktimes = np.load(
        'spiking_network_patterns_fig/spktimes_breather,D=0.2,E=10,J0=-60, J1=20, dt=0.005, N=1000.npy')
    plot_pop(spiktimes, .005, ms=markersz, fontsize=fontsize, labelsize=labelsize)
    plt.xlim([20, 40])
    plt.ylim([0, 1000])
    ax_osc_bumpSpiking.set_xlabel('')
    ax_osc_bumpSpiking.set_ylabel('')
    ax_osc_bumpSpiking.set_xticklabels([])
    ax_osc_bumpSpiking.set_yticklabels([])

    ax_chaoticzigzagMF = fig.add_subplot(gs_chaoticzigzagMF)
    m = np.load('spiking_network_patterns_fig/MF_chaoticzigzag_D_0.2_E_10_J0=-5, J1=-90, dt=0.005, N=1000.npy')
    plot_spatial_MF(m, .005, xlim=[135, 175], fontsize=fontsize, labelsize=labelsize)
    #ax_chaoticzigzagMF.set_xlabel('')
    #ax_chaoticzigzagMF.set_ylabel('')
    ax_chaoticzigzagMF.set_xticklabels([])
    ax_chaoticzigzagMF.set_yticklabels([])

    ax__chaoticzigzagspiking = fig.add_subplot(gs_chaoticzigzagspiking)
    spiktimes = np.load(
        'spiking_network_patterns_fig/spktimes_chaoticzigzag,D=0.2,E=10,J0=-5, J1=-90, dt=0.005, N=1000.npy')
    plot_pop(spiktimes, .005, ms=markersz, fontsize=fontsize, labelsize=labelsize)
    plt.xlim([140, 180])
    plt.ylim([0, 1000])
    #ax__chaoticzigzagspiking.set_xlabel('')
    ax__chaoticzigzagspiking.set_ylabel('')
    ax__chaoticzigzagspiking.set_xticklabels([])
    ax__chaoticzigzagspiking.set_yticklabels([])

    ax_spotsMF = fig.add_subplot(gs_spotsMF)
    m = np.load('spiking_network_patterns_fig/MF_spots_D_0.2_E_10_J0=-5, J1=-100, dt=0.005, N=1000.npy')
    plot_spatial_MF(m, .005, xlim=[210, 240], fontsize=fontsize, labelsize=labelsize)
    ax_spotsMF.set_xlabel('')
    #ax_spotsMF.set_ylabel('')
    ax_spotsMF.set_xticklabels([])
    ax_spotsMF.set_yticklabels([])

    ax_spotsSpiking = fig.add_subplot(gs_spotsspiking)
    spiktimes = np.load(
        'spiking_network_patterns_fig/spktimes_spots,D=0.2,E=10,J0=-5, J1=-100, dt=0.005, N=1000.npy')
    plot_pop(spiktimes, .005, ms=markersz, fontsize=fontsize, labelsize=labelsize)
    plt.xlim([210, 240])
    plt.ylim([0, 1000])
    ax_spotsSpiking.set_xlabel('')
    ax_spotsSpiking.set_ylabel('')
    ax_spotsSpiking.set_xticklabels([])
    ax_spotsSpiking.set_yticklabels([])

    '''ax_weirdMF = fig.add_subplot(gs_weirdMF)
    m = np.load('../spiking_network_patterns_fig/MF_weird_D_0.2_E_10_J0=-15, J1=-100, dt=0.005, N=1000.npy')
    plot_spatial_MF(m, .005, xlim=[190, 230], fontsize=fontsize, labelsize=labelsize)

    ax_weirdSpiking = fig.add_subplot(gs_weirdspiking)
    spiktimes = np.load(
        '../spiking_network_patterns_fig/spktimes_weird,D=0.2,E=10,J0=-15, J1=-100, dt=0.005, N=1000.npy')
    plot_pop(spiktimes, .005, ms=markersz)
    plt.xlim([190, 230])
    plt.ylim([0, 1000])'''

    '''ax_TSW_MF = fig.add_subplot(gs_TSWMF)
    m = np.load('../spiking_network_patterns_fig/MF_travelingSW_D_0.2_E_10_J0=-25, J1=-75, dt=0.005, N=1000.npy')
    plot_spatial_MF(m, .005, xlim=[200, 250], fontsize=fontsize, labelsize=labelsize)

    ax_TSW_Spiking = fig.add_subplot(gs_TSWspiking)
    spiktimes = np.load(
        '../spiking_network_patterns_fig/spktimes_travelingSW,D=0.2,E=10,J0=-25, J1=-75, dt=0.005, N=1000.npy')
    plot_pop(spiktimes, .005, ms=markersz)
    plt.xlim([200, 250])
    plt.ylim([0, 1000])'''

    ax_zigzag_MF = fig.add_subplot(gs_zigzag_MF)
    m = np.load('spiking_network_patterns_fig/MF_zigzag_D_0.2_E_10_J0=-17, J1=-60, dt=0.005, N=1000.npy')
    plot_spatial_MF(m, .005, xlim=[150, 180], fontsize=fontsize, labelsize=labelsize)
    ax_zigzag_MF.set_xlabel('')
    #ax_zigzag_MF.set_ylabel('')
    ax_zigzag_MF.set_xticklabels([])
    ax_zigzag_MF.set_yticklabels([])

    ax_zigzag_Spiking = fig.add_subplot(gs_zigzag_spiking)
    spiktimes = np.load(
        'spiking_network_patterns_fig/spktimes_zigzag,D=0.2,E=10,J0=-17, J1=-60, dt=0.005, N=1000.npy')
    plot_pop(spiktimes, .005, ms=markersz, fontsize=fontsize, labelsize=labelsize)
    plt.xlim([150, 180])
    plt.ylim([0, 1000])
    ax_zigzag_Spiking.set_xlabel('')
    ax_zigzag_Spiking.set_ylabel('')
    ax_zigzag_Spiking.set_xticklabels([])
    ax_zigzag_Spiking.set_yticklabels([])

    ax_lurching_MF = fig.add_subplot(gs_lurching_MF)
    m = np.load('spiking_network_patterns_fig/MF_lurching_D_0.2_E_10_J0=-16, J1=-72, dt=0.005, N=1000.npy')
    plot_spatial_MF(m, .005, xlim=[130, 180], fontsize=fontsize, labelsize=labelsize)
    ax_lurching_MF.set_xlabel('')
    ax_lurching_MF.set_xticklabels([])
    ax_lurching_MF.set_yticklabels([])


    ax_lurching_Spiking = fig.add_subplot(gs_lurching_spiking)
    spiktimes = np.load(
        'spiking_network_patterns_fig/spktimes_lurching,D=0.2,E=10,J0=-16, J1=-72, dt=0.005, N=1000.npy')
    plot_pop(spiktimes, .005, ms=markersz, fontsize=fontsize, labelsize=labelsize)
    plt.xlim([130, 180])
    plt.ylim([0, 1000])
    ax_lurching_Spiking.set_xlabel('')
    ax_lurching_Spiking.set_ylabel('')
    ax_lurching_Spiking.set_xticklabels([])
    ax_lurching_Spiking.set_yticklabels([])

    ax_travelingMixedSW1_MF = fig.add_subplot(gs_travelingMixedSW1MF)
    m = np.load('spiking_network_patterns_fig/MF_mixedSW1moving_D_0.2_E_10_J0=-25, J1=-95, dt=0.005, N=1000.npy')
    plot_spatial_MF(m, .005, xlim=[200, 220], fontsize=fontsize, labelsize=labelsize)
    ax_travelingMixedSW1_MF.set_xlabel('')
    ax_travelingMixedSW1_MF.set_xticklabels([])
    ax_travelingMixedSW1_MF.set_yticklabels([])

    ax_travelingMixedSW1_Spiking = fig.add_subplot(gs_travelingMixedSW1spiking)
    spiktimes = np.load(
        'spiking_network_patterns_fig/spktimes_mixedSW1moving,D=0.2,E=10,J0=-25, J1=-95, dt=0.005, N=1000.npy')
    plot_pop(spiktimes, .005, ms=markersz, fontsize=fontsize, labelsize=labelsize)
    plt.xlim([200, 220])
    plt.ylim([0, 1000])
    ax_travelingMixedSW1_Spiking.set_xlabel('')
    ax_travelingMixedSW1_Spiking.set_ylabel('')
    ax_travelingMixedSW1_Spiking.set_xticklabels([])
    ax_travelingMixedSW1_Spiking.set_yticklabels([])

    '''ax_mixedSW_MF = fig.add_subplot(gs_mixedSW_MF)
    m = np.load('../spiking_network_patterns_fig/MF_mixedSW1_D_0.2_E_10_J0=-20, J1=-90, dt=0.005, N=1000.npy')
    plot_spatial_MF(m, .005, xlim=[220, 250], fontsize=fontsize, labelsize=labelsize)

    ax_mixedSW_Spiking = fig.add_subplot(gs_mixedSW_spiking)
    spiktimes = np.load(
        '../spiking_network_patterns_fig/spktimes_mixedSW1,D=0.2,E=10,J0=-20, J1=-90, dt=0.005, N=1000.npy')
    plot_pop(spiktimes, .005, ms=markersz)
    plt.xlim([220, 250])
    plt.ylim([0, 1000])'''




    ## Lettering the plots
    plt.subplots_adjust(hspace=0.2, wspace=0.3, left=0.05, right=0.95, top=0.8, bottom=0.05)

    ##
    bottom = .87
    width = .15
    height = .1

    ax_instab_above_thresh = fig.add_axes([.1, bottom, width, height])  # fig.add_subplot(gs_instab_above_thresh)
    plot_delta_instabilties_E_gthan_1(fontsize=fontsize, labelsize=labelsize)

    ax_instab_below_thresh = fig.add_axes([.325, bottom, width, height])  # fig.add_subplot(gs_instab_below_thresh)
    plot_delta_instabilties_E_lessthan_1(fontsize=fontsize, labelsize=labelsize)

    ax_THcurve_vary_J1 = fig.add_axes([.55, bottom, width, height])  # fig.add_subplot(gs_THcurve_vary_J1)
    J1_vals = [-6.5, -8, -12, -25]
    plot_turing_hopf_in_E_J0_varying_J1(J1_vals, ylim=[-10, 10])

    ax_THcurve_vary_D = fig.add_axes([.8, bottom, width, height])  # fig.add_subplot(gs_THcurve_vary_D)
    D_vals = [0.35, .4, .65]
    plot_turing_hopf_in_E_J0_varying_D(D_vals, ylim=[-10, 10])



    #plt.tight_layout(rect=[.05,.05,.05,.7]) #change to tight layout before adding labels
    letter_xoffset = -0.04
    letter_yoffset = -.01

    letter_xoffsetB = -0.06
    letter_yoffsetB = 0

    add_subplot_label(fig, ax_instab_above_thresh, 'A',x_offset=letter_xoffsetB, y_offset= letter_yoffsetB, fontsize=letterlabelsize)
    add_subplot_label(fig, ax_instab_below_thresh, 'B', x_offset=letter_xoffsetB, y_offset=letter_yoffsetB, fontsize=letterlabelsize)
    add_subplot_label(fig, ax_THcurve_vary_J1, 'C', x_offset=letter_xoffsetB, y_offset=letter_yoffsetB, fontsize=letterlabelsize)
    add_subplot_label(fig, ax_THcurve_vary_D, 'D', x_offset=letter_xoffsetB, y_offset=letter_yoffsetB, fontsize=letterlabelsize)
    add_subplot_label(fig, ax_traveling_wave, 'F', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_standing_wave, 'E', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_SW_TW_spike, 'G', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_osc_bumpMF, 'H', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_osc_bumpSpiking, 'I', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_zigzag_MF, 'J', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_zigzag_Spiking, 'K', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_lurching_MF, 'L', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_lurching_Spiking, 'M', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_travelingMixedSW1_MF, 'N', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_travelingMixedSW1_Spiking, 'O', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spotsMF, 'P', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spotsSpiking, 'Q', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_chaoticzigzagMF, 'R', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax__chaoticzigzagspiking, 'S', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)



    #add_subplot_label(fig, ax_osc_bump, 'D', x_offset=letter_xoffset, y_offset=letter_yoffset, fontsize=letterlabelsize)

    #############
    sns.despine(fig)
    plt.savefig('figs_saved/spatio_temporal_fig.png', dpi=600)
    plt.savefig('figs_saved/spatio_temporal_fig.pdf')

    plt.show()
    return


if __name__ == '__main__':
    build_spatio_tempo_fig()

