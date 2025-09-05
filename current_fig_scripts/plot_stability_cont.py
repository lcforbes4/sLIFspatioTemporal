import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from build_spatial_fig import detect_upper_fold, detect_lower_fold
from matplotlib.gridspec import GridSpec


def plot_evals():
    fig = plt.figure(layout='constrained')
    gs = GridSpec(2, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])

    maxsteps = 621
    # Load the .mat file
    data = loadmat('spatial_fig_data/eigen_data.mat')

    # Extract relevant variables from the MATLAB structure
    l2_grid = data['l2grid'].flatten()  # This is the solution grid for the continuation
    E = data['Egrid'].flatten()
    mingrid = data['mingrid'].flatten()
    maxgrid = data['maxgrid'].flatten()
    J0 = data['J0'].flatten()
    print(J0)
    eval_grid = data['eval_grid'].flatten()
    eval1_grid = data['eval1_grid'].flatten()
    eval2_grid = data['eval2_grid'].flatten()

    Efoldidx = detect_upper_fold(E)
    Efoldidx_lower = detect_lower_fold(E)
    Efoldidx=Efoldidx[0]
    Efoldidx_lower = Efoldidx_lower[0]
    Eturing = E[0]

    plt.plot(E[:Efoldidx], np.real(mingrid[:Efoldidx]), color='k', linestyle='--')
    plt.plot(E[:Efoldidx], np.real(maxgrid[:Efoldidx]), color='k', linestyle='--')
    plt.plot(E[Efoldidx:Efoldidx_lower], np.real(mingrid[Efoldidx:Efoldidx_lower]), color='k', label='Bump')
    plt.plot(E[Efoldidx:Efoldidx_lower], np.real(maxgrid[Efoldidx:Efoldidx_lower]), color='k')
    plt.plot(E[Efoldidx_lower:maxsteps], np.real(mingrid[Efoldidx_lower:maxsteps]), color='k', linestyle='--')
    plt.plot(E[Efoldidx_lower:maxsteps], np.real(maxgrid[Efoldidx_lower:maxsteps]), color='k', linestyle='--')
    # Add the plots for v = E if E < 1 and v = (J0 + sqrt(J0^2 + 4*(E-J0)))/2 if E > 1
    v_values = np.zeros_like(E[:maxsteps])
    for i in range(maxsteps):
        if E[i] <= 1:
            v_values[i] = E[i]  # If E < 1, v = E
        else:
            temp = E[i] - J0
            v_values[i] = (J0 + np.sqrt(J0 ** 2 + 4 * (temp))) / 2  # If E > 1, v as per given formula




    E_for_E_and_v_minus = np.linspace(min(E), 1, 500)
    # Plot the new curves
    plt.plot(E_for_E_and_v_minus, E_for_E_and_v_minus, color='m')
    plt.plot(E_for_E_and_v_minus, (J0 - np.sqrt(J0 ** 2 + 4 * (E_for_E_and_v_minus - J0))) / 2, color='m', linestyle='--')

    if J0 < 2:
        E_for_v_plus_unstab = np.linspace(1, Eturing, 500)
        E_for_v_plus_stab = np.linspace(Eturing, max(E), 500)
    if J0 >= 2:
        E_for_v_plus_unstab = np.linspace(min(E), Eturing, 500)
        E_for_v_plus_stab = np.linspace(Eturing, max(E), 500)
    plt.plot(E_for_v_plus_stab, (J0 + np.sqrt(J0 ** 2 + 4 * (E_for_v_plus_stab - J0))) / 2, color='m',label="Homogeneous")
    plt.plot(E_for_v_plus_unstab, (J0 + np.sqrt(J0 ** 2 + 4 * (E_for_v_plus_unstab - J0))) / 2, color='m', linestyle='--')
    plt.annotate('Start of Continuation', xy=(1.75, 2.4), xytext=(1.75, 1), fontsize=10,
                 ha='center', va='center', color='k',
                 arrowprops=dict(facecolor='orange', edgecolor='k', arrowstyle='->'))
    plt.annotate('End of Continuation', xy=(1, 1), xytext=(1, -0.75), fontsize=10,
                 ha='center', va='center', color='k',
                 arrowprops=dict(facecolor='orange', edgecolor='k', arrowstyle='->'))

    plt.ylabel(r'Voltage (v)')
    plt.xlabel(r'$E$')
    plt.legend(loc='lower right', fontsize=8, frameon=False)
    plt.ylim([-3, 4])
    plt.xlim([0, 3])

    ax2 = fig.add_subplot(gs[1, 0])
    evalmax2 = 2484
    step_array2 = np.linspace(0, evalmax2, evalmax2)
    plt.plot(step_array2, eval_grid[:evalmax2], 'k.',markersize=1)
    plt.annotate('Upper Fold, E=2.90', xy=(510, 0.1), xytext=(780, 1.2), fontsize=10,
                 ha='center', va='center', color='k',
                 arrowprops=dict(facecolor='orange', edgecolor='k', arrowstyle='->'))
    plt.annotate('Lower Fold, E=0.22', xy=(1785, 0), xytext=(1680, 1), fontsize=10,
                 ha='center', va='center', color='k',
                 arrowprops=dict(facecolor='orange', edgecolor='k', arrowstyle='->'))
    plt.ylim([-5, 2])
    plt.xlim([0, evalmax2])
    plt.ylabel(r'$Re(\lambda)$')
    plt.xlabel(r'Continuation Step')

    plt.savefig('figs_saved/tracking_evals.png', dpi=fig.dpi)
    plt.show()
    return

if __name__ == '__main__':
    plot_evals()
