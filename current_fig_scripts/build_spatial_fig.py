import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.gridspec import GridSpec
from functions_and_sims.mean_field_equation_functions import plot_spatial_MF, MF_spatial_sim, MFsim_with_con_mat
#from calculateMeanFieldProperties import period_varied_J, period_varied_E, period_varied_tau
from functions_and_sims.spiking_network_functions import neuron_population, plot_pop
import seaborn as sns
from functions_and_sims.plot_bifurcation_curves_and_phase_diag import plot_turing_curve_varied_J1, Turing_phase_diag_colors
from matplotlib.animation import FuncAnimation
from functions_and_sims.visualization_functions import add_subplot_label
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fontsize = 8
labelsize = 6
letterlabelsize =10
linewidth = 1


def plot_slice_of_mean_vs_spike_anim(m, dt, t_max, tauD, N, J0, J1, E):
    # m is mean field passed in
    # Time and space parameters
    time_steps = np.arange(0, t_max, dt)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a plot line for the mean field and spike plot (initially empty)
    mean_line, = ax.plot([], [], label='Mean Field', color='blue')
    spike_line, = ax.plot([], [], 'k.', label='Spiking Activity')
    # Add text object for time annotation (initially empty)
    time_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top', fontsize=12, color='black')

    # Setting up the axis limits and labels
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-15, 10)
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')

    # Initialize the population spike values
    initial = np.zeros((int(tauD / dt), N))
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)

    for tt in range(0, int(tauD / dt)):
        initial[tt, :] = 0 + 8 * np.cos(x + np.pi / 2)
    # Run the spiking simulation for the current time t
    vpop, spktimes2, J = neuron_population(1, t_max, dt, N, 0, E, tauD, initial, J0=J0 * (2 * np.pi), J1=J1)

    # Define the update function for the animation
    def update(t):
        # Extract the data at time t from the mean field m
        time_index = int(t / dt)
        space_values_at_t = m[time_index, :]  # Assuming rows are time and columns are space
        xmean = np.linspace(-np.pi, np.pi, len(space_values_at_t), endpoint=False)

        # Update the mean field plot
        mean_line.set_data(xmean, space_values_at_t)

        # Update the spike plot (using the population data at time t)
        spike_line.set_data(x, vpop[time_index, :])

        # Update the time annotation text
        time_text.set_text(f't = {t:.2f} s')

        return mean_line, spike_line, time_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=time_steps, interval=dt * 1000, blit=True)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Optionally, save the animation
    #ani.save('mean_vs_spike_movie.mp4', writer='ffmpeg', fps=30)

    return vpop, spktimes2, J

def plot_slice_of_mean_vs_spike(m, dt, t_max, tauD, N, J0, J1, E, linewidth = 1, fontsize=10, labelsize = 10):
    #m is mean field passed in

    # Time and space parameters
    time_at_10 = t_max-1  # We're interested in t = 10
    # Find the index for t = 10
    time_index_10 = int(time_at_10 / dt)
    # Extract the data at t = 10 (this will give us the corresponding space values)
    space_values_at_10 = m[time_index_10, :]  # Assuming rows represent time, and columns represent space
    xmean = np.linspace(-np.pi, np.pi, len(space_values_at_10), endpoint=False)
    #plt.plot(xmean, space_values_at_10)


    # Run spiking sim and plot
    initial = np.zeros((int(tauD / dt), N))
    v0 = (J0 + np.sqrt(J0 ** 2 + 4 * (E - J0))) / 2
    center = 0
    pertstrength = 2
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    for tt in range(0, int(tauD / dt)):
        initial[tt, :] = center + pertstrength * np.cos(x)

    # These parameters shouldn't matter:
    tau = 0
    g=0

    #vpop, spktimes2, J = neuron_population(1, t_max, dt, N, g, tau, E, tauD, initial, A=J0*(2*np.pi), BB=J1*(2*np.pi))
    vpop, spktimes2, J = neuron_population(1, t_max, dt, N, tau, E, tauD, initial, J0=J0, J1=J1)
    #plt.plot(x, vpop[time_index_10,:],'k.')

    '''# Compute the average of vpop over the last 5 time units
    time_index_last_5 = range(max(0, time_index_10 - int(40 / dt)), time_index_10)  # Indices of the last 20 time steps
    vpop_avg_last_5 = np.mean(vpop[time_index_last_5, :], axis=0)  # Average over the last 20 time points

    # Plot the average value of vpop
    plt.plot(x, vpop_avg_last_5, 'k.')'''
    # Smooth vpop by averaging with the adjacent spatial points within the smooth_window
    smoothed_vpop = np.copy(vpop[time_index_10, :])  # Copy the original values to modify

    # Iterate over each spatial index to smooth it
    smooth_window=10
    half_window = smooth_window // 2  # Determine how many neighbors to average on either side
    time_index_last_5 = range(max(0, time_index_10 - int(10 / dt)), time_index_10)

    for i in range(N):
        # Calculate the start and end indices for the window, ensuring boundaries are respected
        start_idx = max(0, i - half_window)
        end_idx = min(N - 1, i + half_window)

        # Average the values within the window
        smoothed_vpop[i] = np.mean(vpop[time_index_last_5, start_idx:end_idx + 1])

    # Plot the smoothed vpop
    #plt.plot(x, smoothed_vpop, 'k.')
    min_index = np.argmin(smoothed_vpop)  # Find index of the minimum value
    shift = -np.pi - x[min_index]
    x_shifted = x + shift
    x_shifted = np.mod(x_shifted + np.pi, 2 * np.pi) - np.pi  # Wrap to the range [-pi, pi]
    plt.plot(x_shifted, smoothed_vpop, 'k.', label='sLIF',markersize=2, linewidth=linewidth)
    plt.plot(x, np.ones(len(x)), 'grey', linestyle='--', linewidth=linewidth, label='_nolegend_')  # no legend
    plt.plot(xmean, space_values_at_10, label='MF', linewidth=linewidth)

    plt.xlabel('Space (x)', fontsize=fontsize)
    plt.ylabel('Voltage (v)', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([-np.pi, np.pi])

    # Add legend at center bottom with small font
    plt.legend(loc='lower center', ncol=1, fontsize=labelsize, frameon=False)
    return vpop, spktimes2, J


def plot_matlab_code():
    #for E
    # this creates a figure
    maxsteps = 900
    # Load the .mat file
    data = loadmat('spatial_fig_data/lif_data 2.mat')

    # Extract relevant variables from the MATLAB structure
    l2_grid = data['l2grid'].flatten()  # This is the solution grid for the continuation
    E = data['Egrid'].flatten()
    mingrid = data['mingrid'].flatten()
    maxgrid = data['maxgrid'].flatten()
    J0 = data['J0'].flatten()

    # Set up the figure
    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    # Plot real(u) vs x
    plt.plot(E[:maxsteps], np.real(l2_grid[:maxsteps]), colors='k')


    # Label the axes
    plt.xlabel('$E$', fontsize=14)
    plt.ylabel('$\| v - v_*\|_{L^2}$', fontsize=14)

    plt.subplot(2, 1, 2)
    plt.plot(E[:maxsteps], np.real(mingrid[:maxsteps]), colors='k')
    plt.plot(E[:maxsteps], np.real(maxgrid[:maxsteps]), colors='k')
    # Add the plots for v = E if E < 1 and v = (J0 + sqrt(J0^2 + 4*(E-J0)))/2 if E > 1
    v_values = np.zeros_like(E)
    for i in range(maxsteps):
        if E[i] <= 1:
            v_values[i] = E[i]  # If E < 1, v = E
        else:
            v_values[i] = (J0 + np.sqrt(J0 ** 2 + 4 * (E[i] - J0))) / 2  # If E > 1, v as per given formula

    E_for_E_and_v_minus = np.linspace(min(E), 1, 500)
    # Plot the new curves
    plt.plot(E_for_E_and_v_minus, E_for_E_and_v_minus, colors='r')
    plt.plot(E_for_E_and_v_minus, (J0 - np.sqrt(J0 ** 2 + 4 * (E_for_E_and_v_minus- J0))) / 2, colors='r')

    E_for_v_plus = np.linspace(min(E), max(E), 500)
    plt.plot(E_for_v_plus, (J0 + np.sqrt(J0 ** 2 + 4 * (E_for_v_plus - J0))) / 2, colors='r')

    # Show the plot
    #plt.savefig('matlabJ0_-5_J1_10.png')
    plt.show()
    return

def plot_matlab_code_J0():
    # this creates a figure
    maxsteps = 400
    # Load the .mat file
    data = loadmat('spatial_fig_data/lif_data_J0.mat')

    # Extract relevant variables from the MATLAB structure
    l2_grid = data['l2grid'].flatten()  # This is the solution grid for the continuation
    E = data['E'].flatten()
    mingrid = data['mingrid'].flatten()
    maxgrid = data['maxgrid'].flatten()
    J0 = data['J0_grid'].flatten()

    # Set up the figure
    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    # Plot real(u) vs x
    plt.plot(J0[:maxsteps], np.real(l2_grid[:maxsteps]),colors='k')


    # Label the axes
    plt.xlabel('$J0$', fontsize=14)
    plt.ylabel('$\| v - v_*\|_{L^2}$', fontsize=14)
    plt.xlim([-100,5])

    plt.subplot(2, 1, 2)
    plt.plot(J0[:maxsteps], np.real(mingrid[:maxsteps]), colors='k')
    plt.plot(J0[:maxsteps], np.real(maxgrid[:maxsteps]), colors='k')

    '''# Add the plots for v = E if E < 1 and v = (J0 + sqrt(J0^2 + 4*(E-J0)))/2 if E > 1
    v_values = np.zeros_like(J0)
    for i in range(maxsteps):
        if E <= 1:
            v_values[i] = E  # If E < 1, v = E
        else:
            v_values[i] = (J0[i] + np.sqrt(J0[i] ** 2 + 4 * (E - J0[i]))) / 2  # If E > 1, v as per given formula

    E_for_E_and_v_minus = np.linspace(min(E), 1, 500)
    # Plot the new curves
    plt.plot(E_for_E_and_v_minus, E_for_E_and_v_minus)
    plt.plot(E_for_E_and_v_minus, (J0 - np.sqrt(J0 ** 2 + 4 * (E_for_E_and_v_minus- J0))) / 2)
    '''
    J0_for_v_plus = np.linspace(min(J0), max(J0), 500)
    plt.plot(J0_for_v_plus, (J0_for_v_plus + np.sqrt(J0_for_v_plus ** 2 + 4 * (E - J0_for_v_plus))) / 2, colors='r')

    plt.xlim([-15., 5])

    # Show the plot
    plt.savefig('matlabE_5_J1_10.png')
    plt.show()
    return


def detect_upper_fold(Egrid):
    ind = []  # Initialize an empty list to store indices
    # Loop through the elements of Egrid, excluding the first and last elements
    for i in range(1, len(Egrid) - 1):
        if Egrid[i] > Egrid[i - 1] and Egrid[i] > Egrid[i + 1]:
            ind.append(i)  # Add the index to the result if the condition is met
    return ind

def detect_lower_fold(Egrid):
    ind = []  # Initialize an empty list to store indices
    # Loop through the elements of Egrid, excluding the first and last elements
    for i in range(1, len(Egrid) - 2):
        if Egrid[i] < Egrid[i - 1] and Egrid[i] < Egrid[i + 1]:
            ind.append(i)  # Add the index to the result if the condition is met
    return ind

def plot_E_continuation(ax, linewidth=1,fontsize=10, labelsize=10):
    axBif = inset_axes(ax, width="100%", height="40%", loc='upper right')
    axL2 = inset_axes(ax, width="100%", height="35%", loc='lower right')

    maxsteps = 160
    # Load the .mat file
    data = loadmat('spatial_fig_data/lif_data.mat')

    # Extract relevant variables from the MATLAB structure
    l2_grid = data['l2grid'].flatten()  # This is the solution grid for the continuation
    E = data['Egrid'].flatten()
    mingrid = data['mingrid'].flatten()
    maxgrid = data['maxgrid'].flatten()
    J0 = data['J0'].flatten()

    Efoldidx = detect_upper_fold(E)
    Efoldidx=Efoldidx[0]
    Eturing = E[0]

    #ax4 = fig.add_subplot(gs[3:4, 1:2])
    axL2.plot(E[:maxsteps], np.real(l2_grid[:maxsteps]), color='k', linewidth=linewidth)


    #ax5 = fig.add_subplot(gs[2:3, 1:2])
    axBif.plot(E[:Efoldidx], np.real(mingrid[:Efoldidx]), color='k', linestyle='--', linewidth=linewidth)
    axBif.plot(E[:Efoldidx], np.real(maxgrid[:Efoldidx]), color='k', linestyle='--', linewidth=linewidth)
    axBif.plot(E[Efoldidx:maxsteps], np.real(mingrid[Efoldidx:maxsteps]), color='k',linewidth=linewidth)
    axBif.plot(E[Efoldidx:maxsteps], np.real(maxgrid[Efoldidx:maxsteps]), color='k', linewidth=linewidth)
    # Add the plots for v = E if E < 1 and v = (J0 + sqrt(J0^2 + 4*(E-J0)))/2 if E > 1
    v_values = np.zeros_like(E[:maxsteps])
    for i in range(maxsteps):
        if E[i] <= 1:
            v_values[i] = E[i]  # If E < 1, v = E
        else:
            v_values[i] = (J0 + np.sqrt(J0 ** 2 + 4 * (E[i] - J0))) / 2  # If E > 1, v as per given formula

    E_for_E_and_v_minus = np.linspace(min(E), 1, 500)
    # Plot the new curves
    axBif.plot(E_for_E_and_v_minus, E_for_E_and_v_minus, color='m', linewidth=linewidth)
    axBif.plot(E_for_E_and_v_minus, (J0 - np.sqrt(J0 ** 2 + 4 * (E_for_E_and_v_minus - J0))) / 2, color='m', linewidth=linewidth)

    if J0 < 2:
        E_for_v_plus_unstab = np.linspace(1, Eturing, 500)
        E_for_v_plus_stab = np.linspace(Eturing, max(E), 500)
    if J0 >= 2:
        E_for_v_plus_unstab = np.linspace(min(E), Eturing, 500)
        E_for_v_plus_stab = np.linspace(Eturing, max(E), 500)
    axBif.plot(E_for_v_plus_stab, (J0 + np.sqrt(J0 ** 2 + 4 * (E_for_v_plus_stab - J0))) / 2, color='m', linewidth=linewidth)
    axBif.plot(E_for_v_plus_unstab, (J0 + np.sqrt(J0 ** 2 + 4 * (E_for_v_plus_unstab - J0))) / 2, color='m', linestyle='--', linewidth=linewidth)


    # labels
    axL2.set_xlabel(r'$E$', fontsize=fontsize)
    axL2.tick_params(labelsize=labelsize)
    axL2.set_ylabel(r'$\| v - v_0\|_{L^2}$', fontsize=fontsize - 1)
    axBif.set_ylabel(r'v', fontsize=fontsize-1)
    axBif.tick_params(labelsize=labelsize, labelbottom=False)
    axBif.set_ylim([-3, 4])
    axBif.set_xlim([0, 16])
    axL2.set_xlim([0, 16])

    return

def plot_J0_continuation(ax, linewidth=1,fontsize=10, labelsize=10):
    axBif = inset_axes(ax, width="100%", height="40%", loc='upper right')
    axL2 = inset_axes(ax, width="100%", height="35%", loc='lower right')

    # this creates a figure
    maxsteps = 200
    # Load the .mat file
    data = loadmat('spatial_fig_data/lif_data_J0.mat')

    # Extract relevant variables from the MATLAB structure
    l2_grid = data['l2grid'].flatten()  # This is the solution grid for the continuation
    E = data['E'].flatten()
    mingrid = data['mingrid'].flatten()
    maxgrid = data['maxgrid'].flatten()
    J0 = data['J0_grid'].flatten()

    J0turing=J0[0]
    J0fold_idx=detect_upper_fold(J0)
    J0fold_idx=J0fold_idx[0]

    # Set up the figure
    #ax5 = fig.add_subplot(gs[3:4, 2:3])
    # Plot real(u) vs x
    axL2.plot(J0[:J0fold_idx], np.real(l2_grid[:J0fold_idx]),color='k',linestyle='--', linewidth=linewidth)
    axL2.plot(J0[J0fold_idx:maxsteps], np.real(l2_grid[J0fold_idx:maxsteps]), color='k', linewidth=linewidth)
    axL2.set_ylim([0, 5])
    axL2.set_xlabel('$J_0$', fontsize=fontsize)
    axL2.tick_params(labelsize=labelsize)
    #plt.ylabel('$\| v - v_0\|_{L^2}$', fontsize=14)
    axL2.set_xlim([-5, 3])
    axL2.set_ylim([0, 5])


    #plt.plot(J0[:maxsteps], np.real(mingrid[:maxsteps]), color='k')
    #plt.plot(J0[:maxsteps], np.real(maxgrid[:maxsteps]), color='k')
    axBif.plot(J0[:J0fold_idx], np.real(mingrid[:J0fold_idx]), color='k', linestyle='--', linewidth=linewidth)
    axBif.plot(J0[J0fold_idx:maxsteps], np.real(mingrid[J0fold_idx:maxsteps]), color='k', linewidth=linewidth)

    axBif.plot(J0[:J0fold_idx], np.real(maxgrid[:J0fold_idx]), color='k', linestyle='--', linewidth=linewidth)
    axBif.plot(J0[J0fold_idx:maxsteps], np.real(maxgrid[J0fold_idx:maxsteps]), color='k', linewidth=linewidth)

    J0_for_v_plus_unstab = np.linspace(min(J0), J0turing, 500)
    J0_for_v_plus_stab = np.linspace(J0turing,max(J0), 100)
    axBif.plot(J0_for_v_plus_unstab, (J0_for_v_plus_unstab + np.sqrt(J0_for_v_plus_unstab ** 2 + 4 * (E - J0_for_v_plus_unstab))) / 2, color='m',linestyle='--', linewidth=linewidth)
    axBif.plot(J0_for_v_plus_stab, (J0_for_v_plus_stab + np.sqrt(J0_for_v_plus_stab ** 2 + 4 * (E - J0_for_v_plus_stab))) / 2, color='m', linewidth=linewidth)
    axBif.tick_params(labelsize=labelsize, labelbottom=False)
    axL2.set_ylabel(r'$\| v - v_0\|_{L^2}$', fontsize=fontsize - 1)
    axBif.set_ylabel(r'v', fontsize=fontsize - 1)
    axBif.set_ylim([-3, 4])
    axBif.set_xlim([-5, 3])
    return

def plot_J1_continuation(ax, linewidth=1,fontsize=10, labelsize=10):
    axBif = inset_axes(ax, width="100%", height="40%", loc='upper right')
    axL2 = inset_axes(ax, width="100%", height="35%", loc='lower right')

    # this creates a figure
    maxsteps = 86
    # Load the .mat file
    data = loadmat('spatial_fig_data/lif_data_J1.mat')

    # Extract relevant variables from the MATLAB structure
    l2_grid = data['l2grid'].flatten()  # This is the solution grid for the continuation
    E = data['E'].flatten()
    J0 = data['J0'].flatten()
    mingrid = data['mingrid'].flatten()
    maxgrid = data['maxgrid'].flatten()
    J1 = data['J1_grid'].flatten()
    J1fold_idx=detect_lower_fold(J1)
    J1fold_idx=J1fold_idx[0]
    J1_turing=J1[0]

    # Set up the figure
    axL2.plot(J1[:J1fold_idx], np.real(l2_grid[:J1fold_idx]), color='k',linestyle='--', linewidth=linewidth)
    axL2.plot(J1[J1fold_idx:maxsteps], np.real(l2_grid[J1fold_idx:maxsteps]), color='k', linewidth=linewidth)
    axL2.set_ylim([0, 5])
    axL2.set_xlabel(r'$J_1$', fontsize=fontsize)
    axL2.tick_params(labelsize=labelsize)
    #plt.ylabel(r'$\| v - v_0\|_{L^2}$', fontsize=14)
    axL2.set_xlim([4.8, 7])
    axL2.set_ylim([0, 2])

    #plt.plot(J1[:maxsteps], np.real(mingrid[:maxsteps]), color='k')
    #plt.plot(J1[:maxsteps], np.real(maxgrid[:maxsteps]), color='k')
    axBif.plot(J1[:J1fold_idx], np.real(mingrid[:J1fold_idx]), color='k', linestyle='--', linewidth=linewidth)
    axBif.plot(J1[J1fold_idx:maxsteps], np.real(mingrid[J1fold_idx:maxsteps]), color='k', linewidth=linewidth)
    axBif.plot(J1[:J1fold_idx], np.real(maxgrid[:J1fold_idx]), color='k', linestyle='--', linewidth=linewidth)
    axBif.plot(J1[J1fold_idx:maxsteps], np.real(maxgrid[J1fold_idx:maxsteps]), color='k', linewidth=linewidth)

    J1_for_v_plus = np.linspace(min(J1), J1_turing, 100)
    J1_for_v_plus_unstab = np.linspace(J1_turing, max(J1), 500)
    axBif.plot(J1_for_v_plus, ((J0 + np.sqrt(J0 ** 2 + 4 * (E - J0))) / 2) * np.ones(len(J1_for_v_plus)), color='m', linewidth=linewidth)
    axBif.plot(J1_for_v_plus_unstab, ((J0 + np.sqrt(J0 ** 2 + 4 * (E - J0))) / 2) * np.ones(len(J1_for_v_plus_unstab)), color='m',linestyle='--', linewidth=linewidth)
    axBif.tick_params(labelsize=labelsize, labelbottom=False)
    axL2.set_ylabel(r'$\| v - v_0\|_{L^2}$', fontsize=fontsize - 1)
    axBif.set_ylabel(r'v', fontsize=fontsize - 1)
    axBif.set_xlim([4.8, 7])
    axBif.set_ylim([.5, 1.7])
    return


def testing_connect_mat():
    # testing normalization factor of synaptic kernel and scaling of connectivity matrix to make sure consistent

    fig = plt.figure(layout='constrained')
    gs = GridSpec(1, 4, figure=fig)

    t_max = 5
    dt = .001
    tauD = 0
    E = 2
    J0 = -2
    J1 = 8
    v0 = (J0 + np.sqrt(J0 ** 2 + 4 * (E - J0))) / 2

    # run mean field approx
    Nx = 100
    x = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    pertstrength = 10
    m_history = np.zeros((int(tauD / dt), Nx))  # The history
    center = v0
    for tt in range(0, int(tauD / dt)):
        m_history[tt, :] = center + pertstrength * np.cos(x)
    m_initial = center + pertstrength * np.cos(x)
    m = MFsim_with_con_mat(m_initial, m_history, int(t_max / dt), Nx, int(tauD / dt), dt, J0, J1, E)
    m2 = MF_spatial_sim(m_initial, m_history, int(t_max / dt), Nx, int(tauD / dt), dt, J0, J1, E)

    # Plot Mean Field
    ax = fig.add_subplot(gs[0:1, 1:2])
    plt.imshow(m.T, aspect='auto', cmap='viridis', origin='lower', interpolation='none',
               extent=[0, len(m) * dt, -np.pi, np.pi])
    plt.ylabel('Space (x)')
    plt.xlabel('Time (t)')
    plt.xlim([0, t_max])

    ax2 = fig.add_subplot(gs[0:1, 2:3])
    # Time and space parameters
    time_at_10 = t_max - 1  # We're interested in t = 10
    # Find the index for t = 10
    time_index_10 = int(time_at_10 / dt)


    # Extract the data at t = 10 (this will give us the corresponding space values)
    space_values_at_10 = m[time_index_10, :]  # Assuming rows represent time, and columns represent space
    xmean = np.linspace(-np.pi, np.pi, len(space_values_at_10), endpoint=False)
    plt.plot(xmean, space_values_at_10)

    space_values_at_102 = m2[time_index_10, :]  # Assuming rows represent time, and columns represent space
    xmean2 = np.linspace(-np.pi, np.pi, len(space_values_at_102), endpoint=False)
    plt.plot(xmean2, space_values_at_102)

    ax3 = fig.add_subplot(gs[0:1,0:1])
    plt.imshow(m2.T, aspect='auto', cmap='viridis', origin='lower', interpolation='none',
               extent=[0, len(m2) * dt, -np.pi, np.pi])
    plt.ylabel('Space (x)')
    plt.xlabel('Time (t)')
    plt.xlim([0, t_max])


    #########################################################################################
    fig.set_size_inches(10, 5)  # width by height in inches
    sns.despine(fig)
    plt.tight_layout()
    # plt.savefig('figspatial_v1.png')
    plt.show()
    return


def build_spatial_fig_v3():
    #updating v2 now that some of v1s information is redundant with v2
    fig = plt.figure()
    fig.set_size_inches(6.5, 3)  # width by height in inches
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1])
    gs_phaseDiag = gs[0,0]
    gs_turingCurves = gs[0,1]
    gs_MFbump = gs[0,2]
    gs_spikingBump = gs[0,3]
    gs_compare = gs[1,0]
    gs_EbifDiag = gs[1,1]
    #gs_El2norm = gs[2,1]
    gs_J0bifDiag = gs[1, 2]
    #gs_J0l2norm = gs[2, 2]
    gs_J1bifDiag = gs[1, 3]
    #gs_J1l2norm = gs[2, 3]


    t_max = 20
    dt = .01
    tauD = 0
    N = 1000
    E = 3
    J0 = -2
    J1 = 8
    v0 = (J0+np.sqrt(J0**2 +4*(E-J0)))/2

    # run mean field approx
    Nx=100
    x = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    pertstrength = 2
    m_history = np.zeros((int(tauD/dt), Nx))  # The history
    center = v0
    for tt in range(0, int(tauD/dt)):
        m_history[tt, :] = center + pertstrength * np.cos(x)
    m_initial = center + pertstrength * np.cos(x)
    m = MF_spatial_sim(m_initial, m_history, int(t_max / dt), Nx, int(tauD / dt), dt, J0, J1, E)

    # Plot Mean Field
    ax_MFbump = fig.add_subplot(gs_MFbump)
    #m = np.load('../SpatialMeanField/StandingBumpdt.01_E2_D1_J0-2_J8.npy')
    #plt.imshow(m.T, aspect='auto', cmap='viridis', origin='lower', interpolation='none', extent=[0, len(m) * .01, -np.pi, np.pi])
    plot_spatial_MF(m, dt, ax= ax_MFbump, color_bar=True, labelsize=labelsize, vnum=1.2)
    ax_MFbump.set_ylabel('Space (x)', fontsize=fontsize)
    ax_MFbump.set_xlabel('Time (t)', fontsize=fontsize)
    ax_MFbump.tick_params(labelsize=labelsize)
    ax_MFbump.set_xlim([5, t_max])

    ax_compare = fig.add_subplot(gs_compare)
    #vpop, spktimes2, J = plot_slice_of_mean_vs_spike_anim(m, dt, t_max, tauD, N, J0, J1, E)
    vpop, spktimes2, J = plot_slice_of_mean_vs_spike(m, dt, t_max, tauD, N, J0, J1, E, fontsize=fontsize, labelsize=labelsize, linewidth=linewidth)

    ax_turingCurves = fig.add_subplot(gs_turingCurves)
    plot_turing_curve_varied_J1(labelsize=labelsize, fontsize=fontsize, linewidth=linewidth)
    plt.xlabel(r'Avg. Syn. Str. ($J_0$)', fontsize=fontsize)
    plt.ylabel(r'Rest. Pot. ($E$)', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([-5, 10])
    plt.ylim([-5, 5])

    ax_spikingBump = fig.add_subplot(gs_spikingBump)
    plot_pop(spktimes2, dt, ms=0.5)
    plt.xlim([5, t_max])
    plt.ylim([0,N])
    plt.xlabel('Time (t)', fontsize=fontsize)
    plt.ylabel('Neuron (i)', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    ### MATLAB CODE - E ####################
    ax_EbifDiag = fig.add_subplot(gs_EbifDiag)
    ax_EbifDiag.set_visible(False)
    #ax_El2norm = fig.add_subplot(gs_El2norm)
    plot_E_continuation(ax_EbifDiag, fontsize=fontsize, labelsize=labelsize, linewidth=linewidth) #have to change axes location in function

    ### MATLAB CODE - J0 ####################
    ax_J0bifDiag = fig.add_subplot(gs_J0bifDiag)
    ax_J0bifDiag.set_visible(False)
    #ax_J0l2norm = fig.add_subplot(gs_J0l2norm)
    plot_J0_continuation(ax_J0bifDiag, fontsize=fontsize, labelsize=labelsize, linewidth=linewidth)

    ### MATLAB CODE - J1 ####################
    ax_J1bifDiag = fig.add_subplot(gs_J1bifDiag)
    ax_J1bifDiag.set_visible(False)
    #ax_J1l2norm = fig.add_subplot(gs_J1l2norm)
    plot_J1_continuation(ax_J1bifDiag, fontsize=fontsize, labelsize=labelsize, linewidth=linewidth)

    ax_phaseDiag=fig.add_subplot(gs_phaseDiag)
    Turing_phase_diag_colors(fontsize=fontsize, labelsize=labelsize, linewidth=0.6)
    plt.xlabel(r'Avg. Syn. Str. ($J_0$)', fontsize=fontsize)
    plt.ylabel(r'Rest. Pot. ($E$)', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim([-5, 10])
    plt.ylim([-5, 5])

    ####Letter Labels
    plt.tight_layout(h_pad=1, w_pad=1, rect=[0, 0, .99, 0.95])#h_pad=2, w_pad=2, rect=[0, 0, .98, 0.98])  # change to tight layout before adding labels
    letter_xoffset = -0.08
    letter_yoffset = 0.01
    add_subplot_label(fig, ax_phaseDiag, 'A', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_turingCurves, 'B', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_MFbump, 'C', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_spikingBump, 'D', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_compare, 'E', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_EbifDiag, 'F', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_J0bifDiag, 'G', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)
    add_subplot_label(fig, ax_J1bifDiag, 'H', x_offset=letter_xoffset, y_offset=letter_yoffset,
                      fontsize=letterlabelsize)

    #########################################################################################
    sns.despine(fig)
    #plt.tight_layout()
    plt.savefig('figs_saved/figspatial_v3.png')
    plt.savefig('figs_saved/figspatial_v3.pdf')
    plt.show()
    return




if __name__ == '__main__':
    build_spatial_fig_v3()
    #plot_matlab_code_J0()
    #testing_connect_mat()
    #matlab_fold_and_turing_plot()
