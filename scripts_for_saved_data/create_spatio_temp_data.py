import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from functions_and_sims.mean_field_equation_functions import MF_spatial_sim
from functions_and_sims.spiking_network_functions import neuron_population, plot_pop
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation, PillowWriter


def make_profile_animation(J0, J1, save_path="profile_animation.gif", output_fps=20):
    # Simulation parameters
    t_max = 210
    dt = 0.01
    Nx = 30
    E = 10
    tauD = .2
    v = 0
    IC_A = 1.5
    IC_B = 3

    initial_MF = np.zeros((int(tauD / dt), Nx))
    y = np.linspace(-np.pi, np.pi, Nx, endpoint=False)

    for tt in range(0, int(tauD / dt)):
        initial_MF[tt, :] = IC_A + IC_B * np.cos(y + np.pi / 2 + v * tt)

    # Run your simulation
    m = MF_spatial_sim(initial_MF[-1, :], initial_MF, int(t_max / dt), Nx, int(tauD / dt), dt, J0, J1, E)

    # Focus on final 100 seconds
    t_start = int((t_max - 10) / dt)
    m = m[t_start:]  # shape [time, Nx]
    T = m.shape[0]

    # Setup figure
    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='red')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(np.min(m), np.max(m))
    ax.set_title(f"Activity Profile\nJ0={J0}, J1={J1}")
    ax.set_xlabel("y")
    ax.set_ylabel("Activity")

    def update(frame):
        line.set_data(y, m[frame])
        ax.set_title(f"t = {t_start*dt+frame*dt:.2f} s | J0={J0}, J1={J1}")
        return line,

    ani = FuncAnimation(fig, update, frames=T, blit=True, interval=1000/output_fps)

    # Save animation
    if save_path.endswith(".gif"):
        ani.save(save_path, dpi=100, writer=PillowWriter(fps=output_fps))
    elif save_path.endswith(".mp4"):
        ani.save(save_path, dpi=100, writer='ffmpeg', fps=output_fps)
    else:
        raise ValueError("save_path must end with .gif or .mp4")

    plt.close()
    print(f"Saved profile animation to {save_path}")

def all_the_patterns(J0, J1, ax):
    t_max = 500
    t_min = t_max-100
    dt = 0.005
    Nx = 30
    E = 10
    tauD =  .2
    v = 0
    IC_A = 1.5
    IC_B = 3

    initial_MF = np.zeros((int(tauD / dt), Nx))
    x = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    for tt in range(0, int(tauD / dt)):
        initial_MF[tt, :] = IC_A + IC_B * np.cos(x + np.pi / 2 + v * tt)

    m = MF_spatial_sim(initial_MF[-1, :], initial_MF, int(t_max / dt), Nx, int(tauD / dt), dt, J0, J1, E)

    norm = TwoSlopeNorm(vcenter=1, vmin=np.min(m), vmax=np.max(m))

    # Plot on the provided Axes object
    ax.imshow(m.T, aspect='auto', extent=[0, t_max, -np.pi, np.pi], norm=norm, cmap='bwr')
    ax.set_title(f"J0={J0}, J1={J1}", fontsize=8)
    ax.set_xlim(t_min, t_max)
    ax.axis('off')
    print('hi')

def parameter_sweep():
    # Create grid of subplots for parameter sweep
    name = 'spots'
    J0_values = np.linspace(-13, -11, 5)  # Example values
    J1_values = np.linspace(-71, -73, 5)
    fig, axes = plt.subplots(len(J1_values), len(J0_values), figsize=(12, 10))
    # all_the_patterns(J0, J1, axes)

    for i, J1 in enumerate(J1_values):
        for j, J0 in enumerate(J0_values):
            ax = axes[i, j]
            all_the_patterns(J0, J1, ax)

    # plt.tight_layout()
    plt.savefig('parametersweep.pdf')
    plt.show()
    return

def run_compare_and_save():
    t_max = 150
    dt = 0.005
    N = 1000  # num of neurons
    Nx = 50  # MF discretization
    tau = 1  # width of conductance spike
    j_choice = 1  # 1- delta, 2- delayed exp, 3- alpha function
    E = 10  # Resting Voltage
    tauD = .2  # delay
    g = -5  # height of conductance spike
    J_1 = -25
    plot_xmin = 0
    v = 0
    IC_A = 0
    IC_B = 1.5
    name = 'SWTW'

    initial = np.zeros((int(tauD / dt), N))
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)

    for tt in range(0, int(tauD / dt)):
        initial[tt, :] = IC_A + IC_B * np.cos(x + np.pi / 2 + v * tt)

    v0 = (g + np.sqrt(g ** 2 + 4 * (E - g))) / 2
    print(v0 - E)

    vpop, spktimes2, J = neuron_population(j_choice, t_max, dt, N, tau, E, tauD, initial, g, J_1)
    # print(vpop)
    fig = plt.figure(layout='constrained')
    fig.set_size_inches(7, 3)  # width by height in inches
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

    ax2 = fig.add_subplot(gs[0, :2])
    plot_pop(spktimes2, dt)
    plt.xlim([plot_xmin, t_max])
    plt.ylim([0, N])

    ax4 = fig.add_subplot(gs[1, :2])
    initial_MF = np.zeros((int(tauD / dt), Nx))
    x = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    for tt in range(0, int(tauD / dt)):
        initial_MF[tt, :] = IC_A + IC_B * np.cos(x + np.pi / 2 + v * tt)
    m = MF_spatial_sim(initial_MF[-1, :], initial_MF, int((t_max) / dt), Nx, int(tauD / dt), dt, g, J_1, E)
    norm = TwoSlopeNorm(vcenter=1, vmin=np.min(m), vmax=np.max(m))
    plt.imshow(m.T, aspect='auto', cmap='bwr', origin='lower', interpolation='none',
               extent=[0, t_max, -np.pi, np.pi], norm=norm)

    plt.xlim([plot_xmin, t_max])
    plt.colorbar()

    np.save(f'current_fig_scripts/spiking_network_patterns_fig/spktimes_{name},D={tauD},E={E},J0={g}, J1={J_1}, dt={dt}, N={N}.npy', spktimes2)
    np.save(f'current_fig_scripts/spiking_network_patterns_fig/MF_{name}_D_{tauD}_E_{E}_J0={g}, J1={J_1}, dt={dt}, N={N}.npy', m)

    plt.show()
    return

def load_and_compare():
    fig = plt.figure(layout='constrained')
    fig.set_size_inches(7, 3)  # width by height in inches
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

    t_max = 400
    dt = 0.005
    N = 1000  # num of neurons
    plot_xmin = 0

    # load in data
    m = np.load(
        '../current_fig_scripts/spiking_network_patterns_fig/MF_weird_D_0.2_E_10_J0=-15, J1=-100, dt=0.005, N=1000.npy')
    spktimes2 = np.load(
        '../current_fig_scripts/spiking_network_patterns_fig/spktimes_weird,D=0.2,E=10,J0=-15, J1=-100, dt=0.005, N=1000.npy')


    ax2 = fig.add_subplot(gs[0, :2])
    plot_pop(spktimes2, dt)
    plt.xlim([plot_xmin, t_max])
    plt.ylim([0, N])

    ax4 = fig.add_subplot(gs[1, :2])
    norm = TwoSlopeNorm(vcenter=1, vmin=np.min(m), vmax=np.max(m))
    plt.imshow(m.T, aspect='auto', cmap='bwr', origin='lower', interpolation='none',
               extent=[0, m.shape[0] * dt, -np.pi, np.pi], norm=norm)

    plt.xlim([plot_xmin, t_max])
    plt.colorbar()

    plt.show()

    return

if __name__ == '__main__':
    run_compare_and_save()
    #load_and_compare()
    #parameter_sweep()

    '''J0 = -5
    J1 = -100
    make_profile_animation(J0=J0, J1=J1, save_path=f"animations/{name},j0{J0},j1{J1}.gif")'''


