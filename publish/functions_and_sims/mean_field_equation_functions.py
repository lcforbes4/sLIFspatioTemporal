import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from publish.functions_and_sims.spiking_network_functions import generate_spatial_connectivity_mat


# Functions for a numerical time stepper for the mean field approx

def J(x, J_0, J_1):
    '''
    Returns strength of cosine coupling between two neurons (excitatory. or inhib.) given distance between them

    -PARAMETERS- \n
    x : distance between two neurons \n
    J0 : mean coupling strength \n
    J1 : modulation strength \n
    '''
    return (J_0 + J_1*np.cos(x))/(2*np.pi)

def phi(v, b=1, thresh=1):
    '''
    Returns the probability of a neuron(s) firing given the membrane voltage given by a threshold linear function

    -PARAMETERS- \n
    v: input voltage \n
    b=1: slope \n
    thresh=1: threshold value \n
    '''
    return b * np.maximum(v - thresh, 0)

def delta_euler_step(E, J, v, t, dt, dv_dt, Ndelay):
    if t > Ndelay:
        dv_dt[t-1] = E - v[t-1] - v[t-1] * phi(v[t - 1]) + J * phi(v[t - Ndelay - 1])
    else:
        dv_dt[t-1] = E - v[t-1] - v[t-1] * phi(v[t - 1]) + J * phi(v[0]) # Assumes a history of the initial condition

    v[t] = v[t-1] + dt * dv_dt[t-1]

    return v, dv_dt

def delayed_exp_euler_step(E, J, tau, v, dt, dv_dt, t, Ndelay, s):
    if t > Ndelay:
        ds_dt = - s[t-1]/tau + J/tau * phi(v[t - Ndelay - 1])
    else:
        ds_dt = - s[t-1]/tau + J/tau * phi(v[0])

    s[t] = s[t-1] + dt * ds_dt

    dv_dt[t-1] = E - v[t-1] - v[t-1] * phi(v[t-1]) + s[t]

    v[t] = v[t-1] + dt * dv_dt[t-1]

    return v, dv_dt, s

def alpha_func_euler_step(E, J, tau, v, dt, dv_dt, t, Ndelay, s, ss):
    if t > Ndelay:
        dss_dt = -(2/tau)*ss[t-1] - 1/(tau**2)*s[t-1] + J/(tau**2) * phi(v[t - Ndelay - 1])
    else:
        dss_dt = -(2/tau)*ss[t-1] - 1/(tau**2)*s[t-1] + J/(tau**2) * phi(v[0])
    ss[t] = ss[t - 1] + dt * dss_dt
    ds_dt = ss[t]
    s[t] = s[t - 1] + dt * ds_dt
    dv_dt[t-1] = E - v[t - 1] - v[t - 1] * phi(v[t - 1]) + s[t]
    v[t] = v[t - 1] + dt * dv_dt[t-1]
    return v, dv_dt, s, ss

def calculate_mean_field(J_choice, E, J, delay, tstop, dt, initial, tau):
    Nt = int(tstop / dt)
    Ndelay = int(delay / dt)
    v0 = (J + numpy.sqrt(J**2 + 4*(E-J)))/2

    # Initial Conditions
    v_initial = initial

    v = np.zeros((Nt,))
    dv_dt = np.zeros((Nt,))
    s = np.zeros((Nt,))
    ss = np.zeros((Nt,))
    v[0] = v_initial
    # dv_dt[0] = 1
    # s[0] = 1

    # The Euler Step Loop
    for t in range(1, Nt):
        if J_choice == 1:
            v, dv_dt = delta_euler_step(E, J, v, t, dt, dv_dt, Ndelay)
        elif J_choice == 2:
            v, dv_dt, s = delayed_exp_euler_step(E, J, tau, v, dt, dv_dt, t, Ndelay, s)
        elif J_choice == 3:
            v, dv_dt, s, ss = alpha_func_euler_step(E, J, tau, v, dt, dv_dt, t, Ndelay, s, ss)
        else:
            print("Error: Unknown J choice")

    return v, v0

def ramping_mean_field(J_choice, E, J, delay, tstop, dt, initial, tau, ramping_param_ind, ramp_param_start, ramp_param_end):
    #will ignore the input value of the parameter if you're ramping it

    Nt = int(tstop / dt)
    Ndelay = int(delay / dt)

    ramp_param_step = (ramp_param_end- ramp_param_start) / Nt
    print(ramp_param_step)

    # Initial Conditions
    v_initial = initial
    if ramping_param_ind == 0:  # ramping J
        J = ramp_param_start
    elif ramping_param_ind == 1:  # ramping E
        E  = ramp_param_start
    elif ramping_param_ind == 2:  # ramping tau
        tau  = ramp_param_start
    #elif ramping_param_ind == 3:  # ramping delay
        #todo Ndelay makes it weird

    v = np.zeros((Nt,))
    dv_dt = np.zeros((Nt,))
    s = np.zeros((Nt,))
    ss = np.zeros((Nt,))
    v[0] = v_initial
    # dv_dt[0] = 1
    # s[0] = 1

    # The Euler Step Loop
    for t in range(1, Nt): #todo switched this from 1
        #print(J)
        if J_choice == 1:
            v, dv_dt = delta_euler_step(E, J, v, t, dt, dv_dt, Ndelay)
        elif J_choice == 2:
            v, dv_dt, s = delayed_exp_euler_step(E, J, tau, v, dt, dv_dt, t, Ndelay, s)
        elif J_choice == 3:
            v, dv_dt, s, ss = alpha_func_euler_step(E, J, tau, v, dt, dv_dt, t, Ndelay, s, ss)
        else:
            print("Error: Unknown J choice")

        if ramping_param_ind==0: # ramping J
            J += ramp_param_step
        elif ramping_param_ind==1: # ramping E
            E += ramp_param_step
        elif ramping_param_ind==2: # ramping tau
            tau += ramp_param_step
        #elif ramping_param_ind==3: # ramping delay
            #todo Ndelay makes it weird

    return v

def plot_ramped_mean_field_vs_param(v, tstop, dt, ramp_param_start, ramp_param_end, color='k'):
    Nt = int(tstop / dt)
    ramp_param_step = (ramp_param_end - ramp_param_start) / Nt
    tplot = np.linspace(ramp_param_start, ramp_param_end, Nt)

    #plt.plot(np.arange(0, tstop, dt), v0 * np.ones(Nt), color='r')
    plt.plot(tplot, v, color=color, linewidth=1)
    plt.xlim([0, tstop])
    #plt.ylim(v_ax)

    return

def plot_spatial_MF(m, dt, vnum= 1.3, ax=None, xlim=None, fontsize=10, labelsize=10, color_bar=False):
    if ax is None:
        ax = plt.gca()

    v_th = 1  # Threshold to center the color scale
    norm = TwoSlopeNorm(vmin=np.min(m), vcenter=v_th, vmax=np.max(m))

    im = ax.imshow(
        m.T, aspect='auto', cmap='seismic', origin='lower', interpolation='none', norm=norm,
        extent=[0, len(m) * dt, -np.pi, np.pi]
    )

    ax.set_ylabel('Space (x)', fontsize=fontsize)
    ax.set_xlabel('Time (t)', fontsize=fontsize)
    ax.tick_params(labelsize=labelsize)
    if xlim is not None:
        ax.set_xlim(xlim)

    if color_bar:
        # Place colorbar above the axes, outside the main plot
        cax = inset_axes(
            ax,
            width="100%", height="100%",  # Full width of axes
            loc='upper center',
            bbox_to_anchor=(0, vnum, 1, 0.1),  # Shift it above axes
            bbox_transform=ax.transAxes,
            borderpad=0
        )
        cbar = ax.figure.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=labelsize, length=3, pad=1)#, top=True, bottom=False, labeltop=True, labelbottom=False)
        cbar.outline.set_visible(False)
        cbar.set_label("Voltage (v)", labelpad=-22, fontsize=labelsize, loc='center')

    return


def plot_mean_field(v, tstop, dt, v0, color='k', linewidth = 1):
    # made to plot average
    tplot = np.arange(0, tstop, dt)
    Nt = int(tstop / dt)
    #v_ax = [1, 2]

    #plt.plot(np.arange(0, tstop, dt), v0 * np.ones(Nt), color='r')
    plt.plot(tplot, v, color=color, linewidth=linewidth)
    plt.xlim([0, tstop])
    #plt.ylim(v_ax)

    return

def MF_spatial_sim(m_initial, m_history, Tmax, Nx, D, dt, J_0, J_1, E, L=2 * np.pi):
    dx = L / Nx
    x = np.linspace(-np.pi, np.pi, Nx, endpoint=False)  # spatial grid
    m = np.zeros((Tmax, Nx))
    m[0, :] = m_initial

    # --- 1. Precompute distance matrix and J_kernel ---
    distance_matrix = np.abs(x[:, None] - x[None, :])
    distance_matrix = np.minimum(distance_matrix, L - distance_matrix)  # ring topology
    J_kernel = J(distance_matrix, J_0, J_1)  # shape: (Nx, Nx)

    # --- 2. Run simulation over time ---
    for t in range(0, Tmax - 1):
        if t < D:
            delayed_m = m_history[t, :]
        else:
            delayed_m = m[t - D, :]
        phi_delayed = phi(delayed_m)
        integral_term = np.dot(J_kernel, phi_delayed) * dx

        # Update equation:
        m[t + 1, :] = m[t, :] + dt * (E - m[t, :] - m[t, :] * phi(m[t, :]) + integral_term)

    return m

def MFsim_with_con_mat(m_initial, m_history, Tmax, Nx, D, dt, J_0, J_1, E, L=2 * np.pi):
    #uses J matrix (as opposed to function) to test with connectivity matrix code for spiking net
    dx = L/Nx
    J_0 = J_0
    J_1 = J_1

    # Create spatial grid
    x = np.linspace(-np.pi, np.pi, Nx, endpoint=False)  # Spatial grid (periodic)

    # Initialize activity m(x, t) as zeros (no activity initially)
    m = np.zeros((Tmax, Nx))

    m[0, :] = m_initial

    J_mat = generate_spatial_connectivity_mat(J_0,J_1, Nx, p=1)

    #v = np.zeros((Nt,))
    #dv_dt = np.zeros((Nt,))

    for t in range(0, Tmax - 1):
        # Update m(x, t) based on the equation
        for i in range(0, Nx):
            integral_term = 0
            # Spatial integral: J(|x - y|) m(y, t-D)
            for j in range(0, Nx):
                # Periodic boundary condition for the ring
                distance = np.abs(x[i] - x[j])
                if distance > L / 2:  # Ensure minimum distance on the ring
                    distance = L - distance

                if t < D:
                    integral_term += J_mat[i,j] * phi(m_history[t, j])# * dx
                    # integral_term += J(distance, J_0, J_1) * m[0,j] * dx
                else:
                    integral_term += J_mat[i,j] * phi(m[t - D, j]) #* dx

            # Update the activity using the equation
            m[t + 1, i] = m[t, i] + dt * (E - m[t, i] - m[t, i] * phi(m[t, i]) + integral_term)

    return m




if __name__ == '__main__':
    #Parameters
    J_choice = 1 #1- delta, 2-delayed exp, 3-alpha function
    J = -4
    E = 3
    tau = 1
    delay = 2

    #Graphing/Euler Options
    dt = .01
    tstop = 100
    #v_ax = [0, 2]
    #v_dot_ax = [-1.5, 1.5]

    # Calculate Equilibrium
    v0 = J + np.sqrt(J ** 2 + 4 * (E - J))
    v0 /= 2

    # Initial Conditions
    v_initial = 1.69

    #vmean, v0 = ramping_mean_field(J_choice, 2, -16, delay, tstop, dt, 1.5, tau, 0, -5, -2)
    vmean, v0 =calculate_mean_field(J_choice, E, J, delay, tstop, dt, v_initial, tau)
    plot_mean_field(vmean, tstop, dt, v0)

    plt.show()


def MFsimulation_2_unif_pulses(m_initial, m_history, Tmax, Nx, D, dt, J_0, J_1, E, L=2 * np.pi):
    Tmax=int(Tmax/dt)
    D = int(D/dt)
    #uses J matrix (as opposed to function)
    dx = L/Nx

    # Create spatial grid
    x = np.linspace(-np.pi, np.pi, Nx, endpoint=False)  # Spatial grid (periodic)

    # Initialize activity m(x, t) as zeros (no activity initially)
    m = np.zeros((Tmax, Nx))

    m[0, :] = m_initial

    J_mat = generate_spatial_connectivity_mat(J_0,J_1, Nx, p=1)/dx

    #v = np.zeros((Nt,))
    #dv_dt = np.zeros((Nt,))
    pulse1_time=25
    pulse1_duration = 2
    pulse1_amp=1

    pulse2_time = 62.7
    pulse2_amp = -1
    pulse2_duration = 1.75


    for t in range(0, Tmax - 1):

        if t==int(pulse1_time/dt):
            E = E + pulse1_amp
        if t == int((pulse1_time+pulse1_duration) / dt):
            E = E - pulse1_amp
        if t==int(pulse2_time/dt):
            E = E + pulse2_amp
        if t==int((pulse2_time+pulse2_duration)/dt):
            E = E - pulse2_amp


        # Now calculate the integral term based on whether t < D or not
        if t < D:
            # Using m_history[t, :] (history values) for the integral term
            integral_term = J_mat * phi(m_history[t, :])  # Nx x Nx matrix
        else:
            # Using m[t - D, :] (current values at time t-D) for the integral term
            integral_term = J_mat * phi(m[t - D, :])  # Nx x Nx matrix

        # Perform the spatial sum over j for each i by multiplying with dx and summing across axis 1 (axis=1)
        integral_term = np.sum(integral_term, axis=1)*dx  # Resulting in a vector of size Nx

        # Update the activity using the equation for all i at once
        m[t + 1, :] = m[t, :] + dt * (E - m[t, :] - m[t, :] * phi(m[t, :]) + integral_term)

    return m


def MFsim_2_local_pulses(m_initial, m_history, Tmax, Nx, D, dt, J_0, J_1, E, L=2 * np.pi, p=0.5):
    #meanField
    Tmax=int(Tmax/dt)
    D = int(D/dt)
    #uses J matrix (as opposed to function)
    dx = L/Nx

    # Create spatial grid
    x = np.linspace(-np.pi, np.pi, Nx, endpoint=False)  # Spatial grid (periodic)

    # Initialize activity m(x, t) as zeros (no activity initially)
    m = np.zeros((Tmax, Nx))

    m[0, :] = m_initial

    J_mat = generate_spatial_connectivity_mat(J_0,J_1, Nx, p=p)/dx

    #v = np.zeros((Nt,))
    #dv_dt = np.zeros((Nt,))
    pulse1_time= 25
    pulse1_duration = 2
    pulse1_amp = -150

    pulse2_time = 60
    pulse2_amp = 150
    pulse2_duration = 2


    for t in range(0, Tmax - 1):
        # Now calculate the integral term based on whether t < D or not
        if t < D:
            # Using m_history[t, :] (history values) for the integral term
            integral_term = J_mat * phi(m_history[t, :])  # Nx x Nx matrix
        else:
            # Using m[t - D, :] (current values at time t-D) for the integral term
            integral_term = J_mat * phi(m[t - D, :])  # Nx x Nx matrix

        # Perform the spatial sum over j for each i by multiplying with dx and summing across axis 1 (axis=1)
        integral_term = np.sum(integral_term, axis=1)*dx  # Resulting in a vector of size Nx

        # Update the activity using the equation for all i at once
        m[t + 1, :] = m[t, :] + dt * (E - m[t, :] - m[t, :] * phi(m[t, :]) + integral_term)

        if t in [int(pulse1_time/dt), int((pulse1_time+pulse1_duration) / dt)]:
            #E = E + pulse1_amp
            m[t + 1, :int(Nx/2)] += dt*pulse1_amp
        if t in [int(pulse2_time/dt),int((pulse2_time+pulse2_duration)/dt)]:
            #E = E + pulse2_amp
            m[t + 1, :int(Nx/2)] += dt*pulse2_amp

    return m
