import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm
import matplotlib.patches as patches

# This is a spiking simulation functions for different conductivity functions (determined by J_choice)
# Plots, the population, a single neuron, as well as the population average


# This is a spike probability/intensity function
def intensity_func(v, B=1, v_th=1, p=1):
    x = v - v_th

    if len(np.shape(x)) > 0:
        x[x < 0] = 0
    elif x < 0:
        x = 0
    else:
        pass

    return B * x ** p

def generate_connectivity_mat(g, p, N):
    # g_bar, an NxN matrix filled with the conductivity between the neurons, normalized. With a certain probability
    # whether they are connected or not
    g_bar = np.random.binomial(n=1, p=p, size=(N, N)) * g / p / N
    np.fill_diagonal(g_bar, 0)  # make sure connection =0 if i=j
    return g_bar

def generate_spatial_connectivity_mat(A, B, N, p = 0.5):
    # equivalent to generate_connectivity_mat when B=0
    connection_prob = cos_prob(A, B, N)
    g_bar = np.random.binomial(n=1, p=p, size=(N, N)) * connection_prob / p * (2*np.pi/N)
    np.fill_diagonal(g_bar, 0)  # make sure connection =0 if i=j
    print(g_bar)
    return g_bar


def cos_prob(A,B,Nx):
    # returns (1/2pi)*(A+Bcos(x)) where x in [-pi,pi] with Nx grid points

    x  = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    theta_diff = x[:, np.newaxis] - x  # N x N matrix of all pairwise distances

    # Ensure distances are wrapped around the ring by taking the minimum of absolute distances and their complement (2*pi - distance)
    theta_diff = np.abs(theta_diff)
    theta_diff = np.minimum(theta_diff, 2 * np.pi - theta_diff)

    # Connection probability functions using the formula p(r) = p0 + p1 * cos(r)
    p = (A + B  * np.cos(theta_diff))/(2*np.pi)
    return p

# spiking network simulation
def neuron_population(J_choice, tstop, dt, N, tau, E, tauD, initial, J0, J1):

    B = 1 #slope of intensity function
    v_th = 1  # voltage threshold
    p = 1 # power of intensity function (1=linear, 2=quadratic)
    v_r = 0  # Reset Voltage
    Nt = int(tstop / dt)  # number of times steps
    Ndelay = int(tauD / dt)

    # Need to add Ndelay, because im using the first Ndelay steps as initial data
    Nt = Nt + Ndelay

    #connectivity matrix
    g_bar = generate_spatial_connectivity_mat(J0, J1, N)

    v = np.zeros((Nt, N)) # Initialize array to hold voltage at each time step for each neuron
    J = np.zeros((Nt, N)) # NtxN Matrix for the conductivity-input of each neuron
    nu = np.zeros((Nt, N)) # aux synaptic variable for alpha func
    n = np.zeros(N, )  # Array to hold boolean of whether each neuron has spiked or not
    spktimes = []

    # initial conditions:
    v[:Ndelay] = initial


    for t in range(1, Nt):
        if t < Ndelay:
            # dont evolve v, just simulate spiking
            lam = intensity_func(v[t], B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)
        else:
            if J_choice == 1:  # J(t) = g_bar * delta(t-tauD)
                v[t] = v[t - 1] + dt * (-v[t - 1] + E) + J[t - 1, :] - n * (v[t - 1] - v_r)
            elif J_choice == 2 or J_choice == 3:  # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
                v[t] = v[t - 1] + dt * (-v[t - 1] + E + J[t - 1, :]) - n * (v[t - 1] - v_r)
            lam = intensity_func(v[t], B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)

        #rprint(n)
        # Different J(t) Functions:
        if J_choice == 1: # J(t) = g_bar * delta(t-tauD)
            if t < Nt-Ndelay:
                J[t + Ndelay] = J[t + Ndelay] + g_bar @ n
        elif J_choice == 2: # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
            if t < Nt - Ndelay:
                J[t+Ndelay] += g_bar.dot(n)/tau
            J[t] += J[t-1] * (1-dt/tau)
        elif J_choice == 3: # J(t) = g_bar 1/tau^2 t e^(-t/tau) H(t)
            if t < Nt - Ndelay:
                nu[t+Ndelay] += g_bar.dot(n)/tau**2
            nu[t] += nu[t-1] + dt*(-2 * nu[t-1]/tau - J[t-1]/(tau ** 2))
            J[t] += J[t-1] + dt*nu[t-1]

        else:
            print("Error: Unknown J choice")

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t-Ndelay, i])

        #print(v[t])

    # throw out initial data
    v_final = v[Ndelay:]

    return v_final, spktimes, J


# old version, see ramp_param_sim
def synch_pulse_then_ramp_sim(J_choice, tstop, dt, N, g, tau, E, tauD, pulse_time, duration, pulse_amp, initial, ramp_time):
    # This function changes the parameter E by pulse_amp at time pulse_time, then ramps it back to original parameter
    # value over the course of ramp_time

    B = 1 #slope of intensity function
    v_th = 1  # voltage threshold
    p = 1 # power of intensity function (1=linear, 2=quadratic)
    v_r = 0  # Reset Voltage
    Nt = int(tstop / dt)  # number of times steps
    N_pulse_time = pulse_time/dt
    N_duration = duration/dt
    Ndelay = int(tauD / dt)
    ramp_time = int(ramp_time / dt)

    # Calculate Equilibrium
    v0 = g + np.sqrt(g ** 2 + 4 * (E - g))
    v0 /= 2

    #g_bar, an NxN matrix filled with the conductivity between the neurons, normalized. With a certain probability
    #whether they are connected or not
    connection_prob = 0.5
    g_bar = np.random.binomial(n=1, p=connection_prob, size=(N, N)) * g / connection_prob / N
    np.fill_diagonal(g_bar, 0) #make sure connection =0 if i=j
    #print(g_bar)
    #np.save('g_bar.npy',g_bar)
    #g_bar = np.load('g_bar.npy')

    v = np.zeros((Nt, N)) # Initialize array to hold voltage at each time step for each neuron, IV=1
    v[:Ndelay] = initial

    J = np.zeros((Nt, N)) # NtxN Matrix for the conductivity-input of each neuron
    nu = np.zeros((Nt, N)) # Initial J values all 0
    n = np.zeros(N, )  # Array to hold boolean of whether each neuron has spiked or not
    spktimes = []

    #v_temp , v0 = calculate_mean_field(J_choice, E, g, delay, tstop, dt, initial, tau)
    for i in range(0,N):
        v[:Ndelay,i] = initial #v_temp[:Ndelay] #uncomment if you want initial data to be mean field

    for t in range(0, Nt):


        #lam = intensity_func(v[t], B=B, v_th=v_th, p=p)
        #lam[lam > 1 / dt] = 1 / dt # requiring the range
        #n = np.random.binomial(n=1, p=dt * lam)

        if t < Ndelay:
            # dont evolve v
            lam = intensity_func(v[t], B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)
        else:
            if J_choice == 1:  # J(t) = g_bar * delta(t-tauD)
                v[t] = v[t - 1] + dt * (-v[t - 1] + E) + J[t - 1, :] - n * (v[t - 1] - v_r)
            elif J_choice == 2 or J_choice == 3:  # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
                v[t] = v[t - 1] + dt * (-v[t - 1] + E + J[t - 1, :]) - n * (v[t - 1] - v_r)
            lam = intensity_func(v[t], B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)

        # The Synched Pulse
        if t == N_pulse_time:
            E = E + pulse_amp

        if t in range(int(N_pulse_time+N_duration),int(N_pulse_time+N_duration+ramp_time)):
            dE = (- pulse_amp)/len(range(int(N_pulse_time+N_duration),int(N_pulse_time+N_duration+ramp_time)))
            E = E + dE

        # Different J(t) Functions:

        if J_choice==1: # J(t) = g_bar * delta(t-tauD)
            if t < Nt-Ndelay:
                J[t + Ndelay] = J[t + Ndelay] + g_bar @ n
        elif J_choice==2: # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
            if t < Nt - Ndelay:
                J[t+Ndelay] += g_bar.dot(n)/tau
            J[t] += J[t-1] * (1-dt/tau)
        elif J_choice==3: # J(t) = g_bar 1/tau^2 t e^(-t/tau) H(t)
            if t < Nt - Ndelay:
                nu[t+Ndelay] += g_bar.dot(n)/tau**2
            nu[t] += nu[t-1] + dt*(-2 * nu[t-1]/tau - J[t-1]/(tau ** 2))
            J[t] += J[t-1] + dt*nu[t-1]

        else:
            print("Error: Unknown J choice")

        spkind = np.where(n > 0)[0]
        if t>Ndelay:
            for i in spkind:
                spktimes.append([t-Ndelay, i])

        # Shifting the voltage
        v_shift = v#[Ndelay:]


    return v_shift, spktimes, J

def synch_2pulses_sim(J_choice, tstop, dt, N, g, tau, E, tauD, pulse_time, duration, pulse_amp, initial, g_bar):
    # This function applies 2 pulses to the parameter E a network of spiking neurons
    # intended to show bi-stability from homog activity to oscillaitons

    B = 1 #slope of intensity function
    v_th = 1  # voltage threshold
    p = 1 # power of intensity function (1=linear, 2=quadratic)
    v_r = 0  # Reset Voltage
    N_pulse_time = pulse_time/dt #start time of first (on) pulse
    N_duration = duration/dt
    Ndelay = int(tauD / dt)
    Nt = int(tstop / dt) + Ndelay  # number of times steps
    print(Ndelay)
    print(Nt)

    # Second (off) pulse parameters
    N_duration2 = int(1/ dt)  # duration of second pulse
    N_pulse_time2 = int(58 / dt) + Ndelay # start time of the second pulse
    pulse_amp2 = -1


    #ramp_time = int(ramp_time / dt)

    # Calculate Equilibrium
    v0 = g + np.sqrt(g ** 2 + 4 * (E - g))
    v0 /= 2

    #dJ = (g - (-3.05)) / len(range(0, int(N_pulse_time)))
    #g = -3.05


    v = np.zeros((Nt, N)) # Initialize array to hold voltage at each time step for each neuron, IV=1
    v[:Ndelay] = initial

    ramp_param_array = []


    J = np.zeros((Nt, N)) # NtxN Matrix for the conductivity-input of each neuron
    nu = np.zeros((Nt, N)) # Initial J values all 0
    n = np.zeros(N, )  # Array to hold boolean of whether each neuron has spiked or not
    spktimes = []

    #v_temp , v0 = calculate_mean_field(J_choice, E, g, delay, tstop, dt, initial, tau)
    for i in range(0,N):
        v[:Ndelay,i] = initial #v_temp[:Ndelay] #uncomment if you want initial data to be mean field

    for t in range(0, Nt):


        #lam = intensity_func(v[t], B=B, v_th=v_th, p=p)
        #lam[lam > 1 / dt] = 1 / dt # requiring the range
        #n = np.random.binomial(n=1, p=dt * lam)

        if t < Ndelay:
            # dont evolve v
            lam = intensity_func(v[t], B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)
        else:
            if J_choice == 1:  # J(t) = g_bar * delta(t-tauD)
                v[t] = v[t - 1] + dt * (-v[t - 1] + E) + J[t - 1, :] - n * (v[t - 1] - v_r)
            elif J_choice == 2 or J_choice == 3:  # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
                v[t] = v[t - 1] + dt * (-v[t - 1] + E + J[t - 1, :]) - n * (v[t - 1] - v_r)
            lam = intensity_func(v[t], B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)

        # The First Synched Pulse
        if t == N_pulse_time+Ndelay:
            E = E + pulse_amp
        if t == N_pulse_time+Ndelay+N_duration:
            E = E - pulse_amp

        # The Second Pulse
        if t == N_pulse_time2:
            E = E + pulse_amp2
        if t == N_pulse_time2+N_duration2:
            E = E - pulse_amp2

        ramp_param_array.append(E)
        #if t in range(0,int(N_pulse_time)):
        #    g_bar = (g_bar / g) * (g + dJ)
        #    g = g + dJ
        if t % int(1/dt) == 0:
            print(t*dt)
        #if t in range(int(N_pulse_time+N_duration),int(N_pulse_time+N_duration+ramp_time)):
        #    dE = (- pulse_amp)/len(range(int(N_pulse_time+N_duration),int(N_pulse_time+N_duration+ramp_time)))
        #    E = E + dE

        # Different J(t) Functions:

        if J_choice==1: # J(t) = g_bar * delta(t-tauD)
            if t < Nt-Ndelay:
                J[t + Ndelay] = J[t + Ndelay] + g_bar @ n
        elif J_choice==2: # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
            if t < Nt - Ndelay:
                J[t+Ndelay] += g_bar.dot(n)/tau
            J[t] += J[t-1] * (1-dt/tau)
        elif J_choice==3: # J(t) = g_bar 1/tau^2 t e^(-t/tau) H(t)
            if t < Nt - Ndelay:
                nu[t+Ndelay] += g_bar.dot(n)/tau**2
            nu[t] += nu[t-1] + dt*(-2 * nu[t-1]/tau - J[t-1]/(tau ** 2))
            J[t] += J[t-1] + dt*nu[t-1]

        else:
            print("Error: Unknown J choice")

        spkind = np.where(n > 0)[0]
        if t>Ndelay:
            for i in spkind:
                spktimes.append([t-Ndelay, i])

    # Shifting the voltage
    v_shift = v[Ndelay:]
    ramp_param_array.append(g)
    ramp_param_array = ramp_param_array[Ndelay:]
    return v_shift, spktimes, J, ramp_param_array

def synch_2pulses_spatial_sim(J_choice, tstop, dt, N, g, tau, E, tauD, pulse_time, duration, pulse_amp, pulse2_time,
                                                                     duration2, pulse_amp2, initial, g_bar, which_portion):
    # This function applies 2 pulses to the parameter E a network of spiking neurons
    # intended to show bi-stability from homog activity to oscillaitons

    B = 1 #slope of intensity function
    v_th = 1  # voltage threshold
    p = 1 # power of intensity function (1=linear, 2=quadratic)
    v_r = 0  # Reset Voltage

    Ndelay = int(tauD / dt)

    Nt = int(tstop / dt) + Ndelay  # number of times steps

    # First pulse
    N_pulse_time = pulse_time / dt  # start time of first (on) pulse
    N_duration = duration / dt

    # Second (off) pulse parameters
    N_duration2 = int(duration2 / dt)  # duration of second pulse
    N_pulse_time2 = int(pulse2_time/ dt) + Ndelay # start time of the second pulse


    # Calculate Equilibrium
    v0 = g + np.sqrt(g ** 2 + 4 * (E - g))
    v0 /= 2
    v = np.zeros((Nt, N)) # Initialize array to hold voltage at each time step for each neuron, IV=1
    v[0,:] = initial
    print(initial)

    ramp_param_array = []

    J = np.zeros((Nt, N)) # NtxN Matrix for the conductivity-input of each neuron
    nu = np.zeros((Nt, N)) # Initial J values all 0
    n = np.zeros(N, )  # Array to hold boolean of whether each neuron has spiked or not
    spktimes = []

    for t in range(1, Nt):

        if t < Ndelay:
            # dont evolve v
            lam = intensity_func(v[t], B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)
        else:
            if J_choice == 1:  # J(t) = g_bar * delta(t-tauD)
                v[t] = v[t - 1] + dt * (-v[t - 1] + E) + J[t - 1, :] - n * (v[t - 1] - v_r)
            elif J_choice == 2 or J_choice == 3:  # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
                v[t] = v[t - 1] + dt * (-v[t - 1] + E + J[t - 1, :]) - n * (v[t - 1] - v_r)
            lam = intensity_func(v[t], B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)

        #pulses
        x = np.linspace(-np.pi, np.pi, N, endpoint=False)
        if t in [N_pulse_time+Ndelay, N_pulse_time+Ndelay+N_duration]:
            #E = E + pulse1_amp
            v[t, :] += pulse_amp*np.cos(x)
            #v[t, int(N / 3):int(2 * N / 3)] += pulse_amp
        if t in [N_pulse_time2+Ndelay, N_pulse_time2+Ndelay+N_duration2]:
            #E = E + pulse2_amp
            #v[t, :] += pulse_amp2*np.cos(x)
            if which_portion == 0: #middle third:
                v[t, int(N/3):int(2*N/3)] += pulse_amp2
            if which_portion == 1: #outer 2 thirds:
                v[t, :int(N / 3)] += pulse_amp2
                v[t, int(2 * N / 3):] += pulse_amp2



        if J_choice==1: # J(t) = g_bar * delta(t-tauD)
            if t < Nt-Ndelay:
                J[t + Ndelay] = J[t + Ndelay] + g_bar @ n
        elif J_choice==2: # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
            if t < Nt - Ndelay:
                J[t+Ndelay] += g_bar.dot(n)/tau
            J[t] += J[t-1] * (1-dt/tau)
        elif J_choice==3: # J(t) = g_bar 1/tau^2 t e^(-t/tau) H(t)
            if t < Nt - Ndelay:
                nu[t+Ndelay] += g_bar.dot(n)/tau**2
            nu[t] += nu[t-1] + dt*(-2 * nu[t-1]/tau - J[t-1]/(tau ** 2))
            J[t] += J[t-1] + dt*nu[t-1]

        else:
            print("Error: Unknown J choice")

        spkind = np.where(n > 0)[0]
        if t>Ndelay:
            for i in spkind:
                spktimes.append([t-Ndelay, i])

    # Shifting the voltage
    v_shift = v[Ndelay:]
    ramp_param_array.append(g)
    ramp_param_array = ramp_param_array[Ndelay:]
    return v_shift, spktimes, J, ramp_param_array

def ramp_param_sim(J_choice, tstop, dt, N, g, tau, E, tauD, initial, g_bar,  ramping_param_ind, ramp_param_start, ramp_param_end):
#updated version of synch_pulse_then_ramp_sim_memory_save_vers that simpifies it quite a bit and allows ramping in other variables beside E
#will ignore the input value of the parameter if you're ramping it

# outputs average voltage of population

# todo ramping of delay parameter doesnt work
    B = 1 #slope of intensity function
    v_th = 1  # voltage threshold
    p = 1 # power of intensity function (1=linear, 2=quadratic)
    v_r = 0  # Reset Voltage
    Nt = int(tstop / dt)  # number of times steps
    Ndelay = int(tauD / dt)

    ramp_param_step = (ramp_param_end- ramp_param_start) / Nt


    # Initial Conditions
    v_initial = initial
    if ramping_param_ind == 0:  # ramping J
        g_bar = (g_bar/g)*(ramp_param_start)
        g = ramp_param_start
    elif ramping_param_ind == 1:  # ramping E
        E = ramp_param_start
    elif ramping_param_ind == 2:  # ramping tau
        tau = ramp_param_start
        # elif ramping_param_ind == 3:  # ramping delay
        # todo Ndelay makes it weird

    # Calculate Equilibrium
    v0 = g + np.sqrt(g ** 2 + 4 * (E - g))
    v0 /= 2

    v_new = np.zeros(N)
    v_old = np.ones(N)*initial
    v_avg = np.zeros((Nt,))

    J = np.zeros((Ndelay+1, N)) #array to store the J values for all neurons from current time step to a delay later
    nu = np.zeros((Nt, N)) # Initial J values all 0
    n = np.zeros(N, )  # Array to hold boolean of whether each neuron has spiked or not
    spktimes = []


    for t in range(0, Nt):

        # updating v and calculating spikes
        if t < Ndelay:
            v_new = v_old

            lam = intensity_func(v_new, B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)
        else:
            if J_choice == 1:  # J(t) = g_bar * delta(t-tauD)
                v_new = v_old + dt * (-v_old + E) + J[0, :] - n * (v_old - v_r)
            elif J_choice == 2 or J_choice == 3:  # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
                v_new = v_old + dt * (-v_old + E + J[t - 1, :]) - n * (v_old - v_r)
            lam = intensity_func(v_new, B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)


        # the ramping
        if ramping_param_ind == 0:  # ramping J
            g_bar = (g_bar / g) * (g+ramp_param_step)
            g += ramp_param_step
        elif ramping_param_ind == 1:  # ramping E
            E += ramp_param_step
        elif ramping_param_ind == 2:  # ramping tau
            tau += ramp_param_step
        # elif ramping_param_ind==3: # ramping delay
        # todo Ndelay makes it weird

        # Updating Different J(t) Functions:
        if J_choice==1: # J(t) = g_bar * delta(t-tauD)
            if t < Nt-Ndelay:
                J[Ndelay] = J[Ndelay] + g_bar @ n
        elif J_choice==2: # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
            if t < Nt - Ndelay:
                J[t+Ndelay] += g_bar.dot(n)/tau
            J[t] += J[t-1] * (1-dt/tau)
        elif J_choice==3: # J(t) = g_bar 1/tau^2 t e^(-t/tau) H(t)
            if t < Nt - Ndelay:
                nu[t+Ndelay] += g_bar.dot(n)/tau**2
            nu[t] += nu[t-1] + dt*(-2 * nu[t-1]/tau - J[t-1]/(tau ** 2))
            J[t] += J[t-1] + dt*nu[t-1]
        else:
            print("Error: Unknown J choice")

        spkind = np.where(n > 0)[0]
        if t>Ndelay:
            for i in spkind:
                spktimes.append([t-Ndelay, i])

        # Calculate average
        v_avg[t] = np.mean(v_new)

        v_old = v_new
        J[:-1, :] = J[1:, :]
        J[-1, :] = 0
        if t % 1000 == 0:
            print(t)
    return v_avg, spktimes

def neuron_population_output_avg(J_choice, tstop, dt, N, g, tau, E, tauD, initial, g_bar):
    #has all the neurons at the same inital value for one delay length of time

    B = 1 #slope of intensity function
    v_th = 1  # voltage threshold
    p = 1 # power of intensity function (1=linear, 2=quadratic)
    v_r = 0  # Reset Voltage
    Nt = int(tstop / dt)  # number of times steps
    Ndelay = int(tauD / dt)

    # Need to add Ndelay, because im using the first Ndelay steps as initial data
    Nt = Nt + Ndelay

    # Calculate Equilibrium
    v0 = g + np.sqrt(g ** 2 + 4 * (E - g))
    v0 /= 2

    v_new = np.zeros(N)
    v_old = np.ones(N)*initial
    v_avg = np.zeros((Nt,))

    J = np.zeros((Ndelay+1, N)) #array to store the J values for all neurons from current time step to a delay later
    if J_choice==3:
        nu = np.zeros((Ndelay+1, N)) # Initial J values all 0
    n = np.zeros(N, )  # Array to hold boolean of whether each neuron has spiked or not
    spktimes = []

    #v_temp , v0 = calculate_mean_field(J_choice, E, g, delay, tstop, dt, initial, tau)

    for t in range(0, Nt):

        # updating v and calculating spikes
        if t < Ndelay:
            v_new = v_old

            lam = intensity_func(v_new, B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)
        else:
            if J_choice == 1:  # J(t) = g_bar * delta(t-tauD)
                v_new = v_old + dt * (-v_old + E) + J[0, :] - n * (v_old - v_r)
            elif J_choice == 2 or J_choice == 3:  # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
                v_new = v_old + dt * (-v_old + E + J[0, :]) - n * (v_old - v_r)
            lam = intensity_func(v_new, B=B, v_th=v_th, p=p)
            lam[lam > 1 / dt] = 1 / dt  # requiring the range
            n = np.random.binomial(n=1, p=dt * lam)

        # Updating Different J(t) Functions:
        if J_choice==1: # J(t) = g_bar * delta(t-tauD)
            if t < Nt-Ndelay:
                J[Ndelay] = J[Ndelay] + g_bar @ n
        elif J_choice==2: # J(t) = g_bar/tau * E^(-(t-tauD)/tau) * H(t-tauD)
            if t < Nt - Ndelay:
                J[Ndelay] += g_bar.dot(n)/tau
            J[1] += J[0] * (1-dt/tau)
        elif J_choice==3: # J(t) = g_bar 1/tau^2 t e^(-t/tau) H(t)
            if t < Nt - Ndelay:
                nu[Ndelay] += g_bar.dot(n)/tau**2
            nu[1] += nu[0] + dt*(-2 * nu[0]/tau - J[0]/(tau ** 2))#todo fix so dont save whole array
            J[1] += J[0] + dt*nu[0]

        else:
            print("Error: Unknown J choice")

        # Save spike times
        spkind = np.where(n > 0)[0]
        if t>Ndelay:
            for i in spkind:
                spktimes.append([t-Ndelay, i])

        # average
        v_avg[t] = np.mean(v_new)

        v_old = v_new
        J[:-1, :] = J[1:, :]
        J[-1, :] = 0
        if J_choice == 3:
            nu[:-1, :] = nu[1:, :]
            nu[-1, :] = 0
        if t % 1000 == 0:
            print(t)

    # throw out initial data
    v_final = v_avg[Ndelay:]

    return v_final, spktimes, J


def plot_pop(spktimes, dt, plotevery=1, xlim=None, ylim = None, xlabel=True, ylabel=True, fontsize=10, labelsize=10,ms=1):
    '''Creates a Raster Plot for the spike times a neuron population

    Parameters:
        spktimes:   a Nx2 array with the spike time (in time steps) in the first column and which neuron (index) spikedin the 2nd
        dt:         time step
        plotevery=1:how many of the neurons to plot (increase to decrease density of raster plot)
    '''

    datax = np.transpose(spktimes)[0]*dt  # the spike times
    datay = np.transpose(spktimes)[1]  # the index of neuron spiking

    plt.plot(datax[::plotevery], datay[::plotevery], 'k|', markersize=ms)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if xlabel==True:
        plt.xlabel("Time (t)", fontsize=fontsize)
    if ylabel==True:
        plt.ylabel("Neuron (i)",fontsize=fontsize)

    plt.tick_params(labelsize=labelsize)

    return

def plot_single_neuron(v, t_max, dt, i, color, linewidth=1):
    '''
    plots the voltage of the ith neuron

    :param v: (Nt,N) array of voltages (where Nt is total number of time steps in the sim and N is number of neurons)
    :param t_max: sim end time
    :param dt: time step
    :param i: index of neuron to plot
    :param color: color to plot
    '''

    #plot the neuron's voltage:
    v_temp = np.transpose(v)[i]
    plt.plot(np.arange(0, t_max, dt), v_temp, color, linewidth=linewidth)

    #plt.ylabel("Voltage")
    #plt.xlabel("Time")

    return

def plot_avg(t_max, dt, v, color='k', linewidth=1):
    '''Plots the population average voltage against time

    -PARAMETERS-
    t_max: end of simulation time

    dt: time step

    v: (Nt,N) array of voltages (where Nt is total number of time steps in the sim and N is number of neurons)

    color='k': color to plot
    '''

    v_temp = v.mean(axis=1)
    plt.plot(np.arange(0, t_max, dt), v_temp, color=color, linewidth=linewidth)

    #plt.ylabel("Voltage")
    #plt.xlabel("Time")
    return

def plot_spike_times(y_value, spktimes, tauD, dt, check ,i, color, labelsize=8):
    '''Function to plot tick marks for the spike times of neuron i

    -PARAMETERS-

    y_value: which y value of the plot to place the tick marks

    spktimes: a nx2 array with the time of spiking (in time steps) in the first column and which neuron (index) spiked in the 2nd

    tauD: the time delay

    dt: time step

    check: 1 if want to plot an 2 sided arrow after the first spike to show delay

    i: index of which neuron spike times to plot

    color: color of the tick marks'''

    spikes = []
    for j in range(0,len(spktimes)):
        if np.transpose(spktimes)[1][j] == i:
            spikes.append(np.transpose(spktimes)[0][j]*dt)

    plt.plot(spikes,y_value*np.ones(len(spikes)), color)

    if check==1:
        # Plot the double-sided horizontal arrow with the specified coordinates
        plt.annotate('',
                    xy=(spikes[0] + tauD, y_value - 0.2),  # End of the arrow
                    xytext=(spikes[0], y_value - 0.2),  # Start of the arrow
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, linewidth = .8),
                    fontsize=8)

        # add text above the arrow
        plt.text(spikes[0] + tauD+.1, y_value-.1, 'delay', ha='center', va='bottom', fontsize=labelsize)


def plot_neuron_spike_histogram(spike_data,dt):
    """
    Plots a histogram of the number of neurons that spiked at each time.

    Parameters:
    spike_data (np.ndarray): (sim_time * dt)x2 array where the first column is the spike time (in time steps)
                                  and the second column is the neuron index.
    dt: time step used in spike_data
    """

    # Extract spike times from the first column
    spike_times = spike_data[:, 0]*dt

    # Create a histogram of the spike times
    #plt.hist(spike_times, bins='auto', edgecolor='black', alpha=0.7)

    Nt=int(5000/dt)

    #plt.hist(spike_times, bins=625, edgecolor='black', alpha=0.7)
    #plt.hist(spike_times, bins=1250, edgecolor='black', alpha=0.7)
    #plt.hist(spike_times, bins=2500, edgecolor='black', alpha=0.7)
    plt.hist(spike_times, bins=int(Nt/50), alpha=0.5, color='grey')

    # Set labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Neurons Spiked')
    plt.title('Histogram of Neuron Spikes')

    # Show grid
    plt.grid()
    return



if __name__ == '__main__':
    t_max = 50
    dt = 0.01
    N = 1000
    g = -30# height of conductance spike
    tau = 1  # width of conductance spike
    tauD = .2 #delay
    E = 10 # Resting Voltage
    j_choice = 1 #1- delta, 2- delayed exp, 3- alpha function
    J_1 = -100

    #initial = 0

    initial = np.zeros((int(tauD/dt), N))
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    for tt in range(0, int(tauD/dt)):
        initial[tt, :] = 0 + 1.5 * np.cos(x + np.pi / 2)

    v0 = (g + np.sqrt(g ** 2 + 4 * (E - g))) / 2
    print(v0-E)

    vpop, spktimes2, J = neuron_population(j_choice, t_max, dt, N, tau, E, tauD, initial, g, J_1)
    #print(vpop)
    #np.save('spiking_network_patterns_fig/vpop_breather_D_0.2_E_10_J1_20_J0_-60.npy',vpop)
    #np.save('spiking_network_patterns_fig/spktimes_breather_D_0.2_E_10_J1_20_J0_-60.npy', spktimes2)
    fig = plt.figure(layout='constrained')
    gs = GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :2])
    color_list =['red', 'blue','orange']
    for i in range(0,1):
        plot_single_neuron(vpop, t_max, dt, i, color_list[i])
    #plot_spike_times(1.5, spktimes2, 0,dt, 1, 0, 'r|')

    plt.xlim([0, t_max])

    ax2 = fig.add_subplot(gs[1, :2])
    plot_pop(spktimes2, dt)
    plt.xlim([-tauD, t_max])
    plt.ylim([0,N])

    ax3 = fig.add_subplot(gs[2, :2])
    plot_avg(t_max, dt, vpop, N, J)
    plt.xlim([0, t_max])

    plt.show()


