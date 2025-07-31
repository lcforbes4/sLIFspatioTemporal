import numpy as np
import scipy
import matplotlib.pyplot as plt

# This file takes a matlab file of a periodic orbit branch contuation created by BIFTOOL and preps it for use in python

def get_periodic_branch_points(filename,varied_param_name,varied_parameter_ind, delta_check):
    # Load the .mat file
    mat_contents = scipy.io.loadmat(filename, squeeze_me=True, struct_as_record=False)
    branch4 = mat_contents['branch4']

    # Extract point data
    point = branch4.point  # This should be a 1x102 array of point structures
    num_points = len(point)

    # Define the structured array data type
    data_type = [
        (varied_param_name, float),
        ('Equil', float),
        ('Max', float),
        ('Min', float),
        ('Period', float),
        ('L2', float),
        ('Stability_color', 'U10')
    ]

    # Initialize the structured array
    val_list = np.empty(num_points, dtype=data_type)

    # Populate the structured array
    for i in range(num_points):
        param = point[i].parameter
        J = param[0]
        E = param[1]
        v0 = (J + np.sqrt(J ** 2 + 4 * (E - J))) / 2

        if delta_check == 0:
            val_list[i] = (
                param[varied_parameter_ind],  # E value
                v0,  # Equilibrium value
                np.max(point[i].profile[0]),  # Max Amplitude
                np.min(point[i].profile[0]),  # Min Amplitude
                point[i].period,  # Period
                np.sqrt(point[i].period * scipy.integrate.simpson((point[i].profile[0] - np.mean(point[i].profile[0]))**2, x=point[i].mesh)),  # L2 Norm
                #'r*' if np.absolute(point[i].stability.mu[0]) < 1 else 'b*'  # Stability color
                'r*' if ((isinstance(point[i].stability.mu, float) and np.abs(point[i].stability.mu) < 1)) or (
                        not isinstance(point[i].stability.mu, float) and np.abs(point[i].stability.mu[0]) < 1) else 'b*'
            )
        if delta_check==1:
            val_list[i] = (
                param[varied_parameter_ind],  # E value
                v0,  # Equilibrium value
                np.max(point[i].profile),  # Max Amplitude
                np.min(point[i].profile),  # Min Amplitude
                point[i].period,  # Period
                #np.sqrt(point[i].period * scipy.integrate.simpson((point[i].profile - v0) ** 2, x=point[i].mesh)),
                np.sqrt(point[i].period * scipy.integrate.simpson((point[i].profile - np.mean(point[i].profile)) ** 2,
                                                                  x=point[i].mesh)),# L2 Norm
                #'r*' if np.absolute(point[i].stability.mu[0]) < 1 else 'b*'  # Stability color
                'r*' if ( (isinstance(point[i].stability.mu, float) and np.abs(point[i].stability.mu) < 1)) or (
                        not isinstance(point[i].stability.mu, float) and np.abs(point[i].stability.mu[0]) < 1) else 'b*'

            )

        # This plots the profile of each point on the periodic solution branch:
        #plt.plot(point[i].period*np.linspace(0,1,len(point[i].profile)),point[i].profile,'r-' if np.absolute(point[i].stability.mu[0]) < 1 else 'b-')

    # Print the stability color of the first record
    #print(val_list[0]['Stability_color'])

    return val_list

if __name__ == '__main__':
    #Parameters
    J_choice = 1 #1- delta, 2-delayed exp, 3-alpha function
    J = -3.04
    E = 3
    tau = 5
    delay = 2

    val_list = get_periodic_branch_points(
        '../../old_figure_scripts/delayedExpCaseFigures/fig2data/matlabdatav1/branch4_Jsave.mat', 'J_value', 0)
    #plt.plot(val_list[:]['J_value'], val_list[:]['L2'])
    plt.legend

    plt.show()