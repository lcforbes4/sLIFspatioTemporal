import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm, pyplot as plt
from matplotlib.lines import Line2D
import os

from scipy.io import loadmat
from sympy import symbols, sqrt, acos, lambdify

fontsize = 8
labelsize = 6
letterlabelsize =10
linewidth = 1

def cubic(a,b,c,d):
    n = -b**3/27/a**3 + b*c/6/a**2 - d/2/a
    s = (n**2 + (c/3/a - b**2/9/a**2)**3)**0.5
    r0 = (n-s)**(1/3)+(n+s)**(1/3) - b/3/a
    r1 = (n - s) ** (1 / 3) - (n + s) ** (1 / 3) - b / (3 * a)
    r2 = -2 * (n ** (1 / 3)) - b / (3 * a)
    #r1 = (n+s)**(1/3)+(n+s)**(1/3) - b/3/a
    #r2 = (n-s)**(1/3)+(n-s)**(1/3) - b/3/a
    return (r0,r1,r2)

# Function to compute v_0 based on J_0 and E
def compute_vplus(J0, E):
    return (J0 + np.sqrt(J0**2 + 4 * (E - J0))) / 2

def compute_vminus(J0, E):
    return (J0 - np.sqrt(J0**2 + 4 * (E - J0))) / 2

def saddle_eq(J0, E):
    return J0**2 + 4*(E-J0)


def hopf_eq_delta(J0, J1, E, D, v0):
    return np.arccos(2 * v0 / J0) + D * J0 * np.sqrt(1 - (2 * v0 / J0) ** 2)

def hopf_eq_del_exp(J0, E, D, tau):
    v = (J0 + np.sqrt(J0 ** 2 + 4 * (E - J0))) / 2
    A = 1 + 4 * v * tau + 4 * v ** 2 * tau ** 2
    temp = (A * J0 + abs(J0) * np.sqrt(A ** 2 + 4 * tau * (tau * J0 ** 2 - 2 * v * A))) / (
            2 * J0 ** 2 * tau)
    # Define the implicit equation
    return (1 + 2 * v * tau) * np.arccos(temp) + J0 * D * np.sqrt(1 - (temp) ** 2)

def turing_hopf_eq_del_exp(J0, J1,E, D, tau):
    v = (J0 + np.sqrt(J0 ** 2 + 4 * (E - J0))) / 2
    A = 1 + 4 * v * tau + 4 * v ** 2 * tau ** 2
    temp = (A * (J1/2) + abs((J1/2)) * np.sqrt(A ** 2 + 4 * tau * (tau * (J1/2) ** 2 - 2 * v * A))) / (
            2 * (J1/2) ** 2 * tau)
    # Define the implicit equation
    return (1 + 2 * v * tau) * np.arccos(temp) + (J1/2)* D * np.sqrt(1 - (temp) ** 2)

def hopf_eq_alph_func(J0, E, D, tau):
    v = (J0 + np.sqrt(J0 ** 2 + 4 * (E - J0))) / 2
    # Solve cubic
    # y=
    coef3 = -(J0 ** 3 * tau) / (8 * (tau * v + 1) ** 3)
    coef2 = (J0 ** 2 * (4 * tau ** 3 * v ** 3 + 4 * tau ** 2 * v ** 2 + 5 * tau * v + 2)) / (
            4 * (tau * v + 1) ** 3)
    coef0 = (v * (2 * tau * v + 1) ** 4 - J0 ** 2 * tau * (tau * v + 1) ** 3) / (tau * (tau * v + 1) ** 3)
    coef1 = -(J0 * (2 * tau * v + 1) ** 2 * (4 * tau ** 2 * v ** 2 + 2 * tau * v + 1)) / (
            2 * tau * (tau * v + 1) ** 3)
    roots = cubic(coef3, coef2, coef1, coef0)
    y = roots[0]

    # Define the implicit equation
    return ( np.arccos(y)) ** 2 * (-2 * tau - 2 * v * tau ** 2) - D**2 * (J0 * y - 2 * v)

def turing_hopf_eq_alph_func(J0, J1,E, D, tau):
    v = (J0 + np.sqrt(J0 ** 2 + 4 * (E - J0))) / 2
    # Solve cubic
    # y=
    coef3 = -((J1/2) ** 3 * tau) / (8 * (tau * v + 1) ** 3)
    coef2 = ((J1/2) ** 2 * (4 * tau ** 3 * v ** 3 + 4 * tau ** 2 * v ** 2 + 5 * tau * v + 2)) / (4 * (tau * v + 1) ** 3)
    coef0 = (v * (2 * tau * v + 1) ** 4 - (J1/2) ** 2 * tau * (tau * v + 1) ** 3) / (tau * (tau * v + 1) ** 3)
    coef1 = -((J1/2) * (2 * tau * v + 1) ** 2 * (4 * tau ** 2 * v ** 2 + 2 * tau * v + 1)) / (
            2 * tau * (tau * v + 1) ** 3)
    roots = cubic(coef3, coef2, coef1, coef0)
    #coef_list=[coef3, coef2, coef1, coef0]
    #roots = np.roots(coef_list)
    y = roots[0]

    # Define the implicit equation
    return (np.arccos(y)) ** 2 * (-2 * tau - 2 * v * tau ** 2) - D**2 * ((J1/2) * y - 2 * v)

def turing_hopf_eq(J0, J1, E, D, v0):
    return np.arccos(4 * v0 / J1) + D * (J1 / 2) * np.sqrt(1 - (4 * v0 / J1) ** 2)


def plot_delta_instabilties_E_gthan_1(fontsize=10,labelsize=10, linewidth = 1):
    # Constants
    D = .2  # Arbitrary value for D; adjust as needed
    E = 10  # Arbitrary value for E; adjust as needed

    # Grid of J_0 and J_1 values
    J0_values = np.linspace(-50, 5, 200)
    J1_values = np.linspace(-50, 10, 200)

    # Create a meshgrid for contour plotting
    J0_grid, J1_grid = np.meshgrid(J0_values, J1_values)

    # Calculate the value of each equation over the grid
    vplus = compute_vplus(J0_grid, E)
    hopf_values = hopf_eq_delta(J0_grid, J1_grid, E, D, vplus)
    turing_hopf_values = turing_hopf_eq(J0_grid, J1_grid, E, D, vplus)

    # Main instability curve: 2v0 = J0 (i.e., J0 - 2v0 = 0)
    main_instability_values = saddle_eq(J0_grid, E)

    # Turing curve: J1 = 4v0 (i.e., J1 - 4v0 = 0)
    turing_values = J1_grid - 4 * compute_vplus(J0_grid, E)


    # Plot the contour for each equation (level sets where the equation equals 0)
    min_J1 = -6.67
    max_J1 = 4.72
    plt.contour(J0_grid, J1_grid, hopf_values, levels=[0], colors='green', label="Hopf curve", linewidths=linewidth)
    plt.contour(J0_grid, J1_grid, turing_hopf_values, levels=[0], colors='purple', label="Turing Hopf curve", linewidths = linewidth)
    plt.contour(J0_grid, J1_grid, main_instability_values, levels=[0], colors='blue', label="Main instability curve", linewidths = linewidth)
    plt.contour(J0_grid, J1_grid, turing_values, levels=[0], colors='red', label="Turing curve", linewidths = linewidth)

    # Labels and title
    plt.xlabel("$J_0$",fontsize=fontsize)
    plt.ylabel("$J_1$",fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #plt.title(f"E={E} and D={D}")
    # Create custom legend handles
    legend_handles = [
        Line2D([0], [0], color='green', lw=2, label='Hopf curve'),
        Line2D([0], [0], color='purple', lw=2, label='Turing Hopf curve'),
        # Line2D([0], [0], color='blue', lw=2, label='Main instability curve'),
        Line2D([0], [0], color='red', lw=2, label='Turing curve')
    ]

    # Add the legend with the custom handles
    #plt.legend(handles=legend_handles)

    return

def plot_delta_instabilties_E_lessthan_1(fontsize=10, labelsize=10, linewidth = 1):
    # Constants
    D = 1  # Arbitrary value for D; adjust as needed
    E = -2  # Arbitrary value for E; adjust as needed

    # Grid of J_0 and J_1 values
    J0_values = np.linspace(5, 10, 200)
    J1_values = np.linspace(-30, 30, 200)

    # Create a meshgrid for contour plotting
    J0_grid, J1_grid = np.meshgrid(J0_values, J1_values)

    # Calculate the value of each equation over the grid
    vplus = compute_vplus(J0_grid,E)
    vminus = compute_vminus(J0_grid, E)
    hopf_values = hopf_eq_delta(J0_grid, J1_grid, E, D, vplus)
    turing_hopf_values = turing_hopf_eq(J0_grid, J1_grid, E, D, vplus)
    turing_hopf_values2 = turing_hopf_eq(J0_grid, J1_grid, E, D, vminus)

    # Main instability curve: 2v0 = J0 (i.e., J0 - 2v0 = 0)
    main_instability_values = saddle_eq(J0_grid, E)

    # Turing curve: J1 = 4v0 (i.e., J1 - 4v0 = 0)
    turing_values = J1_grid - 4 * vplus
    turing_values2 = J1_grid - 4 * vminus


    # Plot the contour for each equation (level sets where the equation equals 0)
    min_J1 = -6.67
    max_J1 = 4.72
    plt.contour(J0_grid, J1_grid, hopf_values, levels=[0], colors='green', label="Hopf curve", linewidths = linewidth)
    plt.contour(J0_grid, J1_grid, turing_hopf_values, levels=[0], colors='purple', label="Turing Hopf curve", linewidths = linewidth)
    plt.contour(J0_grid, J1_grid, turing_hopf_values2, levels=[0], colors='purple', linestyles = 'dashed',label="Turing Hopf curve", linewidths = linewidth)
    plt.contour(J0_grid, J1_grid, main_instability_values, levels=[0], colors='blue', label="Main instability curve", linewidths = linewidth)
    plt.contour(J0_grid, J1_grid, turing_values, levels=[0], colors='red', label="Turing curve", linewidths = linewidth)
    plt.contour(J0_grid, J1_grid, turing_values2, levels=[0], colors='red', linestyles = 'dashed',label="Turing curve", linewidths = linewidth)

    # Labels and title
    plt.xlabel("$J_0$",fontsize=fontsize)
    plt.ylabel("$J_1$",fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    #plt.title(f"E={E} and D={D}")
    print(E)
    # Create custom legend handles
    legend_handles = [
        # Line2D([0], [0], color='green', lw=2, label='Hopf curve'),
        Line2D([0], [0], color='purple', lw=2, label='Turing Hopf curve'),
        Line2D([0], [0], color='blue', lw=2, label='Saddle Node Bif'),
        Line2D([0], [0], color='red', lw=2, label='Turing curve')
    ]

    # Add the legend with the custom handles
    #plt.legend(handles=legend_handles)

    return

def plot_delta_instabilties_E_gthan_1_v2():
    #version for poster
    # Constants
    D = 1  # Arbitrary value for D; adjust as needed
    E = 2  # Arbitrary value for E; adjust as needed

    # Grid of J_0 and J_1 values
    J0_values = np.linspace(-5, 5, 200)
    J1_values = np.linspace(-10, 10, 200)

    # Create a meshgrid for contour plotting
    J0_grid, J1_grid = np.meshgrid(J0_values, J1_values)

    # Calculate the value of each equation over the grid
    vplus = compute_vplus(J0_grid, E)
    hopf_values = hopf_eq_delta(J0_grid, J1_grid, E, D, vplus)
    turing_hopf_values = turing_hopf_eq(J0_grid, J1_grid, E, D, vplus)

    # Main instability curve: 2v0 = J0 (i.e., J0 - 2v0 = 0)
    main_instability_values = saddle_eq(J0_grid, E)

    # Turing curve: J1 = 4v0 (i.e., J1 - 4v0 = 0)
    turing_values = J1_grid - 4 * compute_vplus(J0_grid, E)


    # Plot the contour for each equation (level sets where the equation equals 0)
    min_J1 = -6.67
    max_J1 = 4.72
    plt.contour(J0_grid, J1_grid, hopf_values, levels=[0], colors='green', label="Hopf curve")
    plt.contour(J0_grid, J1_grid, turing_hopf_values, levels=[0], colors='purple', label="Turing Hopf curve")
    plt.contour(J0_grid, J1_grid, main_instability_values, levels=[0], colors='blue', label="Main instability curve")
    plt.contour(J0_grid, J1_grid, turing_values, levels=[0], colors='red', label="Turing curve")

    # Labels and title
    plt.xlabel(r'Mean Coupling Strength ($J_0$)')
    plt.ylabel(r'Modulation Strength ($J_1$)')
    #plt.title(f"E={E} and D={D}")
    # Create custom legend handles
    legend_handles = [
        Line2D([0], [0], color='green', lw=2, label='Hopf curve'),
        Line2D([0], [0], color='purple', lw=2, label='Turing Hopf curve'),
        # Line2D([0], [0], color='blue', lw=2, label='Main instability curve'),
        Line2D([0], [0], color='red', lw=2, label='Turing curve')
    ]

    # Add the legend with the custom handles
    #plt.legend(handles=legend_handles)

    return

def plot_delta_instabilties_E_lessthan_1_v2():
    #verison for poster
    # Constants
    D = 1  # Arbitrary value for D; adjust as needed
    E = -2  # Arbitrary value for E; adjust as needed

    # Grid of J_0 and J_1 values
    J0_values = np.linspace(5, 10, 200)
    J1_values = np.linspace(-20, 20, 200)

    # Create a meshgrid for contour plotting
    J0_grid, J1_grid = np.meshgrid(J0_values, J1_values)

    # Calculate the value of each equation over the grid
    vplus = compute_vplus(J0_grid,E)
    vminus = compute_vminus(J0_grid, E)
    hopf_values = hopf_eq_delta(J0_grid, J1_grid, E, D, vplus)
    turing_hopf_values = turing_hopf_eq(J0_grid, J1_grid, E, D, vplus)
    turing_hopf_values2 = turing_hopf_eq(J0_grid, J1_grid, E, D, vminus)

    # Main instability curve: 2v0 = J0 (i.e., J0 - 2v0 = 0)
    main_instability_values = saddle_eq(J0_grid, E)

    # Turing curve: J1 = 4v0 (i.e., J1 - 4v0 = 0)
    turing_values = J1_grid - 4 * vplus
    turing_values2 = J1_grid - 4 * vminus


    # Plot the contour for each equation (level sets where the equation equals 0)
    min_J1 = -6.67
    max_J1 = 4.72
    plt.contour(J0_grid, J1_grid, hopf_values, levels=[0], colors='green', label="Hopf curve")
    plt.contour(J0_grid, J1_grid, turing_hopf_values, levels=[0], colors='purple', label="Turing Hopf curve")
    plt.contour(J0_grid, J1_grid, turing_hopf_values2, levels=[0], colors='purple', linestyles = 'dashed',label="Turing Hopf curve")
    plt.contour(J0_grid, J1_grid, main_instability_values, levels=[0], colors='blue', label="Main instability curve")
    plt.contour(J0_grid, J1_grid, turing_values, levels=[0], colors='red', label="Turing curve")
    plt.contour(J0_grid, J1_grid, turing_values2, levels=[0], colors='red', linestyles = 'dashed',label="Turing curve")

    # Labels and title
    plt.xlabel(r'Mean Coupling Strength ($J_0$)')
    plt.ylabel(r'Modulation Strength ($J_1$)')
    #plt.title(f"E={E} and D={D}")
    print(E)
    # Create custom legend handles
    legend_handles = [
        # Line2D([0], [0], color='green', lw=2, label='Hopf curve'),
        Line2D([0], [0], color='purple', lw=2, label='Turing Hopf curve'),
        Line2D([0], [0], color='blue', lw=2, label='Saddle Node Bif'),
        Line2D([0], [0], color='red', lw=2, label='Turing curve')
    ]

    # Add the legend with the custom handles
    #plt.legend(handles=legend_handles)

    return


def plot_del_exp_instabilties_E_gthan_1(linewidth=1):
    # Constants
    D = 1  # Arbitrary value for D; adjust as needed
    E = 2  # Arbitrary value for E; adjust as needed
    tau=1

    # Grid of J_0 and J_1 values
    J0_values = np.linspace(-10, 5, 200)
    J1_values = np.linspace(-15, 10, 200)

    # Create a meshgrid for contour plotting
    J0_grid, J1_grid = np.meshgrid(J0_values, J1_values)

    # Calculate the value of each equation over the grid
    hopf_values = hopf_eq_del_exp(J0_grid, E, D, tau)
    turing_hopf_values = turing_hopf_eq_del_exp(J0_grid, J1_grid, E, D, tau)

    # Main instability curve: 2v0 = J0 (i.e., J0 - 2v0 = 0)
    main_instability_values = saddle_eq(J0_grid, E)

    # Turing curve: J1 = 4v0 (i.e., J1 - 4v0 = 0)
    turing_values = J1_grid - 4 * compute_vplus(J0_grid, E)

    # Plotting


    # Plot the contour for each equation (level sets where the equation equals 0)
    min_J1 = -6.67
    max_J1 = 4.72
    plt.contour(J0_grid, J1_grid, hopf_values, levels=[0], colors='green', label="Hopf curve", linewidths=linewidth)
    plt.contour(J0_grid, J1_grid, turing_hopf_values, levels=[0], colors='purple', label="Turing Hopf curve",linewidths=linewidth)
    plt.contour(J0_grid, J1_grid, main_instability_values, levels=[0], colors='blue', label="Main instability curve",linewidths=linewidth)
    plt.contour(J0_grid, J1_grid, turing_values, levels=[0], colors='red', label="Turing curve",linewidths=linewidth)

    # Labels and title
    plt.xlabel("$J_0$")
    plt.ylabel("$J_1$")
    #plt.title(f"E={E}, $tau$={tau}, and D={D}")
    # Create custom legend handles
    legend_handles = [
        Line2D([0], [0], color='green', lw=2, label='Hopf curve'),
        Line2D([0], [0], color='purple', lw=2, label='Turing Hopf curve'),
        # Line2D([0], [0], color='blue', lw=2, label='Main instability curve'),
        Line2D([0], [0], color='red', lw=2, label='Turing curve')
    ]

    # Add the legend with the custom handles
    #plt.legend(handles=legend_handles)

    return

def plot_del_exp_instabilties_E_lessthan_1(linewidth=1):
    # Constants
    D = 1  # Arbitrary value for D; adjust as needed
    E = -2  # Arbitrary value for E; adjust as needed
    tau = 1

    # Grid of J_0 and J_1 values
    J0_values = np.linspace(5, 8, 200)
    J1_values = np.linspace(-50, 30, 200)

    # Create a meshgrid for contour plotting
    J0_grid, J1_grid = np.meshgrid(J0_values, J1_values)

    # Calculate the value of each equation over the grid
    hopf_values = hopf_eq_del_exp(J0_grid, E, D, tau)
    turing_hopf_values = turing_hopf_eq_del_exp(J0_grid, J1_grid, E, D, tau)

    # Main instability curve: 2v0 = J0 (i.e., J0 - 2v0 = 0)
    main_instability_values = saddle_eq(J0_grid, E)

    # Turing curve: J1 = 4v0 (i.e., J1 - 4v0 = 0)
    turing_values = J1_grid - 4 * compute_vplus(J0_grid, E)


    # Plot the contour for each equation (level sets where the equation equals 0)
    min_J1 = -6.67
    max_J1 = 4.72
    plt.contour(J0_grid, J1_grid, hopf_values, levels=[0], colors='green', label="Hopf curve",linewidths=linewidth)
    plt.contour(J0_grid, J1_grid, turing_hopf_values, levels=[0], colors='purple', label="Turing Hopf curve",linewidths=linewidth)
    plt.contour(J0_grid, J1_grid, main_instability_values, levels=[0], colors='blue', label="Main instability curve",linewidths=linewidth)
    plt.contour(J0_grid, J1_grid, turing_values, levels=[0], colors='red', label="Turing curve",linewidths=linewidth)

    # Labels and title
    plt.xlabel("$J_0$")
    plt.ylabel("$J_1$")
    #plt.title(f"E={E}, $tau$={tau}, and D={D}")
    # Create custom legend handles
    legend_handles = [
        Line2D([0], [0], color='green', lw=2, label='Hopf curve'),
        Line2D([0], [0], color='purple', lw=2, label='Turing Hopf curve'),
        # Line2D([0], [0], color='blue', lw=2, label='Main instability curve'),
        Line2D([0], [0], color='red', lw=2, label='Turing curve')
    ]

    # Add the legend with the custom handles
    #plt.legend(handles=legend_handles)

    return

def plot_alph_func_instabilties_E_gthan_1(linewidth=1):
    ## NOTE: Turing-Hopf Curve will not plot all values with python's plt.contour, so not plotted here
    # computed in Mathematica (with same E, D, and tau as below) and imported in

    # Constants
    D = 1  # Arbitrary value for D; adjust as needed
    E = 2  # Arbitrary value for E; adjust as needed
    tau=1

    # Grid of J_0 and J_1 values
    J0_values = np.linspace(-10, 4, 10000)
    J1_values = np.linspace(-20, 10, 100)

    # Create a meshgrid for contour plotting
    J0_grid, J1_grid = np.meshgrid(J0_values, J1_values)

    # Calculate the value of each equation over the grid
    hopf_values = hopf_eq_alph_func(J0_grid, E, D, tau)
    #turing_hopf_values = turing_hopf_eq_alph_func(J0_grid, J1_grid, E, D, tau)

    # Main instability curve: 2v0 = J0 (i.e., J0 - 2v0 = 0)
    main_instability_values = saddle_eq(J0_grid, E)

    # Turing curve: J1 = 4v0 (i.e., J1 - 4v0 = 0)
    turing_values = J1_grid - 4 * compute_vplus(J0_grid, E)

    #heatmap = plt.contourf(
    #    J0_grid, J1_grid, turing_hopf_values,
    #    levels=100, cmap='coolwarm', alpha=0.6
    #)
    #plt.colorbar(heatmap, label="Turing-Hopf value")
    #this_dir = os.path.dirname(os.path.abspath(__file__))
    #csv_path = os.path.join(this_dir, "turingHopf_zero_contour.csv")
    csv_path = '../current_fig_scripts/more_realistic_cases_data/turingHopf_zero_contour.csv'
    contour = np.loadtxt(csv_path, delimiter=",")
    #contour = np.loadtxt("turingHopf_zero_contour.csv", delimiter=",")
    plt.plot(contour[:, 0], contour[:, 1], color='purple', label="Turing Hopf curve", linewidth=linewidth)

    # Plot the contour for each equation (level sets where the equation equals 0)
    min_J1 = -6.67
    max_J1 = 4.72
    plt.contour(J0_grid, J1_grid, hopf_values, levels=[0], colors='green', label="Hopf curve",linewidths=linewidth)
    #plt.contour(J0_grid, J1_grid, turing_hopf_values, levels=[0], colors='purple', label="Turing Hopf curve")
    plt.contour(J0_grid, J1_grid, main_instability_values, levels=[0], colors='blue', label="Main instability curve",linewidths=linewidth)
    plt.contour(J0_grid, J1_grid, turing_values, levels=[0], colors='red', label="Turing curve", linewidths=linewidth)

    # Labels and title
    plt.xlabel("$J_0$")
    plt.ylabel("$J_1$")
    #plt.title(f"E={E}, $tau$={tau}, and D={D}")
    # Create custom legend handles
    legend_handles = [
        Line2D([0], [0], color='green', lw=2, label='Hopf curve'),
        Line2D([0], [0], color='purple', lw=2, label='Turing Hopf curve'),
        # Line2D([0], [0], color='blue', lw=2, label='Main instability curve'),
        Line2D([0], [0], color='red', lw=2, label='Turing curve')
    ]

    # Add the legend with the custom handles
    #plt.legend(handles=legend_handles)

    return

def plot_alph_func_instabilties_E_lessthan_1(linewidth =1):
    ## NOTE: Turing-Hopf Curve will not plot all values with python's plt.contour, so not plotted here
    # computed in Mathematica (with same E, D, and tau as below) and imported in


    # Constants
    D = 1  # Arbitrary value for D; adjust as needed
    E = -2  # Arbitrary value for E; adjust as needed
    tau = 1

    # Grid of J_0 and J_1 values
    J0_values = np.linspace(5, 8, 500)
    J1_values = np.linspace(-60, 30, 500)

    # Create a meshgrid for contour plotting
    J0_grid, J1_grid = np.meshgrid(J0_values, J1_values)

    # Calculate the value of each equation over the grid
    hopf_values = hopf_eq_alph_func(J0_grid, E, D, tau)
    turing_hopf_values = turing_hopf_eq_alph_func(J0_grid, J1_grid, E, D, tau)

    # Main instability curve: 2v0 = J0 (i.e., J0 - 2v0 = 0)
    main_instability_values = saddle_eq(J0_grid, E)

    # Turing curve: J1 = 4v0 (i.e., J1 - 4v0 = 0)
    turing_values = J1_grid - 4 * compute_vplus(J0_grid, E)

    #heatmap = plt.contourf(
    #    J0_grid, J1_grid, turing_hopf_values,
    #    levels=100, cmap='coolwarm', alpha=0.6
    #)
    #plt.colorbar(heatmap, label="Turing-Hopf value")
    #this_dir = os.path.dirname(os.path.abspath(__file__))
    #csv_path = os.path.join(this_dir, "turingHopf_zero_contour_belowThres.csv")
    csv_path = csv_path = '../current_fig_scripts/more_realistic_cases_data/turingHopf_zero_contour_belowThres.csv'
    contour = np.loadtxt(csv_path, delimiter=",")
    startIdx = 200
    plt.plot(contour[startIdx:, 0], contour[startIdx:, 1], color='purple',  label="Turing Hopf curve",linewidth=linewidth)

    # Plot the contour for each equation (level sets where the equation equals 0)
    min_J1 = -6.67
    max_J1 = 4.72
    plt.contour(J0_grid, J1_grid, hopf_values, levels=[0], colors='green', label="Hopf curve",linewidths=linewidth)
    #plt.contour(J0_grid, J1_grid, turing_hopf_values, levels=[0], colors='purple', label="Turing Hopf curve",linewidths=linewidth)
    plt.contour(J0_grid, J1_grid, main_instability_values, levels=[0], colors='blue', label="Main instability curve",linewidths=linewidth)
    plt.contour(J0_grid, J1_grid, turing_values, levels=[0], colors='red', label="Turing curve",linewidths=linewidth)

    # Labels and title
    plt.xlabel("$J_0$")
    plt.ylabel("$J_1$")
    plt.xlim([5,8])
    plt.ylim([-60,25])

    #plt.title(f"E={E}, $tau$={tau}, and D={D}")
    # Create custom legend handles
    legend_handles = [
        Line2D([0], [0], color='green', lw=2, label='Hopf curve'),
        Line2D([0], [0], color='purple', lw=2, label='Turing Hopf curve'),
        # Line2D([0], [0], color='blue', lw=2, label='Main instability curve'),
        Line2D([0], [0], color='red', lw=2, label='Turing curve')
    ]

    # Add the legend with the custom handles
    #plt.legend(handles=legend_handles)

    # Show plot
    return

def plot_Hopf_curves_varied_D(linewidth=1.5, fontsize=7):
    # this plot the multiple Hopf Curves

    # Define the symbolic variables
    J_var, e = symbols('J e')

    # Define constants
    d_vals = [0.2, 0.5, 1]
    cmap3 = cm.get_cmap('tab20b')
    #d_colors = [cmap3(3/20), cmap3(2/20),cmap3(1/20)]#['#0000FF', '#3399FF', '#66CCFF']#['lightgrey', 'darkgrey', 'black']
    d_colors = ['k', 'k', 'k']
    # Create a list to hold legend handles
    legend_handles = []

    label_x_loc=[-9,-4.5,-3.1]
    label_y_loc=[2.5,2.5,2.5]

    for i in range(0, len(d_vals)):
        dd = d_vals[i]

        # Define the function v
        v = (J_var + sqrt(J_var ** 2 + 4 * (e - J_var))) / 2

        # Define the implicit equation
        implicit_eq = (-1 / dd) * acos(2 * v / J_var) - J_var * sqrt(1 - (2 * v / J_var) ** 2)

        # Convert the implicit equation to a numerical function
        implicit_function = lambdify((J_var, e), implicit_eq, 'numpy')

        # Create a grid of J and e values
        J_vals = np.linspace(-20, -2, 100)
        e_vals = np.linspace(1, 3, 100)
        J_grid, e_grid = np.meshgrid(J_vals, e_vals)

        # Compute the implicit function on the grid
        F_grid = implicit_function(J_grid, e_grid)

        # Plotting the contour
        contour = plt.contour(J_grid, e_grid, F_grid, levels=[0], colors=[d_colors[i]], linewidths=linewidth)

        plt.text(label_x_loc[i], label_y_loc[i], r'$D={}$'.format(d_vals[i]), rotation=-85, fontsize=fontsize, ha='center', va='center')


        # Add a handle for the legend using a proxy artist
        # Add labels along the contour line
        #plt.clabel(contour, inline=False, fontsize=10, fmt=f'$D={dd}$',colors='black', inline_spacing=30)
        #legend_handles.append(plt.Line2D([0], [0], color=d_colors[i], label=f'$D={dd}$'))

    j_vals3 = np.linspace(-4, -2, 10)
    #plt.plot(j_vals3, (3 * j_vals3 ** 2 + 4 * j_vals3) / 4, color=cmap3(0/20),linestyle='-',label=r'$D \rightarrow \infty$')
    plt.plot(j_vals3, (3 * j_vals3 ** 2 + 4 * j_vals3) / 4, color='k',linestyle='--',label=r'$D \rightarrow \infty$', linewidth=linewidth)
    plt.text(-2.25, 2.5, r'$D\gg1$', rotation=-85, fontsize=fontsize, ha='center', va='center')
    plt.plot([-20, 5], [1, 1], 'k', linewidth=linewidth)
    j_vals2 = np.linspace(2, 5, 20)
    plt.plot(j_vals2, 1 - ((j_vals2 - 2) / 2) ** 2, 'k', linewidth=linewidth)

    # Customize legend: remove border and add labels
    #plt.legend(handles=legend_handles + [plt.Line2D([0], [0], color='#0000FF', linestyle='--', label=r'$D \rightarrow \infty$')],
    #           frameon=False, fontsize='small',loc='lower left')

    return


def calculate_E(J_0, J_1):
    return (J_1 ** 2 / 16) - (J_0 * J_1 / 4) + J_0


def plot_turing_curve_varied_J1(labelsize=10, fontsize=10, linewidth=1):
    J_1_values = [4, 6, 10]
    #only plot half of turing bif line version (correct)
    J_0 = np.linspace(-10, 10, 400)  # Range for J_0
    E = np.linspace(-8, 5, 900)  # Range for E
    J_0_grid, E_grid = np.meshgrid(J_0, E)  # Create a grid of J_0 and E values

    label_x_loc=[-2.5, -.5, 2.7]
    label_y_loc=[1.5, 3.2, 3.5]
    rotation_loc=[0,-40,-75]
    i=0

    for J_1 in J_1_values:
        # Calculate 4v_0
        v0 = (J_0_grid + np.sqrt(J_0_grid**2 + 4*(E_grid - J_0_grid))) / 2
        vminus = (J_0_grid - np.sqrt(J_0_grid ** 2 + 4 * (E_grid - J_0_grid))) / 2
        contour = plt.contour(J_0_grid, E_grid, 4 * v0, colors = 'r',levels=[J_1], label=f'J_1 = {J_1}', linewidths=linewidth)
        contour2 = plt.contour(J_0_grid, E_grid, 4 * vminus, colors='r', linestyles='--' ,levels=[J_1], label=f'J_1 = {J_1}', linewidths=linewidth)
        plt.text(label_x_loc[i], label_y_loc[i], r'$J_1={}$'.format(J_1), rotation=rotation_loc[i], fontsize=labelsize, color='r', ha='center',
                 va='center')
        i=i+1

    plt.plot( 2+2*np.sqrt(1-E),E,color='k', linewidth=linewidth)

    return


def plot_Hopf_phase_diag(linewidth=1, fontsize=9, showlabels=1, showSimloc=0):
    # same function as v1 just without the orange square anc blue circle

    # Define the symbolic variables
    J_var, e = symbols('J e')

    # Define constants
    dd= 2
    J_max = 5.5
    cmap3 = cm.get_cmap('Set3')
    #d_colors = cmap3(0 / 12)#['#0000FF', '#3399FF', '#66CCFF']#['lightgrey', 'darkgrey', 'black']

    # Create a list to hold legend handles
    legend_handles = []

    # Define the function v
    v = (J_var + sqrt(J_var ** 2 + 4 * (e - J_var))) / 2

    # Define the implicit equation
    implicit_eq = (-1 / dd) * acos(2 * v / J_var) - J_var * sqrt(1 - (2 * v / J_var) ** 2)

    # Convert the implicit equation to a numerical function
    implicit_function = lambdify((J_var, e), implicit_eq, 'numpy')

    # Create a grid of J and e values
    J_vals = np.linspace(-20, -2, 100)
    e_vals = np.linspace(1, 4, 100)
    J_grid, e_grid = np.meshgrid(J_vals, e_vals)

    # Compute the implicit function on the grid
    F_grid = implicit_function(J_grid, e_grid)

    # Plotting the contour/Hopf Curve
    contour = plt.contour(J_grid, e_grid, F_grid, levels=[0], colors='black', linestyles='--', linewidths=linewidth)

    # Extract the contour lines
    contour_lines = contour.collections[0].get_paths()

    # For each contour path, extract the E and J values (X and Y coordinates of the contour line)
    for contour_path in contour_lines:
        contour_coords = contour_path.vertices
        J_contour = contour_coords[:, 0]
        e_contour = contour_coords[:, 1]

        # Now we will fill the area to the left and right of the contour line
        # Find indices where e > 1 (to only consider above E=1)
        #indices_above_1 = e_contour > 1

        # Plot shaded regions to the left of the contour
        plt.fill_betweenx(e_contour, -5,J_contour, color=cmap3(1 / 12), alpha=0.5)

        # Plot shaded regions to the right of the contour
        plt.fill_betweenx(e_contour,J_contour, J_max, color=cmap3(2 / 12), alpha=0.5)

    # Add a handle for the legend using a proxy artist
    #legend_handles.append(plt.Line2D([0], [0], color=d_colors[0], label=f'$D={dd}$'))

    #j_vals3 = np.linspace(-4, -2, 10)
    #plt.plot(j_vals3, (3 * j_vals3 ** 2 + 4 * j_vals3) / 4, color='#0000FF',linestyle='--',label=r'$D \rightarrow \infty$')
    plt.plot([-20, J_max], [1, 1], 'k', linewidth=linewidth)
    j_vals2 = np.linspace(2, J_max, 20)
    plt.plot(j_vals2, 1 - ((j_vals2 - 2) / 2) ** 2, 'k', linewidth=linewidth)

    e_vals2 = np.linspace(-1, 1, 20)
    # Plot shaded regions to the left of the contour
    plt.fill_betweenx(e_vals2, -5, 2*np.sqrt(1-e_vals2)+2, color=cmap3(3 / 12), alpha=0.5,
                      label="Region F < 0 (Left)")

    # Plot shaded regions to the right of the contour
    plt.fill_betweenx(e_vals2, 2*np.sqrt(1-e_vals2)+2, J_max, color=cmap3(0 / 12), alpha=0.5,
                      label="Region F > 0 (Right)")

    if showSimloc==1:
        non_osc_color = '#1f77b4'  # blue
        osc_color = '#ff7f0e'  # orange
        plt.plot(-4, 3, color=osc_color, marker='o', linestyle=' ')
        plt.plot(-2.5, 3, color=non_osc_color, marker='s')

    if showlabels==1:
        # Plot the text labels
        labelfont=fontsize
        plt.text(2, 2, 'H', fontsize=labelfont, ha='center', va='center', color='k')
        plt.text(0, 0, 'Q', fontsize=labelfont, ha='center', va='center', color='k')
        plt.text(4.3, 0.65, 'Q-H', fontsize=labelfont, ha='center', va='center', color='k')
        plt.text(-4, 2, 'O', fontsize=labelfont, ha='center', va='center', color='k')

        # Plot 'B2' with an arrow pointing to (-2.75, 2)
        plt.annotate('O-H', xy=(-2.75, 2), xytext=(-1, 3), fontsize=labelfont,
                     ha='center', va='center', color='k',
                     arrowprops=dict(facecolor='orange', edgecolor='k', arrowstyle='->'))


    return


def Turing_phase_diag_zoom(labelsize = 10, linewidth = 1):
    maxsteps = 51
    # Load the .mat file
    data = loadmat('spatial_fig_data/fold_cont_data.mat')
    # 'fold_cont_data.mat', 'J0foldgrid', 'Eturinggrid_top', 'Efoldgrid_top', 'Eturinggrid_bottom', 'Efoldgrid_bottom', 'N', 'x', 'L', 'J1', 'num_of_sec_param_steps'

    # Extract relevant variables from the MATLAB structure
    J0 = data['J0foldgrid'].flatten()
    Eturinggrid_top = data['Eturinggrid_top'].flatten()
    Efoldgrid_top = data['Efoldgrid_top'].flatten()
    Eturinggrid_bottom = data['Eturinggrid_bottom'].flatten()
    Efoldgrid_bottom = data['Efoldgrid_bottom'].flatten()
    J1 = data['J1'].flatten()

    # Create the figure and axis
    #fig = plt.figure(layout='constrained')

    # Plot the Saddle curve (green) with label
    j02location = 29
    plt.plot(J0[j02location:maxsteps], 1 - ((J0[j02location:maxsteps] - 2) / 2) ** 2, 'k', label='Saddle-node', linewidth=linewidth)

    # Plot the Fold grids (red) with label
    lower_fld_begin_idx = 17
    fld_idx = 46 #where the folds end
    plt.plot(J0[:fld_idx ], Efoldgrid_top[:fld_idx], 'g', label='Upper Fold', linestyle='dashed', linewidth=linewidth)
    plt.plot(J0[lower_fld_begin_idx:fld_idx ], Efoldgrid_bottom[lower_fld_begin_idx:fld_idx], 'b', linestyle='dashed', label='Lower Fold', linewidth=linewidth)

    # Plot the Turing grids (blue) with label
    saddle_turing_idx = 41
    plt.plot(J0[:saddle_turing_idx], Eturinggrid_top[:saddle_turing_idx], 'r', label=r'Turing($v_+$)', linewidth=linewidth)
    plt.plot(J0[saddle_turing_idx-1:maxsteps], Eturinggrid_top[saddle_turing_idx-1:maxsteps], 'r',linestyle='dotted', label=r'Secondary Instab.($v_-$)', linewidth=linewidth)
    #plt.plot(J0[:maxsteps], Eturinggrid_bottom[:maxsteps], 'b')
    plt.plot(J0[:maxsteps], np.ones(len(J0[:maxsteps])), 'orange', linewidth=linewidth)

    # Add legend to the plot in the lower-left corner
    plt.legend(loc='lower left',fontsize=labelsize,frameon=False)

    #add simulation markers
    plt.plot(3.1,1.75,color='k',marker='o')
    plt.plot(4.2, .5, color='k', marker='^')
    plt.plot(3, .75, color='k', marker='s')

    #add labels
    plt.text(2.5, 1.5, 'L', fontsize=fontsize, ha='center', va='center', color='k')
    plt.text(3.5, -1, 'Q', fontsize=fontsize, ha='center', va='center', color='k')
    plt.text(6, 1.5, 'H', fontsize=fontsize, ha='center', va='center', color='k')
    plt.text(5.9, -0.5, 'Q-H', fontsize=fontsize, ha='center', va='center', color='k')
    plt.text(3, 0.5, 'Q-L', fontsize=fontsize, ha='center', va='center', color='k')
    plt.text(3.6, 1.5, 'L-H', fontsize=fontsize, ha='center', va='center', color='k')
    plt.text(4.6, -0.17, 'L-Q-H', fontsize=fontsize, ha='center', va='center', color='k')

    plt.text(5.12, -1.45, '\u25CB', fontsize=12, fontweight='bold', va='bottom', ha='right')#circle
    plt.text(6.29, -2.26, '\u25A1', fontsize=12, fontweight='bold', va='bottom', ha='right')#square


    # Show the plot
    #plt.show()

    return


def Turing_phase_diag_colors(linewidth=linewidth, fontsize = fontsize, labelsize = labelsize):
    maxsteps = 51
    # Load the .mat file
    data = loadmat('spatial_fig_data/fold_cont_data.mat')
    # 'fold_cont_data.mat', 'J0foldgrid', 'Eturinggrid_top', 'Efoldgrid_top', 'Eturinggrid_bottom', 'Efoldgrid_bottom', 'N', 'x', 'L', 'J1', 'num_of_sec_param_steps'

    # Extract relevant variables from the MATLAB structure
    J0 = data['J0foldgrid'].flatten()
    Eturinggrid_top = data['Eturinggrid_top'].flatten()
    Efoldgrid_top = data['Efoldgrid_top'].flatten()
    Eturinggrid_bottom = data['Eturinggrid_bottom'].flatten()
    Efoldgrid_bottom = data['Efoldgrid_bottom'].flatten()
    J1 = data['J1'].flatten()

    # Create the figure and axis
    #fig = plt.figure(layout='constrained')

    # Plot the Saddle curve
    j02location = 29
    plt.plot(J0[j02location:maxsteps], 1 - ((J0[j02location:maxsteps] - 2) / 2) ** 2, 'k', label='Saddle', linewidth=linewidth)

    # Plot the Fold grids
    lower_fld_begin_idx = 17
    fld_idx = 46 #where the folds end
    plt.plot(J0[:fld_idx ], Efoldgrid_top[:fld_idx], 'k', label='Fold', linestyle='dashed', linewidth=linewidth)
    plt.plot(J0[lower_fld_begin_idx:fld_idx ], Efoldgrid_bottom[lower_fld_begin_idx:fld_idx], 'k', linestyle='dashed', linewidth=linewidth)


    # Plot the Turing grids
    saddle_turing_idx = 41
    plt.plot(J0[:saddle_turing_idx], Eturinggrid_top[:saddle_turing_idx], 'r', label='Turing', linewidth=linewidth)
    plt.plot(J0[saddle_turing_idx-1:maxsteps], Eturinggrid_top[saddle_turing_idx-1:maxsteps], 'r--', label='Turing', linewidth=linewidth)
    #plt.plot(J0[:maxsteps], Eturinggrid_bottom[:maxsteps], 'b')
    plt.plot(np.concatenate((J0[:maxsteps],[10])), np.ones(len(J0[:maxsteps])+1), 'k', linewidth=linewidth)



    # Add legend to the plot
    #plt.legend()

    #stab of E
    color1 = 'red'
    color2 = 'blue'
    color3 = 'gold'
    minE=-10
    jvals = np.linspace(-20,20,len(J0[:maxsteps]))

    plt.fill_between(jvals,np.ones(len(J0[:maxsteps]))*minE,np.ones(len(J0[:maxsteps])),color=color1, alpha=0.4)

    '''# stab of v+
    vplus_lower_bd = np.zeros(len(J0));
    vplus_lower_bd[:saddle_turing_idx] = Eturinggrid_top[:saddle_turing_idx]
    vplus_lower_bd[saddle_turing_idx:] = 1 - ((J0[saddle_turing_idx:] - 2) / 2) ** 2
    maxJ0=10
    plt.fill_between(J0[:maxsteps], vplus_lower_bd[:maxsteps], np.ones(len(J0[:maxsteps]))*maxJ0,color=color2, alpha=0.4)'''

    temp = np.linspace( min(Eturinggrid_top[:saddle_turing_idx]),-10,200)
    Evals = np.concatenate((Eturinggrid_top[:saddle_turing_idx],temp))
    Jvals2 = np.concatenate((J0[:saddle_turing_idx] , (2+2*np.sqrt(1-temp))))
    maxJ0 = 10
    plt.fill_betweenx(Evals, Jvals2, np.ones(len(Jvals2)) * maxJ0, color=color2,alpha=0.4)
    Efoldgrid_bottom[:lower_fld_begin_idx]=1
    plt.fill_between(J0[:fld_idx], Efoldgrid_bottom[:fld_idx], Efoldgrid_top[:fld_idx], color=color3, alpha=0.35)

    #add labels
    plt.text(-1, 3, 'L', fontsize=labelsize, ha='center', va='center', color='k')
    plt.text(0, -3, 'Q', fontsize=labelsize, ha='center', va='center', color='k')
    plt.text(7, 3, 'H', fontsize=labelsize, ha='center', va='center', color='k')
    plt.text(8, -2.8, 'Q-H', fontsize=labelsize, ha='center', va='center', color='k')

    color1 = 'k'
    plt.annotate('Q-L', xy=(3, 0.6), xytext=(2, -.7), fontsize=labelsize-.5,
                 ha='center', va='center', color=color1,
                 arrowprops=dict(facecolor='orange', edgecolor=color1, arrowstyle='->',mutation_scale=5, lw=0.5, shrinkA=0, shrinkB=0))
    plt.annotate('L-H', xy=(3.8, 1.5), xytext=(0, 1.5), fontsize=labelsize-.5,
                 ha='center', va='center', color=color1,
                 arrowprops=dict(facecolor='orange', edgecolor=color1, arrowstyle='->',mutation_scale=5, lw=0.5, shrinkA=0, shrinkB=0))
    plt.annotate('L-Q-H', xy=(4.7, -.7), xytext=(7.7, 0.4), fontsize=labelsize-.5,
                 ha='center', va='center', color=color1,
                 arrowprops=dict(facecolor='orange', edgecolor=color1, arrowstyle='->',mutation_scale=5, lw=0.5, shrinkA=0, shrinkB=0))


    # Show the plot
    #plt.show()



    return


if __name__ == '__main__':
    #plt.figure(figsize=(5, 4))
    plot_delta_instabilties_E_gthan_1()
    #plot_delta_instabilties_E_lessthan_1()
    #plot_del_exp_instabilties_E_gthan_1()
    #plot_del_exp_instabilties_E_lessthan_1()
    #plot_alph_func_instabilties_E_gthan_1()
    #plot_alph_func_instabilties_E_lessthan_1()
    #create_figure_phase_portraits_and_patterns()
    plt.show()


