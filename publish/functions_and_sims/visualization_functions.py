import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from publish.functions_and_sims.spiking_network_functions import generate_spatial_connectivity_mat

def plot_network_graph_diagram(g_bar):

    # Create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(g_bar, create_using=nx.Graph)

    # Create a spring layout for the graph
    #pos = nx.spring_layout(G,k=50)
    pos = nx.circular_layout(G)

    # Draw the graph
    #nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=50, edge_color='gray', linewidths=0)
    nx.draw(G, pos, with_labels=False, node_color='black', node_size=80, alpha=0.5,edge_color='gray', linewidths=0)
    return

def add_subplot_label(fig, ax, label, x_offset=0, y_offset= 0, fontsize=12):
    # Get axes position in figure coordinates
    bbox = ax.get_position()
    x = bbox.x0 + x_offset
    y = bbox.y1 + y_offset
    fig.text(x, y, label, fontsize=fontsize, fontweight='bold', va='bottom', ha='left')

def plot_delta_function():
    # Define the Dirac delta function approximation
    def dirac_delta(t, t_star, D, width=0.01):
        return np.where(np.abs(t - (t_star + D)) < width, 1 / (2 * width), 0)

    # Set values for t*, D, and the time range
    t_star = 3.33  # Example t*
    D = 3.33  # Example D
    t_min = 0  # Min value for t axis
    t_max = 10  # Max value for t axis
    t_vals = np.linspace(t_min, t_max, 1000)  # Time points for plotting

    # Compute the Dirac delta approximation
    v_vals = 1+dirac_delta(t_vals, t_star, D)

    # Plotting
    fig = plt.figure(figsize=(4, 3))
    plt.plot(t_vals, v_vals, label=r'$\delta(t - t^* - D)$', color='blue')

    # Labeling the axes
    plt.xlabel('Time(t)')
    plt.ylabel('Voltage(v)')

    # Set only t* and t* + D as tick marks on the t-axis
    plt.xticks([t_star, t_star + D], labels=[r'$t^*$', r'$t^* + D$'])

    # Set no tick marks on the y-axis (empty list for yticks)
    plt.yticks([])

    # Set limits for better visualization
    plt.ylim(0, 60)
    plt.xlim(t_min, t_max)
    sns.despine(fig)
    # Title
    #plt.title(r'Plot of $v = \delta(t - t^* - D)$')
    return

def plot_delayed_exp_function(linewidth=1):

    # Set values for t*, D, and the time range
    tau = 1
    J=0.5
    t_star = 1  # Example t*
    D = 3  # Example D
    t_min = 0  # Min value for t axis
    t_max = 10  # Max value for t axis
    t_vals = np.linspace(t_min, t_max, 1000)  # Time points for plotting

    # Compute the Dirac delta approximation
    v_vals = ((J/tau)*np.exp(-(t_vals- (t_star + D))/tau))*np.heaviside(t_vals-(t_star + D),1)

    # Plotting
    #fig = plt.figure(figsize=(4, 3))
    plt.plot(t_vals, v_vals, label=r'$\delta(t - t^* - D)$', color='black', linewidth=linewidth)

    # Labeling the axes
    plt.xlabel('Time(t)')
    plt.ylabel('Voltage(v)')

    # Set only t* and t* + D as tick marks on the t-axis
    plt.xticks([t_star, t_star + D], labels=[r'$t^*$', r'$t^* + D$'])

    # Set no tick marks on the y-axis (empty list for yticks)
    plt.yticks([])

    # Set limits for better visualization
    plt.ylim(0, 0.5)
    plt.xlim(t_min, t_max)
    #sns.despine(fig)
    # Title
    #plt.title(r'Plot of $v = \delta(t - t^* - D)$')
    return

def plot_alpha_function(linewidth=1):

    # Set values for t*, D, and the time range
    tau = 1
    J=1
    t_star = 1  # Example t*
    D = 3  # Example D
    t_min = 0  # Min value for t axis
    t_max = 10  # Max value for t axis
    t_vals = np.linspace(t_min, t_max, 1000)  # Time points for plotting

    # Compute the Dirac delta approximation
    v_vals = (J/tau**2)*(t_vals - (t_star + D))*np.exp(-(t_vals- (t_star + D))/tau)

    # Plotting
    #fig = plt.figure(figsize=(4, 3))
    plt.plot(t_vals, v_vals, label=r'$\delta(t - t^* - D)$', color='black', linewidth=linewidth)

    # Labeling the axes
    plt.xlabel('Time(t)')
    plt.ylabel('Voltage(v)')

    # Set only t* and t* + D as tick marks on the t-axis
    plt.xticks([t_star, t_star + D], labels=[r'$t^*$', r'$t^* + D$'])

    # Set no tick marks on the y-axis (empty list for yticks)
    plt.yticks([])

    # Set limits for better visualization
    plt.ylim(0, 0.5)
    plt.xlim(t_min, t_max)
    #sns.despine(fig)
    # Title
    #plt.title(r'Plot of $v = \delta(t - t^* - D)$')
    return

if __name__ == '__main__':
    N = 15  # Number of neurons
    g=4
    connection_prob = 0.5
    #g_bar = np.random.binomial(n=1, p=connection_prob, size=(N, N)) * g / connection_prob / N
    #np.fill_diagonal(g_bar, 0)  # make sure connection =0 if i=j

    g_bar = generate_spatial_connectivity_mat(1,1,N)
    np.fill_diagonal(g_bar, 0)  # make sure connection =0 if i=j

    plt.figure(figsize=(4, 4))
    plot_network_graph_diagram(g_bar)
    plt.title("Force-Directed Graph of Neuron Connections")
    plt.show()