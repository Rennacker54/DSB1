###############################################################################
# Functions for generating toy networks
###############################################################################

# helper functions
def plot_graph(G, labels=False):
    """
    Plots the network G with node and edge labels
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)

    if labels:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()


import numpy as np

import numpy as np

def normalize_weights(G):
    """
    Adjusts the edge weights such that the weights of the incoming edges for each node sum to 1.
    """
    for node in G.nodes():
        predecessors = list(G.predecessors(node))
        if predecessors:
            total_weight = sum(G[predecessor][node].get('weight', 1) for predecessor in predecessors)
            for predecessor in predecessors:
                G[predecessor][node]['weight'] /= total_weight
    return G


def randomize_weights(G):
    """
    Randomizes the edge weights of the network.
    """
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = np.random.rand()

    G = normalize_weights(G)

    return G


# toy functions
def horizontal_toy_network(n_nodes):
    """
    Generates directed graph without connections with n nodes
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    return G


def circle_toy_network(n_nodes):
    """
    Generates directed graph with connections in a circle with n nodes
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        G.add_edge(i, (i + 1) % n_nodes, weight=1.0)

    return G


def vertical_toy_network(n_nodes):
    """
    Generates directed graph with connections in a line with n nodes
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    for i in range(n_nodes - 1):
        G.add_edge(i, i + 1, weight=1.0)

    return G


# unidirectional star network
def star_toy_network(n_nodes):
    """
    Generates directed graph with connections from a central node with n nodes
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    for i in range(1, n_nodes):
        G.add_edge(0, i, weight=1.0)

    return G

## bi-directional star network
# def star_toy_network(n_nodes):
#     """
#     Generates a bi-directional graph with connections from a central node to n nodes.
#     """
#     import networkx as nx
#
#     G = nx.DiGraph()
#     G.add_nodes_from(range(n_nodes))
#
#     for i in range(1, n_nodes):
#         # Add edges from the central hub (node 0) to other nodes
#         G.add_edge(0, i, weight=1.0)
#         # Add edges from other nodes back to the central hub
#         G.add_edge(i, 0, weight=1.0)
#
#     return G



def full_toy_network(n_nodes):
    """
    Generates directed graph with connections from each node to all other nodes with n nodes
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                G.add_edge(i, j, weight=1.0)

    return G


def tree_toy_network(n_nodes, n_parents=2):
    """
    Generates a tree with n_nodes and n_parents
    """
    import networkx as nx
    import math

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    for node in G.nodes():
        if node != 0:
            parent = math.floor((node - 1) / n_parents)
            G.add_edge(node, parent, weight=1.0)

    return G


# random networks
def random_network(n_nodes, p=.15):
    """
    Generates a random graph with n_nodes and probability p
    """
    import networkx as nx
    import numpy as np

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and np.random.rand() < p:
                G.add_edge(i, j, weight=1.0)

    return G


def random_tree_network(n_nodes, p=.15, n_parents=2):
    """
    Generates a tree with n_parents. At a probability p, there exist links between nodes of different branches.
    """
    import numpy as np

    G = tree_toy_network(n_nodes, n_parents)

    for i in range(n_nodes):
        if np.random.rand() < p:
            potential_targets = [j for j in range(n_nodes) if j != i]
            target = np.random.choice(potential_targets)
            G.add_edge(i, target, weight=1.0)

    return G


def directed_barabasi_albert_network(n_nodes, m=2, delete_fraction=0.4):
    """
    Generates a directed graph using the preferential attachment model and deletes a fraction of edges.

    Parameters:
    - n_nodes: Number of nodes in the graph.
    - m: Number of edges to attach from a new node to existing nodes.
    - delete_fraction: Fraction of edges to delete randomly.

    Returns:
    - G: A directed graph with preferential attachment.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(m))  # Start with m initial nodes

    # Add initial edges to form a connected graph
    for i in range(m):
        for j in range(i + 1, m):
            G.add_edge(i, j, weight=1.0)

    # Add remaining nodes
    for new_node in range(m, n_nodes):
        targets = set()
        while len(targets) < m:
            # Calculate the probability of connecting to existing nodes
            node_probabilities = np.array(
                [G.in_degree(node) + 1 for node in G.nodes()])
            node_probabilities = node_probabilities / node_probabilities.sum()
            target = np.random.choice(G.nodes(), p=node_probabilities)
            targets.add(target)

        for target in targets:
            G.add_edge(new_node, target, weight=1.0)

    # Randomly delete a fraction of all edges
    edges = list(G.edges())
    np.random.shuffle(edges)
    num_edges_to_delete = int(delete_fraction * len(edges))
    edges_to_delete = edges[:num_edges_to_delete]
    G.remove_edges_from(edges_to_delete)

    return G

# assign network properties
def assign_productivity(G):
    """
    Assigns a productivity value of 1 to each node in the graph.
    """
    import networkx as nx

    nx.set_node_attributes(G, 1, 'productivity')

    return G


def get_total_productivity(G):
    """
    Returns the average productivity of all nodes in the graph.
    """
    import numpy as np

    # Initialize a list to store the productivities of all nodes
    productivities = [G.nodes[node]['productivity'] for node in G.nodes()]

    # Calculate the average productivity of all nodes
    if productivities:
        return sum(productivities) / len(productivities)
    else:
        return 0  # Return 0 if no nodes are found (should not happen in a non-empty graph)


def calculate_volatility(total_productivity_list):
    """
    Calculate the volatility as the standard deviation of log outputs.
    """
    import numpy as np
    log_productivity = np.log(total_productivity_list)
    return np.std(log_productivity)


# # propagation
# def propagation_step(G, use_min=True):
#     """
#     Performs a single step of productivity propagation through the network.
#
#     The new productivity of a node is calculated as the minimum or maximum of the productivities
#     of its incoming neighbors, each weighted by the corresponding edge weight.
#
#     If a node has no incoming neighbors, its productivity remains unchanged.
#
#     Parameters:
#     - G: The graph.
#     - use_min: If True, use the minimum of the incoming productivities. If False, use the maximum.
#     """
#     new_productivity = {}
#
#     for node in G.nodes():
#         # Get predecessors (incoming neighbors)
#         predecessors = list(G.predecessors(node))
#
#         if predecessors:  # If the node has incoming neighbors
#             # Compute the weighted productivity of all incoming edges
#             incoming_productivities = [
#                 G.nodes[neighbor]['productivity'] * G[neighbor][node]['weight']
#                 for neighbor in predecessors
#             ]
#             # The new productivity is the minimum or maximum of the incoming productivities
#             if use_min:
#                 new_productivity[node] = min(incoming_productivities)
#             else:
#                 new_productivity[node] = sum(incoming_productivities)/len(incoming_productivities)
#         else:
#             # If no predecessors, retain the current productivity
#             new_productivity[node] = G.nodes[node]['productivity']
#
#     # Update the graph with the new productivity values
#     for node in G.nodes():
#         G.nodes[node]['productivity'] = new_productivity[node]
#
#     return G

def propagation_step(G, use_min=True):
    """
    Performs a single step of productivity propagation through the network.

    The new productivity of a node is calculated as the minimum or maximum of the productivities
    of its incoming neighbors, each weighted by the corresponding edge weight.

    If a node has no incoming neighbors, its productivity remains unchanged.

    Parameters:
    - G: The graph.
    - use_min: If True, use the minimum of the incoming productivities. If False, use the maximum.
    """
    # Create a dictionary to store the current productivities
    current_productivities = {node: G.nodes[node]['productivity'] for node in G.nodes()}
    new_productivity = {}

    for node in G.nodes():
        # Get predecessors (incoming neighbors)
        predecessors = list(G.predecessors(node))

        if predecessors:  # If the node has incoming neighbors
            # Compute the weighted productivity of all incoming edges using current productivities
            incoming_productivities = [
                current_productivities[neighbor] * G[neighbor][node]['weight']
                for neighbor in predecessors
            ]
            # The new productivity is the minimum or maximum of the incoming productivities
            if use_min:
                new_productivity[node] = min(incoming_productivities)
            else:
                new_productivity[node] = sum(incoming_productivities) / len(incoming_productivities)
        else:
            # If no predecessors, retain the current productivity
            new_productivity[node] = current_productivities[node]

    # Update the graph with the new productivity values
    for node in G.nodes():
        G.nodes[node]['productivity'] = new_productivity[node]

    return G



###############################################################################
# 03_Network_Measures.py
###############################################################################
# basic statistics
def get_n(G):
    """
    Returns the number of nodes in the network G.
    """
    return G.number_of_nodes()


def get_m(G):
    """
    Returns the number of edges in the network G.
    """
    return G.number_of_edges()


def get_mean_degree(G):
    """
    Returns the average degree of the nodes in the network G.
    """
    return sum(G.out_degree(node) for node in G.nodes()) / G.number_of_nodes()


def get_mean_l(G):
    """
    Returns the average shortest path length of the network G.
    """
    return nx.average_shortest_path_length(G)


def get_alpha(G):
    """
    Returns the alpha coefficients for out-degree and in-degree if the distribution of the network G is a power law.
    Returns None if the power law does not fit.
    """
    import powerlaw
    try:
        out_degree_values = list(dict(G.out_degree()).values())
        out_fit = powerlaw.Fit(out_degree_values)
        out_alpha = out_fit.alpha
    except Exception:
        out_alpha = None

    try:
        in_degree_values = list(dict(G.in_degree()).values())
        in_fit = powerlaw.Fit(in_degree_values)
        in_alpha = in_fit.alpha
    except Exception:
        in_alpha = None

    return


def get_diameter(G):
    """
    Returns longest shortest path length in the network G.
    """
    if nx.is_strongly_connected(G):
        return nx.diameter(G)
    else:
        return None


def plot_indegree_distribution(G):
    """
    Plots the in-degree distribution of the network G as a probability density function.
    """
    in_degrees = list(dict(G.in_degree()).values())
    plt.figure()
    plt.hist(in_degrees, bins=range(max(in_degrees) + 1), density=True, alpha=0.75)
    plt.xlabel('In-degree')
    plt.ylabel('Probability Density')
    plt.title('In-degree Distribution (PDF)')
    plt.show()

def plot_outdegree_distribution(G):
    """
    Plots the out-degree distribution of the network G as a probability density function.
    """
    out_degrees = list(dict(G.out_degree()).values())
    plt.figure()
    plt.hist(out_degrees, bins=range(max(out_degrees) + 1), density=True, alpha=0.75)
    plt.xlabel('Out-degree')
    plt.ylabel('Probability Density')
    plt.title('Out-degree Distribution (PDF)')
    plt.show()


def get_eigenvec_centrality(G):
    """
    Returns the eigenvalue centrality of the network G.
    """
    return nx.eigenvector_centrality(G)


def get_global_clustering_coefficient(G):
    """
    Returns the global clustering coefficient of the network G.
    """
    return nx.transitivity(G)


def degree_centrality(G):
    """
    Returns the degree centrality of the network G.
    """
    return nx.degree_centrality(G)


import networkx as nx
import matplotlib.pyplot as plt
import powerlaw

def analyze_network(G):
    """
    Analyzes the network G and returns various metrics and plots.
    """
    results = {}

    # Number of nodes
    results['number_of_nodes'] = G.number_of_nodes()

    # Number of edges
    results['number_of_edges'] = G.number_of_edges()

    # Mean degree
    results['mean_degree'] = sum(G.out_degree(node) for node in G.nodes()) / G.number_of_nodes()

    # Average shortest path length
    if nx.is_strongly_connected(G):
        results['average_shortest_path_length'] = nx.average_shortest_path_length(G)
    else:
        results['average_shortest_path_length'] = None

    # Alpha coefficients for out-degree and in-degree
    try:
        out_degree_values = list(dict(G.out_degree()).values())
        out_fit = powerlaw.Fit(out_degree_values)
        results['out_alpha'] = out_fit.alpha
    except Exception:
        results['out_alpha'] = None

    try:
        in_degree_values = list(dict(G.in_degree()).values())
        in_fit = powerlaw.Fit(in_degree_values)
        results['in_alpha'] = in_fit.alpha
    except Exception:
        results['in_alpha'] = None

    # Diameter
    if nx.is_strongly_connected(G):
        results['diameter'] = nx.diameter(G)
    else:
        results['diameter'] = None

    # Plot in-degree distribution
    # in_degrees = dict(G.in_degree()).values()
    # plt.figure()
    # plt.hist(in_degrees, bins=range(max(in_degrees) + 1))
    # plt.xlabel('In-degree')
    # plt.ylabel('Frequency')
    # plt.title('In-degree distribution')
    # plt.show()

    # Plot out-degree distribution
    # out_degrees = dict(G.out_degree()).values()
    # plt.figure()
    # plt.hist(out_degrees, bins=range(max(out_degrees) + 1))
    # plt.xlabel('Out-degree')
    # plt.ylabel('Frequency')
    # plt.title('Out-degree distribution')
    # plt.show()

    # Eigenvector centrality
    results['eigenvector_centrality'] = nx.eigenvector_centrality(G)
    results['max_eigenvector_centrality'] = max(results['eigenvector_centrality'].values())
    results['sum_eigenvector_centrality'] = sum(results['eigenvector_centrality'].values())
    results['abs_difference_eigenvector_centrality'] = max(results['eigenvector_centrality'].values()) - min(results['eigenvector_centrality'].values())

    # Global clustering coefficient
    results['global_clustering_coefficient'] = nx.transitivity(G)

    # Degree centrality
    results['degree_centrality'] = nx.degree_centrality(G)
    results['max_degree_centrality'] = max(results['degree_centrality'].values())
    results['sum_degree_centrality'] = sum(results['degree_centrality'].values())
    results['abs_difference_degree_centrality'] = max(results['degree_centrality'].values()) - min(results['degree_centrality'].values())

    # Eigenvector centrality spread


    return results


