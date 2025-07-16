import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from agentsearch.graph.types import GraphData

def visualize_graph(data: GraphData, title="Graph Visualization"):
    """Visualizes the graph with edge colors based on trust score."""
    g = nx.DiGraph() # Use DiGraph for directed edges

    # Add nodes
    g.add_nodes_from(range(data.x.shape[0]))

    # Add edges and trust scores as attributes
    edges = data.edge_index.t().cpu().numpy()
    trust_scores = data.edge_trust_score.cpu().numpy().flatten()
    g.add_edges_from(edges)

    # Use trust scores for edge colors, ensuring it's a flattened float array and handling potential NaNs
    edge_colors = trust_scores.astype(float).flatten()
    # Replace any non-finite values (like NaNs) with a default value (e.g., 0.0)
    edge_colors[np.isnan(edge_colors)] = 0.0

    # Create a ScalarMappable to map trust scores to colors
    cmap = cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([]) # Or pass the actual data: sm.set_array(edge_colors)

    # Map the edge_colors data to RGBA values using the ScalarMappable
    rgba_colors = sm.to_rgba(edge_colors)

    # Explicitly create figure and axes
    fig, ax = plt.subplots(figsize=(12, 12))

    pos = nx.spring_layout(g, seed=42) # for reproducible layout
    nx.draw_networkx_nodes(g, pos, node_color='lightblue', node_size=200, ax=ax)
    edges = nx.draw_networkx_edges(g, pos, edge_color=rgba_colors, # Use the mapped RGBA colors
                                   arrows=True, arrowstyle='->', arrowsize=10, width=1.5, ax=ax)

    # Add a colorbar, explicitly linked to the axes
    cbar = plt.colorbar(sm, shrink=0.8, ax=ax) # Use the ScalarMappable for the colorbar and link to axes
    cbar.set_label('Edge Trust Score', rotation=270, labelpad=15)

    ax.set_title(title, fontsize=16)
    plt.show()
