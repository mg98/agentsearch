import numpy as np
import networkx as nx
from pyvis.network import Network
from agentsearch.graph.types import GraphData

def visualize_graph(data: GraphData, title="Graph Visualization", hide_isolated_nodes=True, output_file="graph.html"):
    """Visualizes the graph with edge colors based on trust score using pyvis."""
    
    # Create a pyvis network with more stable settings
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    net.repulsion(node_distance=200, central_gravity=0.1, spring_length=200, spring_strength=0.05, damping=0.95)
    
    # Disable physics for a static layout
    net.set_options("""
    var options = {
      "physics": {
        "enabled": false
      },
      "layout": {
        "hierarchical": {
          "enabled": false
        }
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      }
    }
    """)
    
    # Get data as numpy arrays
    num_nodes = data.x.shape[0]
    edges = data.edge_index.t().cpu().numpy()
    trust_scores = data.edge_attributes[:, -1].cpu().numpy().flatten()
    
    # Create a NetworkX graph to compute layout
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    
    # Remove isolated nodes if requested
    if hide_isolated_nodes:
        nodes_to_remove = [node for node in G.nodes() if G.degree(node) == 0]
        G.remove_nodes_from(nodes_to_remove)
    
    # Compute layout positions using spring layout with better spacing
    try:
        # Use larger k value for more spacing between nodes
        pos = nx.spring_layout(G, k=3/np.sqrt(len(G.nodes())), iterations=100, seed=42)
    except:
        # Fallback to circular layout if spring layout fails
        pos = nx.circular_layout(G)
    
    # Scale positions to spread across much larger area
    scale_factor = 800  # Much larger scaling
    center_x, center_y = 600, 400  # Center position
    for node in pos:
        pos[node] = (pos[node][0] * scale_factor + center_x, 
                    pos[node][1] * scale_factor + center_y)
    
    # Add nodes with computed positions
    for node_id in G.nodes():
        x, y = pos[node_id]
        net.add_node(int(node_id), 
                    label=f"Node {node_id}", 
                    title=f"Node {node_id}",
                    color="#97c2fc",
                    size=20,
                    x=x, y=y)
    
    # Add edges with colors based on trust scores
    for i, edge in enumerate(edges):
        source, target = int(edge[0]), int(edge[1])
        trust_score = float(trust_scores[i]) if i < len(trust_scores) else 0.0
        
        # Skip if either node is not in our graph (e.g., was removed due to isolation)
        if source not in G.nodes() or target not in G.nodes():
            continue
            
        # Handle NaN values
        if np.isnan(trust_score):
            trust_score = 0.0
            
        # Map trust score to color (0 = red, 1 = green)
        # Convert trust score to RGB color
        red = int(255 * (1 - trust_score))
        green = int(255 * trust_score)
        blue = 50
        color = f"rgb({red},{green},{blue})"
        
        # Map trust score to edge width (0.5 to 3.0)
        width = 0.5 + 2.5 * trust_score
        
        net.add_edge(source, target, 
                    color=color,
                    width=width,
                    title=f"Trust Score: {trust_score:.3f}",
                    arrows={"to": {"enabled": True, "scaleFactor": 1.2}})
    
    # Set the title
    net.heading = title
    
    # Save and return the network
    net.save_graph(output_file)
    print(f"Graph visualization saved to {output_file}")
    return net
