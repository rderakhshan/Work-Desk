import spacy
import networkx as nx
from pyvis.network import Network
import os
import json
from typing import List, Dict, Tuple
from loguru import logger
import tempfile

class GraphRAG:
    """
    GraphRAG implementation for entity extraction, graph building, and visualization.
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize GraphRAG with a spaCy model."""
        try:
            logger.info(f"Loading spaCy model: {model}")
            self.nlp = spacy.load(model)
            self.graph = nx.Graph()
            logger.info("GraphRAG initialized successfully")
        except OSError:
            logger.warning(f"Model {model} not found. Downloading...")
            os.system(f"python -m spacy download {model}")
            self.nlp = spacy.load(model)
            
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from text using spaCy NER.
        Returns a list of (entity_text, entity_label).
        """
        doc = self.nlp(text)
        entities = []
        # Filter for relevant entity types
        relevant_types = ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]
        
        for ent in doc.ents:
            if ent.label_ in relevant_types:
                entities.append((ent.text, ent.label_))
                
        return entities

    def build_graph(self, documents: List[str], clear: bool = False):
        """
        Build a knowledge graph from a list of documents.
        Nodes are entities, edges represent co-occurrence in the same document.
        
        Args:
            documents: List of document texts
            clear: Whether to clear the existing graph before building
        """
        logger.info(f"Building graph from {len(documents)} documents (clear={clear})...")
        if clear:
            self.graph.clear()
        
        for i, doc_text in enumerate(documents):
            entities = self.extract_entities(doc_text)
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(documents)} documents...")
            
            # Add nodes
            for entity, label in entities:
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, label=entity, title=f"{entity} ({label})", group=label)
                else:
                    # Increment frequency/weight
                    current_weight = self.graph.nodes[entity].get('value', 1)
                    self.graph.nodes[entity]['value'] = current_weight + 1

            # Add edges (co-occurrence)
            # Connect all entities found in the same document with labeled relationships
            unique_entities = [(e[0], e[1]) for e in entities]  # Keep entity type info
            for j in range(len(unique_entities)):
                for k in range(j + 1, len(unique_entities)):
                    source_name, source_type = unique_entities[j]
                    target_name, target_type = unique_entities[k]
                    
                    # Create relationship label based on entity types
                    relationship = f"{source_type}-{target_type}"
                    
                    if self.graph.has_edge(source_name, target_name):
                        self.graph[source_name][target_name]['weight'] += 1
                        self.graph[source_name][target_name]['title'] = f"{relationship} (co-occurs {self.graph[source_name][target_name]['weight']}x)"
                    else:
                        self.graph.add_edge(
                            source_name, 
                            target_name, 
                            weight=1,
                            label=relationship,
                            title=f"{relationship} (co-occurs 1x)"
                        )
                        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def visualize_graph(self, output_path: str = "graph.html", max_nodes: int = 100, min_edge_weight: int = 1, mode: str = "2D") -> str:
        """
        Generate an interactive HTML visualization of the graph.
        
        Args:
            output_path: Path to save the HTML file
            max_nodes: Maximum number of nodes to display (most connected first)
            min_edge_weight: Minimum edge weight to display
            mode: Visualization mode - "2D" or "3D"
            
        Returns:
            The path to the generated HTML file.
        """
        if mode == "3D":
            return self.visualize_graph_3d(output_path, max_nodes, min_edge_weight)
        else:
            return self.visualize_graph_2d(output_path, max_nodes, min_edge_weight)
    
    def visualize_graph_2d(self, output_path: str = "graph.html", max_nodes: int = 100, min_edge_weight: int = 1) -> str:
        """Generate 2D visualization using vis.js."""
        if self.graph.number_of_nodes() == 0:
            return ""
            
        try:
            # Create a subgraph for visualization based on filters
            # 1. Filter edges by weight
            filtered_graph = self.graph.copy()
            edges_to_remove = [(u, v) for u, v, d in filtered_graph.edges(data=True) if d.get('weight', 1) < min_edge_weight]
            filtered_graph.remove_edges_from(edges_to_remove)
            
            # 2. Remove isolated nodes after edge filtering (optional, but good for cleanliness)
            # isolated_nodes = list(nx.isolates(filtered_graph))
            # filtered_graph.remove_nodes_from(isolated_nodes)
            
            # 3. Filter nodes by degree (keep top max_nodes)
            if filtered_graph.number_of_nodes() > max_nodes:
                # Calculate degrees
                degrees = dict(filtered_graph.degree())
                # Sort by degree (descending)
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
                # Create subgraph with only top nodes
                filtered_graph = filtered_graph.subgraph(top_nodes).copy()
            
            logger.info(f"Visualizing graph with {filtered_graph.number_of_nodes()} nodes and {filtered_graph.number_of_edges()} edges (Original: {self.graph.number_of_nodes()}/{self.graph.number_of_edges()})")

            # Define a sharp, vibrant color palette
            colors = [
                "#E63946", "#06FFA5", "#4361EE", "#F72585", "#06D6A0", 
                "#FFD60A", "#FF6B35", "#7209B7", "#F15BB5", "#00F5FF",
                "#00B4D8", "#9B5DE5", "#F94144", "#F3722C", "#43AA8B"
            ]
            
            # Map groups to colors
            groups = set(nx.get_node_attributes(filtered_graph, 'group').values())
            group_colors = {group: colors[i % len(colors)] for i, group in enumerate(groups)}

            # Convert NetworkX graph to vis.js format manually
            nodes_data = []
            for node in filtered_graph.nodes():
                node_attrs = filtered_graph.nodes[node]
                # Assign colors based on group/label
                group = node_attrs.get('group', 'default')
                color = group_colors.get(group, '#97C2FC')
                
                nodes_data.append({
                    "id": node,
                    "label": node,
                    "title": node_attrs.get('title', node),
                    "group": group,
                    "value": node_attrs.get('value', 1),
                    "color": color
                })
            
            edges_data = []
            for source, target in filtered_graph.edges():
                edge_attrs = filtered_graph[source][target]
                edges_data.append({
                    "from": source,
                    "to": target,
                    "label": edge_attrs.get('label', ''),
                    "title": edge_attrs.get('title', ''),
                    "width": edge_attrs.get('weight', 1)
                })
            
            # Create custom HTML without loading bar
            html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        #mynetwork {{
            width: 100%;
            height: 100vh;
            background-color: #ffffff !important;
            background: #ffffff !important;
        }}
        .legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.9);
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            z-index: 1000;
            max-height: 90vh;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});
        
        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                font: {{
                    size: 14,
                    color: '#000000',
                    face: 'Segoe UI',
                    strokeWidth: 0,
                    strokeColor: '#ffffff'
                }},
                borderWidth: 2,
                borderWidthSelected: 3,
                shadow: false,
                scaling: {{
                    min: 15,
                    max: 35
                }}
            }},
            edges: {{
                font: {{
                    size: 10,
                    color: '#000000',
                    align: 'middle'
                }},
                color: {{
                    color: '#000000',
                    highlight: '#4361EE',
                    hover: '#4361EE',
                    opacity: 0.3
                }},
                smooth: {{
                    enabled: true,
                    type: 'continuous'
                }},
                width: 0.8,
                arrows: {{
                    to: {{
                        enabled: false
                    }}
                }}
            }},
            physics: {{
                enabled: false,
                stabilization: {{
                    enabled: false
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                hideEdgesOnDrag: true,
                navigationButtons: false,
                keyboard: false
            }},
            layout: {{
                randomSeed: 42,
                improvedLayout: true,
                clusterThreshold: 150
            }}
        }};
        
        
        var network = new vis.Network(container, data, options);
        
        // Fit the network to view without physics boundary
        setTimeout(function() {{
            network.fit({{
                animation: {{
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }}
            }});
        }}, 100);
    </script>
</body>
</html>
"""
            
            # Save to file
            if output_path == "graph.html":
                fd, path = tempfile.mkstemp(suffix=".html")
                os.close(fd)
                output_path = path
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            logger.info(f"Graph visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            return ""
    
    def visualize_graph_3d(self, output_path: str = "graph.html", max_nodes: int = 100, min_edge_weight: int = 1) -> str:
        """Generate 3D visualization using 3d-force-graph library."""
        if self.graph.number_of_nodes() == 0:
            return ""
            
        try:
            # Create a subgraph for visualization based on filters
            # 1. Filter edges by weight
            filtered_graph = self.graph.copy()
            edges_to_remove = [(u, v) for u, v, d in filtered_graph.edges(data=True) if d.get('weight', 1) < min_edge_weight]
            filtered_graph.remove_edges_from(edges_to_remove)
            
            # 2. Filter nodes by degree (keep top max_nodes)
            if filtered_graph.number_of_nodes() > max_nodes:
                degrees = dict(filtered_graph.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
                filtered_graph = filtered_graph.subgraph(top_nodes).copy()
            
            logger.info(f"Visualizing 3D graph with {filtered_graph.number_of_nodes()} nodes and {filtered_graph.number_of_edges()} edges")

            # Define a bright, premium color palette
            colors = [
                "#FF5733", "#33FF57", "#3357FF", "#FF33F6", "#33FFF6", 
                "#F6FF33", "#FF8C33", "#8C33FF", "#FF338C", "#33FF8C",
                "#00C853", "#6200EA", "#D50000", "#AA00FF", "#0091EA"
            ]
            
            # Map groups to colors
            groups = set(nx.get_node_attributes(filtered_graph, 'group').values())
            group_colors = {group: colors[i % len(colors)] for i, group in enumerate(groups)}

            # Convert NetworkX graph to 3d-force-graph format
            nodes_data = []
            for node in filtered_graph.nodes():
                node_attrs = filtered_graph.nodes[node]
                group = node_attrs.get('group', 'default')
                color = group_colors.get(group, '#97C2FC')
                
                nodes_data.append({
                    "id": node,
                    "name": node,
                    "group": group,
                    "val": node_attrs.get('value', 1),
                    "color": color
                })
            
            links_data = []
            for source, target in filtered_graph.edges():
                edge_attrs = filtered_graph[source][target]
                links_data.append({
                    "source": source,
                    "target": target,
                    "value": edge_attrs.get('weight', 1)
                })
            
            # Create 3D HTML using 3d-force-graph
            html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #ffffff;
        }}
        #3d-graph {{
            width: 100vw;
            height: 100vh;
        }}
        .controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            z-index: 1000;
            font-size: 12px;
        }}
        .controls h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #333;
        }}
        .controls p {{
            margin: 5px 0;
            color: #666;
        }}
    </style>
    <script src="https://unpkg.com/3d-force-graph"></script>
</head>
<body>
    <div class="controls">
        <h3>🎮 3D Graph Controls</h3>
        <p><strong>Rotate:</strong> Left click + drag</p>
        <p><strong>Zoom:</strong> Scroll wheel</p>
        <p><strong>Pan:</strong> Right click + drag</p>
        <p><strong>Nodes:</strong> {len(nodes_data)}</p>
        <p><strong>Edges:</strong> {len(links_data)}</p>
    </div>
    <div id="3d-graph"></div>
    <script>
        const graphData = {{
            nodes: {json.dumps(nodes_data)},
            links: {json.dumps(links_data)}
        }};
        
        const Graph = ForceGraph3D()
            (document.getElementById('3d-graph'))
            .graphData(graphData)
            .nodeLabel('name')
            .nodeAutoColorBy('group')
            .enableNavigationControls(true)
            .showNavInfo(false)
            .d3Force('charge').strength(-120)
            .d3Force('link').distance(link => 30 / link.value);
        
        // Camera controls
        Graph.cameraPosition({{ z: 300 }});
        
        // Node click handler
        Graph.onNodeClick(node => {{
            // Look at node
            const distance = 100;
            const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
            Graph.cameraPosition(
                {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
                node,
                1000
            );
        }});
        
        // Auto-rotate
        let angle = 0;
        setInterval(() => {{
            angle += 0.2;
            Graph.cameraPosition({{
                x: 300 * Math.sin(angle * Math.PI / 180),
                z: 300 * Math.cos(angle * Math.PI / 180)
            }});
        }}, 50);
    </script>
</body>
</html>
"""
            
            # Save to file
            if output_path == "graph.html":
                fd, path = tempfile.mkstemp(suffix=".html")
                os.close(fd)
                output_path = path
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            logger.info(f"3D Graph visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error visualizing 3D graph: {e}")
            return ""


    def get_graph_stats(self) -> Dict:
        """Return statistics about the graph."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "components": nx.number_connected_components(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }

    def graph_search(self, query: str, max_hops: int = 2) -> List[str]:
        """
        Search the graph for entities related to the query.
        
        Args:
            query: Search query
            max_hops: Maximum number of hops from query entities
            
        Returns:
            List of related entity names
        """
        if self.graph.number_of_nodes() == 0:
            return []
            
        # Extract entities from query
        query_entities = self.extract_entities(query)
        query_entity_names = [e[0] for e in query_entities]
        
        # Find entities in graph that match query entities
        related_entities = set()
        for entity_name in query_entity_names:
            if self.graph.has_node(entity_name):
                related_entities.add(entity_name)
                
                # Expand to neighbors (graph traversal)
                try:
                    # Get neighbors within max_hops
                    for node in nx.single_source_shortest_path_length(self.graph, entity_name, cutoff=max_hops):
                        related_entities.add(node)
                except nx.NodeNotFound:
                    continue
        
        return list(related_entities)

    def get_entity_context(self, entity: str) -> Dict:
        """
        Get context information about an entity from the graph.
        
        Returns:
            Dictionary with entity metadata, neighbors, and relationships
        """
        if not self.graph.has_node(entity):
            return {}
            
        neighbors = list(self.graph.neighbors(entity))
        node_data = self.graph.nodes[entity]
        
        return {
            "entity": entity,
            "label": node_data.get("label", "UNKNOWN"),
            "frequency": node_data.get("value", 1),
            "neighbors": neighbors,
            "neighbor_count": len(neighbors)
        }

