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
            if (i + 1) % 1000 == 0:
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
        """Generate 2D visualization using vis.js with Flourish-style aesthetics."""
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
            
            logger.info(f"Visualizing graph with {filtered_graph.number_of_nodes()} nodes and {filtered_graph.number_of_edges()} edges")

            # --- Flourish-style Enhancements ---

            # 1. Community Detection for Coloring
            try:
                from networkx.algorithms import community
                # Use greedy modularity communities
                communities = list(community.greedy_modularity_communities(filtered_graph))
                # Create a map of node -> community_id
                community_map = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        community_map[node] = i
            except ImportError:
                logger.warning("NetworkX community module not found, falling back to connected components")
                community_map = {node: 0 for node in filtered_graph.nodes()}

            # 2. Dynamic Node Sizing based on Degree
            degrees = dict(filtered_graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            min_degree = min(degrees.values()) if degrees else 1
            
            # Define a premium color palette (Flourish-inspired)
            colors = [
                "#E63946", "#06FFA5", "#4361EE", "#F72585", "#06D6A0", 
                "#FFD60A", "#FF6B35", "#7209B7", "#F15BB5", "#00F5FF",
                "#00B4D8", "#9B5DE5", "#F94144", "#F3722C", "#43AA8B"
            ]
            
            # Convert NetworkX graph to vis.js format
            nodes_data = []
            for node in filtered_graph.nodes():
                node_attrs = filtered_graph.nodes[node]
                
                # Color by community
                comm_id = community_map.get(node, 0)
                color = colors[comm_id % len(colors)]
                
                # Size by degree (Linear interpolation: min_size=15, max_size=45)
                degree = degrees.get(node, 0)
                # Normalize degree between 0 and 1
                norm_degree = (degree - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 0.5
                size = 15 + (norm_degree * 30)

                nodes_data.append({
                    "id": node,
                    "label": node,
                    "title": f"{node} (Connections: {degree})",
                    "group": comm_id,
                    "value": size, # vis.js uses 'value' for scaling if enabled, or we can set 'size' directly
                    "size": size,
                    "color": color,
                    "font": {"size": 14 + (norm_degree * 10)} # Scale font with node
                })
            
            edges_data = []
            for source, target in filtered_graph.edges():
                edge_attrs = filtered_graph[source][target]
                edges_data.append({
                    "from": source,
                    "to": target,
                    "title": f"Co-occurs {edge_attrs.get('weight', 1)}x",
                    "width": edge_attrs.get('weight', 1) * 0.5 # Thinner edges
                })
            
            # Create custom HTML
            html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; }}
        #mynetwork {{ width: 100%; height: 100vh; background-color: #ffffff !important; }}
    </style>
</head>
<body>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});
        var container = document.getElementById('mynetwork');
        var data = {{ nodes: nodes, edges: edges }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                font: {{
                    face: 'Segoe UI',
                    color: '#333333',
                    strokeWidth: 2,
                    strokeColor: '#ffffff'
                }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                color: {{
                    color: '#cccccc',
                    highlight: '#4361EE',
                    hover: '#4361EE',
                    opacity: 0.3
                }},
                smooth: {{ enabled: true, type: 'continuous', roundness: 0.5 }},
                selectionWidth: 2,
                hoverWidth: 1.5
            }},
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.08,
                    damping: 0.4,
                    avoidOverlap: 0.5
                }},
                stabilization: {{
                    enabled: true,
                    iterations: 1000,
                    updateInterval: 25
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                hideEdgesOnDrag: true
            }},
            layout: {{
                randomSeed: 42
            }}
        }};
        
        var network = new vis.Network(container, data, options);
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
        """Generate 3D visualization using 3d-force-graph library with Flourish-style aesthetics."""
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

            # --- Flourish-style Enhancements ---

            # 1. Community Detection for Coloring
            try:
                from networkx.algorithms import community
                communities = list(community.greedy_modularity_communities(filtered_graph))
                community_map = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        community_map[node] = i
            except ImportError:
                community_map = {node: 0 for node in filtered_graph.nodes()}

            # 2. Dynamic Node Sizing based on Degree
            degrees = dict(filtered_graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            min_degree = min(degrees.values()) if degrees else 1

            # Define a bright, premium color palette
            colors = [
                "#FF5733", "#33FF57", "#3357FF", "#FF33F6", "#33FFF6", 
                "#F6FF33", "#FF8C33", "#8C33FF", "#FF338C", "#33FF8C",
                "#00C853", "#6200EA", "#D50000", "#AA00FF", "#0091EA"
            ]
            
            # Convert NetworkX graph to 3d-force-graph format
            nodes_data = []
            for node in filtered_graph.nodes():
                node_attrs = filtered_graph.nodes[node]
                
                # Color by community
                comm_id = community_map.get(node, 0)
                color = colors[comm_id % len(colors)]
                
                # Size by degree
                degree = degrees.get(node, 0)
                norm_degree = (degree - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 0.5
                size = 5 + (norm_degree * 15) # Base size 5, max extra 15

                nodes_data.append({
                    "id": node,
                    "name": node,
                    "group": comm_id,
                    "val": size,
                    "color": color,
                    "desc": f"Connections: {degree}"
                })
            
            links_data = []
            for source, target in filtered_graph.edges():
                edge_attrs = filtered_graph[source][target]
                links_data.append({
                    "source": source,
                    "target": target,
                    "value": edge_attrs.get('weight', 1)
                })
            
            # Create 3D HTML using 3d-force-graph with advanced visual effects
            html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; background-color: #000003; }}
        #3d-graph {{ width: 100vw; height: 100vh; }}
        .controls {{
            position: absolute; top: 10px; left: 10px;
            background: rgba(0, 0, 0, 0.6); backdrop-filter: blur(10px);
            padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
            color: #e0e0e0; font-size: 12px; pointer-events: none;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        .controls h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #fff; font-weight: 600; }}
    </style>
    <script src="https://unpkg.com/3d-force-graph"></script>
    <script src="https://unpkg.com/three"></script>
    <script src="https://unpkg.com/three-spritetext"></script>
    <script src="https://unpkg.com/three/examples/jsm/postprocessing/EffectComposer.js"></script>
    <script src="https://unpkg.com/three/examples/jsm/postprocessing/RenderPass.js"></script>
    <script src="https://unpkg.com/three/examples/jsm/postprocessing/UnrealBloomPass.js"></script>
</head>
<body>
    <div class="controls">
        <h3>🌌 Neural Graph 3D</h3>
        <p>Left click: Rotate | Right click: Pan | Scroll: Zoom</p>
        <p>Nodes: {len(nodes_data)} | Edges: {len(links_data)}</p>
    </div>
    <div id="3d-graph"></div>
    <script>
        const graphData = {{
            nodes: {json.dumps(nodes_data)},
            links: {json.dumps(links_data)}
        }};

        // Calculate degree for each node to determine label visibility
        const nodeDegrees = {{}};
        graphData.links.forEach(link => {{
            nodeDegrees[link.source] = (nodeDegrees[link.source] || 0) + 1;
            nodeDegrees[link.target] = (nodeDegrees[link.target] || 0) + 1;
        }});

        const Graph = ForceGraph3D()
            (document.getElementById('3d-graph'))
            .graphData(graphData)
            .nodeColor('color')
            .nodeVal('val')
            .nodeResolution(24) // Smoother spheres
            .nodeOpacity(0.9)
            .linkWidth(link => 0.3)
            .linkOpacity(0.2)
            .linkDirectionalParticles(2) // Add particles for "flow"
            .linkDirectionalParticleWidth(1.5)
            .linkDirectionalParticleSpeed(0.005)
            .backgroundColor('#000003')
            .showNavInfo(false)
            
            // Custom Node Object (Text Sprite)
            .nodeThreeObject(node => {{
                // Only show labels for important nodes (degree > 2) or if graph is small
                const degree = nodeDegrees[node.id] || 0;
                if (degree < 2 && graphData.nodes.length > 50) return null;

                const sprite = new SpriteText(node.name);
                sprite.color = node.color;
                sprite.textHeight = 4 + (node.val / 5); // Scale text with node size
                sprite.padding = 2;
                sprite.backgroundColor = 'rgba(0,0,0,0.5)';
                sprite.borderRadius = 4;
                return sprite;
            }})
            .nodeThreeObjectExtend(true) // Draw sphere AND text

            // Physics Tuning
            .d3Force('charge', d3.forceManyBody().strength(-150)) // Stronger repulsion
            .d3Force('link', d3.forceLink().distance(link => 60).strength(0.1)) // Longer links
            
            // Post-Processing (Bloom)
            .onEngineStop(() => Graph.zoomToFit(400)); // Auto-fit on settle

        // Add Bloom Effect
        const bloomPass = new THREE.UnrealBloomPass();
        bloomPass.strength = 1.5;
        bloomPass.radius = 0.4;
        bloomPass.threshold = 0.1;
        Graph.postProcessingComposer().addPass(bloomPass);

        // Camera Orbit
        let angle = 0;
        setInterval(() => {{
            angle += 0.0005;
            Graph.cameraPosition({{
                x: 500 * Math.sin(angle),
                z: 500 * Math.cos(angle)
            }});
        }}, 10);
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

