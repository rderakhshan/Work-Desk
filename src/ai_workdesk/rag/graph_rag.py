import spacy
import networkx as nx
from pyvis.network import Network
import os
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

    def build_graph(self, documents: List[str]):
        """
        Build a knowledge graph from a list of documents.
        Nodes are entities, edges represent co-occurrence in the same document.
        """
        logger.info(f"Building graph from {len(documents)} documents...")
        self.graph.clear()
        
        for i, doc_text in enumerate(documents):
            entities = self.extract_entities(doc_text)
            
            # Add nodes
            for entity, label in entities:
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, label=label, title=f"{entity} ({label})", group=label)
                else:
                    # Increment frequency/weight
                    current_weight = self.graph.nodes[entity].get('value', 1)
                    self.graph.nodes[entity]['value'] = current_weight + 1

            # Add edges (co-occurrence)
            # Connect all entities found in the same document
            unique_entities = list(set([e[0] for e in entities]))
            for j in range(len(unique_entities)):
                for k in range(j + 1, len(unique_entities)):
                    source = unique_entities[j]
                    target = unique_entities[k]
                    
                    if self.graph.has_edge(source, target):
                        self.graph[source][target]['weight'] += 1
                    else:
                        self.graph.add_edge(source, target, weight=1)
                        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def visualize_graph(self, output_path: str = "graph.html") -> str:
        """
        Generate an interactive HTML visualization of the graph.
        Returns the path to the generated HTML file.
        """
        if self.graph.number_of_nodes() == 0:
            return ""
            
        try:
            net = Network(height="600px", width="100%", bgcolor="#1e1e1e", font_color="white", notebook=False)
            net.from_nx(self.graph)
            
            # Physics options for better layout
            net.set_options("""
            var options = {
              "nodes": {
                "font": {
                  "size": 16
                }
              },
              "edges": {
                "color": {
                  "inherit": true
                },
                "smooth": false
              },
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 100,
                  "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                  "enabled": true,
                  "iterations": 1000
                }
              }
            }
            """)
            
            # Save to temporary file if no path provided, or use specific path
            # For Gradio, we might want to return the HTML content or a path
            # Let's save to a temp file to be safe
            if output_path == "graph.html":
                 fd, path = tempfile.mkstemp(suffix=".html")
                 os.close(fd)
                 output_path = path

            net.save_graph(output_path)
            logger.info(f"Graph visualization saved to {output_path}")
            
            # Read the content to return (optional, but Gradio HTML component might need content)
            # Or we can just return the path for an IFrame or File component
            # For now, let's return the path
            return output_path
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
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

