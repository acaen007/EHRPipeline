from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import logging
import rdflib

class GraphEmbedder(ABC):
    def __init__(self, embeddingModel):
        """
        Initialize the GraphEmbedder with an embedding model.

        :param embeddingModel: Pre-trained embedding model
        """
        self.embeddingModel = embeddingModel
        self.empty_embedding = self.embeddingModel.encode("")  # Cache empty string embedding

    @lru_cache(maxsize=32)
    def loadGraph(self, graphPath):
        """
        Load an RDF graph from a file path.

        :param graphPath: Path to the RDF graph file (e.g., Turtle, RDF/XML)
        :return: An RDF graph object
        """
        KG = rdflib.Graph()
        try:
            _, format = os.path.splitext(graphPath)
            if format == ".ttl":
                format = "turtle"
            elif format in {".rdf", ".xml"}:
                format = "xml"
            elif format == ".n3":
                format = "n3"
            else:
                raise ValueError(f"Unsupported file extension: {format}")
            KG.parse(graphPath, format=format)
        except Exception as e:
            raise ValueError(f"Failed to load RDF graph from {graphPath}: {e}")
        return KG

    def embedEdges(self, edges):
        """
        Embed edges using the embedding model.

        :param edges: List of edges (predicates)
        :return: Aggregated embedding of edges
        """
        if not edges:
            return self.empty_embedding
        embeddings = [self.embeddingModel.encode(edge) for edge in edges]
        return sum(embeddings) / len(embeddings)

    def embedLabels(self, labels):
        """
        Embed labels using the embedding model.

        :param labels: List of labels (objects)
        :return: Aggregated embedding and a mapping of labels to embeddings
        """
        if not labels:
            return self.empty_embedding, {}
        embeddings = [self.embeddingModel.encode(label) for label in labels]
        label_map = dict(zip(labels, embeddings))
        aggregated_embedding = sum(embeddings) / len(embeddings)
        return aggregated_embedding, label_map

    def processBlankNode(self, KG, node, visited=None):
        """
        Process a blank node and recursively extract its semantic content.

        :param KG: The RDF graph
        :param node: The blank node to process
        :param visited: A set of already-visited nodes to prevent infinite recursion
        :return: A tuple (edges, labels) representing the semantic content of the blank node
        """
        if visited is None:
            visited = set()

        if node in visited:
            return [], []
        visited.add(node)

        edges, labels = [], []
        for predicate, obj in KG.predicate_objects(node):
            edges.append(str(predicate))
            if isinstance(obj, rdflib.term.BNode):
                obj_edges, obj_labels = self.processBlankNode(KG, obj, visited)
                edges.extend(obj_edges)
                labels.extend(obj_labels)
            elif isinstance(obj, rdflib.term.Literal):
                labels.append(str(obj))
            else:
                labels.append(str(obj))

        return edges, labels

    def _processSubject(self, KG, subject):
        """
        Process a single subject, extracting edges, labels, and embeddings.
        """
        edges, labels = [], []
        for predicate, obj in KG.predicate_objects(subject):
            edges.append(str(predicate))
            if isinstance(obj, rdflib.term.BNode):
                obj_edges, obj_labels = self.processBlankNode(KG, obj)
                edges.extend(obj_edges)
                labels.extend(obj_labels)
            elif isinstance(obj, rdflib.term.Literal):
                labels.append(str(obj))
            else:
                labels.append(str(obj))
        edge_embeddings = self.embedEdges(edges)
        label_embeddings, _ = self.embedLabels(labels)
        return str(subject), edge_embeddings, label_embeddings

    def embedGraph(self, graphPath):
        """
        Embed a graph by processing its vertices, edges, and labels in parallel.

        :param graphPath: Path to the RDF graph file (e.g., Turtle format)
        :return: List of tuples (vertex, edge_embeddings, label_embeddings)
        """
        logging.info(f"Embedding graph from {graphPath}...")
        KG = self.loadGraph(graphPath)
        logging.debug(f"Graph loaded with {len(KG)} triples.")

        embedded_graph = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(lambda subject: self._processSubject(KG, subject), KG.subjects()),
                desc="Embedding graph",
                unit="vertex",
                leave=True,
                total=len(KG)
            ))

        embedded_graph.extend(results)
        logging.info(f"Graph embedding completed with {len(embedded_graph)} vertices.")
        return embedded_graph

import rdflib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class SimpleDataEmbedder:
    def __init__(self, embeddingModel: object) -> None:
        self.embeddingModel = embeddingModel

    def encode(self, data: rdflib.Graph) -> rdflib.Graph:
        embeddedGraph = rdflib.Graph()

        labels = [
            (subject, rdflib.RDFS.label, str(obj))
            for subject, _, obj in data.triples((None, rdflib.RDFS.label, None))
        ]
        with tqdm(total=len(labels), desc="Embedding Labels", unit="label") as pbar:
            for subject, edge, label in labels:
                embedding = self.embeddingModel.encode(label)
                embeddedGraph.add((subject, edge, rdflib.Literal(embedding)))
                pbar.update(1)
        return embeddedGraph


class DataGraphEmbedder(GraphEmbedder):
    def embedGraph(self, graphPath):
        return super().embedGraph(graphPath)

class ClusterGenerator:
    def __init__(self, dataGraph, n_clusters=5):
        self.dataGraph = dataGraph
        self.n_clusters = n_clusters

    def generate_clusters(self):
        subjects = []
        embeddings = []

        for subject, _, obj in self.dataGraph.triples((None, None, None)):
            try:
                if isinstance(obj, rdflib.term.Literal):
                    embedding = np.fromstring(str(obj).strip("[]"), sep=" ")
                    subjects.append(subject)
                    embeddings.append(embedding)
                else:
                    raise AttributeError("non-literal object for subject '%s'.", subject)
            except Exception as e:
                logging.warning("Failed to encode embedding for subject '%s': %s", subject, e)

        if not subjects:
            logging.warning("No subjects found.")
            return {}

        logging.info("Performing clustering with %d clusters.", self.n_clusters)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(embeddings)

        clusters = {}
        for cluster_id in range(self.n_clusters):
            clusters[cluster_id] = []

        for subject, cluster_id in zip(subjects, cluster_assignments):
            clusters[cluster_id].append(subject)

        logging.info("Clustering completed. Generated %d clusters.", len(clusters))
        return clusters

