from .invokers import GraphInvoker
from scipy.spatial.distance import cosine
import numpy as np
import rdflib
import logging

class CrossOntologyAligner:
    def __init__(self, dataGraph, clusters, embeddingModel) -> None:
        self.clusterCentroids = None
        self.dataGraph = dataGraph
        self.clusters = clusters
        self.embeddingModel = embeddingModel

    def transcribe(self, query: rdflib.Graph, Invoker: str, Threshold=0.8, Namespace="http://example.org/sphn#") -> rdflib.Graph:
        self.CustomNameSpace = Namespace
        alignedEntities = self._align(query, Invoker, Threshold)
        if len(alignedEntities) == 0:
            raise TypeError("NullPointerException")

        transcribedGraph = rdflib.Graph()

        for subject in query.subjects():
            if subject in alignedEntities.subjects():
                for s, p, o in alignedEntities.triples((subject, None, None)):
                    transcribedGraph.add((s, p, o))
            else:
                for s, p, o in query.triples((subject, None, None)):
                    transcribedGraph.add((s, p, o))
        return transcribedGraph 

    def merge(self, query: rdflib.Graph, Invoker: str, Threshold=0.8, Namespace="http://example.org/sphn#") -> rdflib.Graph:
        self.CustomNameSpace = Namespace
        alignedEntities = self._align(query, Invoker, Threshold)
        if len(alignedEntities) == 0:
            raise TypeError("NullPointerException")
        
        for subject in alignedEntities.subjects():
            if subject in query.subjects():
                for predicate, obj in alignedEntities.predicate_objects(subject):
                    if (subject, predicate, obj) not in query:
                        query.add((subject, predicate, obj))

        return query
    def _align(self, query: rdflib.Graph, Invoker: str, candidateThreshold=0.8) -> rdflib.Graph:
        try:
            logging.info("Starting alignment process.")
            logging.debug("Invoking remote method '%s' with the query graph.", Invoker)
            runnable = GraphInvoker(embeddingModel=self.embeddingModel, namespace=self.CustomNameSpace)
            queryEmbeddingGraph = runnable.remoteInvoker(Invoker, query)

            if len(queryEmbeddingGraph) == 0:
                logging.info("No entities to align found in query.")
                raise TypeError("Valid Query Entities are Null")
            if self.clusterCentroids is None:
                logging.info("Precomputing cluster centroids.")
                self.precompute_centroids()

            # Coarse-grained candidate selection
            logging.info("Start coarse-grained alignment.")
            candidate_selection = {}
            for querySubject, _, queryEmbedding in queryEmbeddingGraph.triples((None, None, None)):
                embedding = self._reconstructEmbeddings(queryEmbedding)
                similarity_scores = [
                    (clusterID, 1 - cosine(embedding, centroid))
                    for clusterID, centroid in self.clusterCentroids.items()
                    if centroid is not None  # Note I've chosen to skip clusters without centroids
                ]

                if similarity_scores:
                    selectedClusterID, max_similarity = max(similarity_scores, key=lambda x: x[1])
                    candidate_selection[querySubject] = selectedClusterID

            logging.info("Coarse-grained candidate selection completed. %d candidates found.", len(candidate_selection))

            # Fine-grained comparison
            alignedGraph = rdflib.Graph()
            logging.info("Starting fine-grained comparison.")

            for querySubject, clusterID in candidate_selection.items():
                cluster_embeddings = [
                    (candidate_subject, obj)
                    for candidate_subject in self.clusters[clusterID]
                    for _, _, obj in self.dataGraph.triples((candidate_subject, None, None))
                ]

                if not cluster_embeddings:
                    logging.warning("No embeddings found for cluster %d", clusterID)
                    continue

                query_embedding = None
                for _, _, obj in queryEmbeddingGraph.triples((querySubject, None, None)):
                    query_embedding = self._reconstructEmbeddings(obj)
                    break

                if query_embedding is None:
                    logging.warning("No embedding found for query subject '%s'. Skipping.", querySubject)
                    continue

                best_similarity = 0
                best_candidate_subject = None
                for candidate_subject, candidate_embedding in cluster_embeddings:
                    similarity = 1 - cosine(query_embedding, self._reconstructEmbeddings(candidate_embedding))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_candidate_subject = candidate_subject

                if best_similarity >= candidateThreshold:
                    logging.info("Matched query '%s' with '%s' : %.2f", querySubject, best_candidate_subject, best_similarity)
                    alignedGraph.add((querySubject, self.CustomNameSpace, best_candidate_subject))

            logging.info("Fine-grained comparison completed. Returning aligned graph.")
            return alignedGraph

        except Exception as e:
            logging.error("An error occurred during alignment: %s", e, exc_info=True)
            return rdflib.Graph()
        finally:
            self._restoreState()

    def precompute_centroids(self):
        cluster_centroids = {}

        for clusterID, candidateSubjects in self.clusters.items():
            embeddings = [
                np.fromstring(str(obj).strip("[]"), sep=" ")  # Convert Literal to NumPy array
                for candidate_subject in candidateSubjects
                for _, _, obj in self.dataGraph.triples((candidate_subject, None, None))
                if isinstance(obj, rdflib.term.Literal)  # Ensure obj is a Literal
            ]
            if embeddings:
                cluster_centroids[clusterID] = sum(embeddings) / len(embeddings)
            else:
                raise ArithmeticError("Cannot compute centroid")

        self.clusterCentroids = cluster_centroids
        logging.info("Centroids precomputed for %d clusters.", len(self.clusterCentroids))

    def _reconstructEmbeddings(self, literal: rdflib.term.Literal):
        if isinstance(literal, rdflib.term.Literal):
            embedding = np.fromstring(str(literal).strip("[]"), sep=" ")
        if not isinstance(embedding, np.ndarray):
            raise AssertionError("Cannot convert literal back to ndarray")
        return embedding

    def _restoreState(self) -> None:
        self.query, self.invoker, self.clusterCentroids = None, None, None
