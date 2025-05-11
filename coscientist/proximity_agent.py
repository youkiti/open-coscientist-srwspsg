"""
Proximity agent
--------------
- Calculates similarity between hypotheses and builds a graph
"""

from typing import List, Set

import numpy as np
import networkx as nx
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

from coscientist.types import HypothesisWithID


def create_embedding(text: str, dimensions: int = 256) -> np.ndarray:
    """Create a vector embedding for a text."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=dimensions)
    return np.array(embeddings.embed_query(text))


class ProximityGraph:
    """A graph of hypotheses and their similarity scores."""

    def __init__(self):
        self.graph = nx.Graph()

    def add_hypothesis(self, hypothesis: HypothesisWithID):
        """Add a hypothesis to the graph."""
        embedding = create_embedding(hypothesis.content)
        self.graph.add_node(
            hypothesis.id, hypothesis=hypothesis.content, embedding=embedding
        )

    def _compute_weighted_edges(
        self, hypothesis_ids_x: List[int], hypothesis_ids_y: List[int]
    ):
        """Compute the weighted edges between two sets of hypotheses."""
        embeddings_x = [self.graph.nodes[id]["embedding"] for id in hypothesis_ids_x]
        embeddings_y = [self.graph.nodes[id]["embedding"] for id in hypothesis_ids_y]
        similarities = cosine_similarity(embeddings_x, embeddings_y)
        # return similarities
        # Add the edges with weights to the graph
        for i, id_x in enumerate(hypothesis_ids_x):
            for j, id_y in enumerate(hypothesis_ids_y):
                self.graph.add_edge(id_x, id_y, weight=similarities[i, j])

    def update_edges(self):
        """
        Finds all nodes without an edge and all nodes with an edge and
        computes the weighted edges between them. If no nodes have edges,
        it will compute the weighted edges between all nodes.
        """
        # Hypothesis ids x are the nodes with degree greater than 0
        hypothesis_ids_x = [
            node for node in self.graph.nodes if self.graph.degree(node) > 0
        ]
        hypothesis_ids_y = [
            node for node in self.graph.nodes if self.graph.degree(node) == 0
        ]
        if len(hypothesis_ids_y) == 0:
            # Nothing to do, we're already up to date
            return
        elif len(hypothesis_ids_x) == 0:
            # No nodes with edges, compute all edges
            self._compute_weighted_edges(hypothesis_ids_y, hypothesis_ids_y)
        else:
            # Compute edges between nodes with and without edges
            self._compute_weighted_edges(hypothesis_ids_y, hypothesis_ids_y)
            self._compute_weighted_edges(hypothesis_ids_x, hypothesis_ids_y)

    def get_partitions(
        self, resolution: float = 1.0, min_weight: float = 0.3
    ) -> List[Set[int]]:
        """Get the partitions of the graph using the Louvain method."""
        # Prune edges from the graph with weight less than min_weight
        pruned_graph = self.graph.copy()
        edges_to_remove = [
            (u, v)
            for u, v, d in pruned_graph.edges(data=True)
            if d["weight"] < min_weight
        ]
        pruned_graph.remove_edges_from(edges_to_remove)

        return nx.community.louvain_communities(pruned_graph, resolution=resolution)
