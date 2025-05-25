"""
Proximity agent
--------------
- Calculates similarity between hypotheses and builds a graph
"""

from typing import List, Set, TypedDict

import networkx as nx
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from sklearn.metrics.pairwise import cosine_similarity

from coscientist.common import load_prompt
from coscientist.custom_types import HypothesisWithID


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
        print(f"Similarities: {similarities}")
        # return similarities
        # Add the edges with weights to the graph
        for i, id_x in enumerate(hypothesis_ids_x):
            for j, id_y in enumerate(hypothesis_ids_y):
                if id_x == id_y:
                    continue
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


class ProximityState(TypedDict):
    """
    Represents the state of the proximity analysis process.

    Parameters
    ----------
    hypotheses: List[HypothesisWithID]
        All hypotheses to analyze for similarity
    proximity_graph: ProximityGraph
        The graph of hypothesis similarities
    clusters: List[Set[int]]
        Clusters of similar hypotheses
    """

    hypotheses: List[HypothesisWithID]
    proximity_graph: ProximityGraph
    clusters: List[Set[int]]


def embedding_computation_node(
    state: ProximityState, llm: BaseChatModel
) -> ProximityState:
    """
    Computes embeddings for all hypotheses and builds the proximity graph.

    Parameters
    ----------
    state: ProximityState
        Current proximity state
    llm: BaseChatModel
        Language model (not used in this node)

    Returns
    -------
    ProximityState
        Updated state with embeddings computed
    """
    graph = state["proximity_graph"]

    # Add all hypotheses to the graph
    for hypothesis in state["hypotheses"]:
        if hypothesis.id not in graph.graph.nodes:
            graph.add_hypothesis(hypothesis)

    # Update edges between all hypotheses
    graph.update_edges()

    return {**state, "proximity_graph": graph}


def clustering_node(state: ProximityState, llm: BaseChatModel) -> ProximityState:
    """
    Performs clustering analysis on the proximity graph.

    Parameters
    ----------
    state: ProximityState
        Current proximity state
    llm: BaseChatModel
        Language model (not used in this node)

    Returns
    -------
    ProximityState
        Updated state with clusters identified
    """
    graph = state["proximity_graph"]

    # Get clusters using community detection
    clusters = graph.get_partitions(resolution=1.0, min_weight=0.3)

    return {**state, "clusters": clusters}


def similarity_optimization_node(
    state: ProximityState, llm: BaseChatModel
) -> ProximityState:
    """
    Optimizes similarity calculations and identifies key patterns.

    Parameters
    ----------
    state: ProximityState
        Current proximity state
    llm: BaseChatModel
        Language model for pattern analysis

    Returns
    -------
    ProximityState
        Updated state with optimized similarity insights
    """
    # Analyze cluster characteristics using the LLM
    cluster_analysis = []

    for i, cluster in enumerate(state["clusters"]):
        if len(cluster) > 1:
            # Get hypotheses in this cluster
            cluster_hypotheses = [
                hyp for hyp in state["hypotheses"] if hyp.id in cluster
            ]

            # Analyze what makes this cluster similar
            cluster_content = "\n\n".join(
                [
                    f"Hypothesis {hyp.id}: {hyp.content}"
                    for hyp in cluster_hypotheses[
                        :3
                    ]  # Limit to first 3 for LLM context
                ]
            )

            prompt = load_prompt(
                "cluster_analysis",
                cluster_id=i + 1,
                cluster_content=cluster_content,
            )

            try:
                response = llm.invoke(prompt)
                cluster_analysis.append(
                    {
                        "cluster_id": i,
                        "size": len(cluster),
                        "analysis": response.content,
                        "hypothesis_ids": list(cluster),
                    }
                )
            except Exception:
                # Skip analysis if LLM fails
                cluster_analysis.append(
                    {
                        "cluster_id": i,
                        "size": len(cluster),
                        "analysis": "Analysis unavailable",
                        "hypothesis_ids": list(cluster),
                    }
                )

    # Store analysis in the state (we can extend ProximityState to include this)
    return state


def build_proximity_agent(llm: BaseChatModel):
    """
    Builds and configures a LangGraph for the proximity agent process.

    The graph analyzes hypothesis similarity through:
    1. Embedding computation - creating vector representations
    2. Clustering - grouping similar hypotheses
    3. Similarity optimization - analyzing patterns and insights

    Parameters
    ----------
    llm: BaseChatModel
        The language model to use for pattern analysis

    Returns
    -------
    StateGraph
        A compiled LangGraph for the proximity agent
    """
    graph = StateGraph(ProximityState)

    # Add nodes
    graph.add_node(
        "embedding_computation", lambda state: embedding_computation_node(state, llm)
    )
    graph.add_node("clustering", lambda state: clustering_node(state, llm))
    graph.add_node(
        "similarity_optimization",
        lambda state: similarity_optimization_node(state, llm),
    )

    # Define transitions
    graph.add_edge("embedding_computation", "clustering")
    graph.add_edge("clustering", "similarity_optimization")
    graph.add_edge("similarity_optimization", END)

    # Set entry point
    graph.set_entry_point("embedding_computation")

    return graph.compile()
