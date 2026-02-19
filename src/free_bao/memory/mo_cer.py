import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import dataclasses
import json

@dataclasses.dataclass
class Episode:
    task_description: str
    trajectory: str
    success: bool
    turns: int
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

class MOCER:
    def __init__(self, collection_name: str = "free_bao_memory", persist_directory: str = "./memory_db", alpha: float = 0.1):
        self.client = chromadb.Client(Settings(persist_directory=persist_directory, is_persistent=True))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.alpha = alpha

    def add_episode(self, episode: Episode):
        """Adds an episode to the memory."""
        embedding = self.model.encode(episode.task_description).tolist()
        
        # We store metadata for filtering and retrieval
        metadata = {
            "success": episode.success,
            "turns": episode.turns,
            "task": episode.task_description,
            **episode.metadata
        }
        
        self.collection.add(
            ids=[str(hash(episode.task_description + str(episode.turns) + episode.trajectory[:50]))], # Simple hash
            embeddings=[embedding],
            documents=[episode.trajectory],
            metadatas=[metadata]
        )

    def retrieve_pareto_efficient(self, task_description: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves episodes that are:
        1. Contextually relevant (high similarity)
        2. Successful
        3. Efficient (low turns)
        
        Uses a heuristic Pareto sort.
        """
        query_embedding = self.model.encode(task_description).tolist()
        
        # 1. Fetch relevant successful candidates
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 3, # Fetch more to filter
            where={"success": True}
        )
        
        if not results['documents'][0]:
            return []

        candidates = []
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            candidates.append({
                "trajectory": doc,
                "turns": meta["turns"],
                "distance": dist,
                "task": meta["task"]
            })
            
        # 2. Sort by simple weighted score (Similarity vs Efficiency)
        # Lower distance is better. Lower turns is better.
        # Score = Distance + (Turns * alpha)
        # This is a simplification of Pareto sorting for immediate practicality.
        # We value similarity highly, but penalties for turns apply.
        
        candidates.sort(key=lambda x: x['distance'] + (x['turns'] * self.alpha))
        
        return candidates[:k]

    def get_formatted_retrieval(self, task_description: str, k: int = 1) -> str:
        items = self.retrieve_pareto_efficient(task_description, k)
        if not items:
            return ""
        
        formatted = "Here are efficient examples of how to solve similar tasks:\n\n"
        for i, item in enumerate(items):
            formatted += f"Example {i+1} (Solved in {item['turns']} turns):\n"
            formatted += f"Task: {item['task']}\n"
            formatted += f"Trajectory:\n{item['trajectory']}\n\n"
            
        return formatted
