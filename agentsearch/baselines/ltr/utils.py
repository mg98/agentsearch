from dataclasses import dataclass
import numpy as np
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from sklearn.cluster import KMeans
from collections import defaultdict

LTRData = tuple[Agent, Question, float]

@dataclass
class FeatureVector:
    cosine_similarity: float
    num_reports: int
    success_rate: float
    topic_k32_num_reports: int
    topic_k32_success_rate: float
    topic_k8_num_reports: int
    topic_k8_success_rate: float


class ClusterData:
    clusters_k32: dict[int, list[list[str]]]
    centroids_k32: np.ndarray
    clusters_k8: dict[int, list[list[str]]]
    centroids_k8: np.ndarray

    def __init__(self, questions: list[Question]):
        question_ids = [q.id for q in questions]
        question_embeddings = np.array([q.embedding for q in questions])

        kmeans_k32 = KMeans(n_clusters=32, random_state=42)
        preds_k32 = kmeans_k32.fit_predict(question_embeddings)
        self.centroids_k32 = kmeans_k32.cluster_centers_

        clusters_k32 = defaultdict(list)
        for qid, pred in zip(question_ids, preds_k32):
            clusters_k32[pred].append(qid)
        self.clusters_k32 = {cid: [qids] for cid, qids in clusters_k32.items()}

        kmeans_k8 = KMeans(n_clusters=8, random_state=42)
        preds_k8 = kmeans_k8.fit_predict(question_embeddings)
        self.centroids_k8 = kmeans_k8.cluster_centers_

        clusters_k8 = defaultdict(list)
        for qid, pred in zip(question_ids, preds_k8):
            clusters_k8[pred].append(qid)
        self.clusters_k8 = {cid: [qids] for cid, qids in clusters_k8.items()}

    def closest_cluster_k32(self, question: Question) -> int:
        distances = np.linalg.norm(self.centroids_k32 - question.embedding, axis=1)
        return np.argmin(distances)

    def closest_cluster_k8(self, question: Question) -> int:
        distances = np.linalg.norm(self.centroids_k8 - question.embedding, axis=1)
        return np.argmin(distances)
    

def compile_feature_vector(history: list[LTRData], cluster_data: ClusterData, agent: Agent, question: Question, exclude_current: bool = True) -> FeatureVector:
    cos_sim = np.dot(agent.embedding, question.embedding) / (np.linalg.norm(agent.embedding) * np.linalg.norm(question.embedding))

    if exclude_current:
        history = [d for d in history if not (d[0].id == agent.id and d[1].id == question.id)]

    relevant_history = [d for d in history if d[0].id == agent.id]
    num_reports = len(relevant_history)
    success_rate = sum(1 for d in relevant_history if d[2] > 0) / num_reports if num_reports > 0 else 0.0

    closest_cluster_k32 = cluster_data.closest_cluster_k32(question)
    cluster_qids_k32 = set(cluster_data.clusters_k32[closest_cluster_k32][0])
    topic_k32_history = [d for d in history if d[0].id == agent.id and d[1].id in cluster_qids_k32]
    topic_k32_num_reports = len(topic_k32_history)
    topic_k32_success_rate = sum(1 for d in topic_k32_history if d[2] > 0) / topic_k32_num_reports if topic_k32_num_reports > 0 else 0.0

    closest_cluster_k8 = cluster_data.closest_cluster_k8(question)
    cluster_qids_k8 = set(cluster_data.clusters_k8[closest_cluster_k8][0])
    topic_k8_history = [d for d in history if d[0].id == agent.id and d[1].id in cluster_qids_k8]
    topic_k8_num_reports = len(topic_k8_history)
    topic_k8_success_rate = sum(1 for d in topic_k8_history if d[2] > 0) / topic_k8_num_reports if topic_k8_num_reports > 0 else 0.0

    return FeatureVector(
        cosine_similarity=float(cos_sim),
        num_reports=num_reports,
        success_rate=success_rate,
        topic_k32_num_reports=topic_k32_num_reports,
        topic_k32_success_rate=topic_k32_success_rate,
        topic_k8_num_reports=topic_k8_num_reports,
        topic_k8_success_rate=topic_k8_success_rate,
    )