from dataclasses import dataclass
import numpy as np
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from sklearn.cluster import KMeans
from collections import defaultdict

PointwiseData = tuple[Agent, Question, float]

@dataclass
class FeatureVector:
    cosine_similarity: float
    num_reports: int
    score_min: int
    score_max: int
    score_mean: float
    score_variance: float
    topic_num_reports: int
    topic_score_min: int
    topic_score_max: int
    topic_score_mean: float
    topic_score_variance: float


class ClusterData:
    clusters: dict[int, list[list[str]]]  # cluster_id -> [[question_ids, ...]]
    centroids: np.ndarray  # shape (n_clusters, embedding_dim)

    def __init__(self, questions: list[Question]):
        question_ids = [q.id for q in questions]
        question_embeddings = np.array([q.embedding for q in questions])

        # Choose number of clusters - default to sqrt of number of questions, minimum 2.
        n_clusters = 32 #max(2, int(np.sqrt(len(questions))))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        preds = kmeans.fit_predict(question_embeddings)
        self.centroids = kmeans.cluster_centers_

        clusters = defaultdict(list)
        for qid, pred in zip(question_ids, preds):
            clusters[pred].append(qid)

        self.clusters = {cid: [qids] for cid, qids in clusters.items()}

    def closest_cluster(self, question: Question) -> int:
        distances = np.linalg.norm(self.centroids - question.embedding, axis=1)
        return np.argmin(distances)
    

def compile_feature_vector(history: list[PointwiseData], cluster_data: ClusterData, agent: Agent, question: Question) -> FeatureVector:
    cos_sim = np.dot(agent.embedding, question.embedding) / (np.linalg.norm(agent.embedding) * np.linalg.norm(question.embedding))
    relevant_history = [d for d in history if d[0].id == agent.id]
    num_reports = len(relevant_history)
    scores = [d[2] for d in relevant_history] or [0]

    # Filter relevant_history to only include those where the question is in the same cluster
    closest_cluster_id = cluster_data.closest_cluster(question)
    cluster_qids = set(cluster_data.clusters[closest_cluster_id][0])
    topic_relevant_history = [d for d in history if d[0].id == agent.id and d[1].id in cluster_qids]
    topic_num_reports = len(topic_relevant_history)
    topic_scores = [d[2] for d in topic_relevant_history] or [0]
    
    return FeatureVector(
        cosine_similarity = float(cos_sim),
        num_reports=num_reports,
        score_min=np.min(scores),
        score_max=np.max(scores),
        score_mean=np.mean(scores),
        score_variance=np.var(scores),
        topic_num_reports=topic_num_reports,
        topic_score_min=np.min(topic_scores),
        topic_score_max=np.max(topic_scores),
        topic_score_mean=np.mean(topic_scores),
        topic_score_variance=np.var(topic_scores),
    )