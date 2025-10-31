from dataclasses import dataclass
import numpy as np
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from sklearn.cluster import KMeans
from collections import defaultdict

LTRData = tuple[Agent, Question, float]
K_VALUES = [8, 16, 32, 64, 128, 256]

@dataclass
class FeatureVector:
    cosine_similarity: float
    num_reports: int
    success_rate: float
    topic_features: dict[int, tuple[int, float, float]]

    def to_list(self) -> list[float]:
        features = [self.cosine_similarity, self.num_reports, self.success_rate]
        for k in K_VALUES:
            num_reports, success_rate, cos_sim_centroid = self.topic_features[k]
            features.extend([num_reports, success_rate, cos_sim_centroid])
        return features


class ClusterData:
    clusters: dict[int, dict[int, list[list[str]]]]
    centroids: dict[int, np.ndarray]

    def __init__(self, questions: list[Question]):
        question_ids = [q.id for q in questions]
        question_embeddings = np.array([q.embedding for q in questions])

        self.clusters = {}
        self.centroids = {}

        for k in K_VALUES:
            kmeans = KMeans(n_clusters=k, random_state=42)
            preds = kmeans.fit_predict(question_embeddings)
            self.centroids[k] = kmeans.cluster_centers_

            clusters_k = defaultdict(list)
            for qid, pred in zip(question_ids, preds):
                clusters_k[pred].append(qid)
            self.clusters[k] = {cid: [qids] for cid, qids in clusters_k.items()}

    def closest_cluster(self, question: Question, k: int) -> int:
        distances = np.linalg.norm(self.centroids[k] - question.embedding, axis=1)
        return np.argmin(distances)


@dataclass
class QuestionContext:
    question: Question
    normalized_embedding: np.ndarray
    embedding_norm: float
    cluster_info: dict[int, tuple[int, set[str], np.ndarray, float]]

def precompute_question_context(question: Question, cluster_data: ClusterData) -> QuestionContext:
    embedding_norm = np.linalg.norm(question.embedding)
    normalized_embedding = question.embedding / embedding_norm

    cluster_info = {}
    for k in K_VALUES:
        closest_cluster = cluster_data.closest_cluster(question, k)
        cluster_qids = set(cluster_data.clusters[k][closest_cluster][0])
        centroid = cluster_data.centroids[k][closest_cluster]
        cos_sim = np.dot(normalized_embedding, centroid) / np.linalg.norm(centroid)
        cluster_info[k] = (closest_cluster, cluster_qids, centroid, float(cos_sim))

    return QuestionContext(
        question=question,
        normalized_embedding=normalized_embedding,
        embedding_norm=embedding_norm,
        cluster_info=cluster_info
    )

@dataclass
class AgentHistory:
    relevant_history: list[LTRData]
    num_reports: int
    avg_relevance: float

def precompute_agent_histories(history: list[LTRData]) -> dict[str, AgentHistory]:
    agent_histories = defaultdict(list)
    for d in history:
        agent_histories[d[0].id].append(d)

    result = {}
    for agent_id, relevant_history in agent_histories.items():
        num_reports = len(relevant_history)
        avg_relevance = sum(d[2] for d in relevant_history) / num_reports if num_reports > 0 else 0.0
        result[agent_id] = AgentHistory(relevant_history, num_reports, avg_relevance)

    return result

def compile_feature_vector(
    history: list[LTRData],
    cluster_data: ClusterData,
    agent: Agent,
    question: Question,
    question_context: QuestionContext | None = None,
    agent_histories: dict[str, AgentHistory] | None = None
) -> FeatureVector:
    if question_context is None:
        question_context = precompute_question_context(question, cluster_data)

    agent_embedding_norm = np.linalg.norm(agent.embedding)
    cos_sim = np.dot(agent.embedding, question_context.normalized_embedding) / agent_embedding_norm

    if agent_histories is None:
        history = [d for d in history if not (d[0].id == agent.id and d[1].id == question.id)]
        relevant_history = [d for d in history if d[0].id == agent.id]
    else:
        agent_hist = agent_histories.get(agent.id)
        if agent_hist:
            relevant_history = [d for d in agent_hist.relevant_history if d[1].id != question.id]
        else:
            relevant_history = []

    num_reports = len(relevant_history)
    success_rate = sum(1 for d in relevant_history if d[2] > 0) / num_reports if num_reports > 0 else 0.0

    topic_features = {}
    for k in K_VALUES:
        _, cluster_qids, _, cos_sim_centroid = question_context.cluster_info[k]
        topic_history = [d for d in relevant_history if d[1].id in cluster_qids]
        topic_num_reports = len(topic_history)
        topic_success_rate = sum(1 for d in topic_history if d[2] > 0) / topic_num_reports if topic_num_reports > 0 else 0.0
        topic_features[k] = (topic_num_reports, topic_success_rate, cos_sim_centroid)

    return FeatureVector(
        cosine_similarity=float(cos_sim),
        num_reports=num_reports,
        success_rate=success_rate,
        topic_features=topic_features
    )
