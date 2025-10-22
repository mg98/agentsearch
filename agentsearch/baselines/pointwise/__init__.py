from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question
from agentsearch.baselines.pointwise.utils import ClusterData, compile_feature_vector
from agentsearch.baselines.pointwise.model import PointwiseModel, train_model, feature_vector_to_tensor
from agentsearch.utils.globals import get_torch_device
import torch

PointwiseData = tuple[Agent, Question, float]

def init_pointwise(data: list[PointwiseData]) -> tuple[ClusterData, PointwiseModel]:
    questions = list(map(lambda d: d[1], data))
    cluster_data = ClusterData(questions)

    x_y_data = []
    for agent, question, score in data:
        feature_vector = compile_feature_vector(data, cluster_data, agent, question)
        x_y_data.append((feature_vector, score))

    model = train_model(x_y_data)
    return cluster_data, model

def pointwise_match(history: list[PointwiseData], cluster_data: ClusterData, model: PointwiseModel, agent_store: AgentStore, question: Question) -> list[Agent]:
    matches = agent_store.match_by_qid(question.id, top_k=8)
    agents = list(map(lambda m: m.agent, matches))

    feature_vectors = [compile_feature_vector(history, cluster_data, agent, question) for agent in agents]
    device = get_torch_device()
    X = torch.stack([feature_vector_to_tensor(fv) for fv in feature_vectors]).to(device)

    with torch.no_grad():
        scores = model(X).cpu().numpy()

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_agents = [agents[i] for i in ranked_indices]
    return ranked_agents