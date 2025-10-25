from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question
from agentsearch.baselines.ltr.utils import ClusterData, compile_feature_vector
from agentsearch.baselines.ltr.model import LTRModel, train_model, feature_vector_to_tensor
from agentsearch.utils.globals import get_torch_device
import torch

LTRData = tuple[Agent, Question, float]

def init_ltr(data: list[LTRData]) -> tuple[ClusterData, LTRModel]:
    questions = list(map(lambda d: d[1], data))
    cluster_data = ClusterData(questions)

    x_y_data = []
    for agent, question, score in data:
        feature_vector = compile_feature_vector(data, cluster_data, agent, question)
        x_y_data.append((feature_vector, score))

    model = train_model(x_y_data)
    return cluster_data, model

def ltr_match(history: list[LTRData], cluster_data: ClusterData, model: LTRModel, agent_store: AgentStore, question: Question) -> list[Agent]:
    matches = agent_store.match_by_qid(question.id, top_k=64)
    agents = list(map(lambda m: m.agent, matches))

    feature_vectors = [compile_feature_vector(history, cluster_data, agent, question) for agent in agents]
    device = get_torch_device()
    X = torch.stack([feature_vector_to_tensor(fv) for fv in feature_vectors]).to(device)

    with torch.no_grad():
        scores = model(X).cpu().numpy()

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_agents = [agents[i] for i in ranked_indices]
    return ranked_agents