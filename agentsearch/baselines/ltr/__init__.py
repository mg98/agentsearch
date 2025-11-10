from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from agentsearch.baselines.ltr.utils import (
    ClusterData,
    compile_feature_vector,
    precompute_question_context,
    precompute_agent_histories
)
from agentsearch.baselines.ltr.model import LTRModel, train_model, feature_vector_to_tensor
from agentsearch.utils.globals import get_torch_device
import torch
from tqdm import tqdm
import random

LTRData = tuple[Agent, Question, float]

def init_ltr(data: list[LTRData]) -> tuple[ClusterData, LTRModel]:
    questions = list(map(lambda d: d[1], data))
    print("Clustering questions...")
    cluster_data = ClusterData(questions)

    print("Precomputing question contexts...")
    question_contexts = {q.id: precompute_question_context(q, cluster_data) for q in tqdm(questions)}

    print("Precomputing agent histories...")
    agent_histories = precompute_agent_histories(data)

    x_y_data = []
    for agent, question, score in tqdm(data, desc="Compiling feature vectors"):
        question_context = question_contexts[question.id]
        feature_vector = compile_feature_vector(data, cluster_data, agent, question, question_context, agent_histories)
        x_y_data.append((feature_vector, score))

    print("Training model...")
    model = train_model(x_y_data)
    print("Model trained")
    return cluster_data, model

def ltr_match(history: list[LTRData], cluster_data: ClusterData, model: LTRModel, question: Question, collection: str = "agents") -> list[Agent]:
    matches = Agent.match(question, top_k=8, collection=collection)
    agents = list(map(lambda m: m.agent, matches))

    question_context = precompute_question_context(question, cluster_data)
    agent_histories = precompute_agent_histories(history)

    feature_vectors = [
        compile_feature_vector(history, cluster_data, agent, question, question_context, agent_histories)
        for agent in agents
    ]
    device = get_torch_device()
    X = torch.stack([feature_vector_to_tensor(fv) for fv in feature_vectors]).to(device)

    with torch.no_grad():
        scores = model(X).cpu().numpy()

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_agents = [agents[i] for i in ranked_indices]
    return ranked_agents