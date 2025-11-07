import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from agentsearch.utils.eval import load_test_questions, load_data, TestOracle
from agentsearch.baselines.semantic import semantic_match
from agentsearch.baselines.bm25 import bm25_match, init_bm25
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from agentsearch.baselines.rerank import create_trained_reranker, rerank_match, RerankData
from agentsearch.baselines.forc import create_trained_meta_model, forc_match, FORCData
from agentsearch.baselines.ltr import init_ltr, ltr_match, LTRData
from agentsearch.baselines.regressive_ltr import init_regressive_ltr, regressive_ltr_match
from agentsearch.baselines.set_transformer import init_set_transformer, set_transformer_match, SetTransformerData

np.random.seed(42)

def evaluate_baseline(baseline: str, test_questions: list[Question],
                     oracle: TestOracle, reports: pd.DataFrame, collection: str = "agents"):
    """Evaluate a specific baseline and return results."""
    scores = []

    if baseline == "forc":
        data: list[FORCData] = []
        for _, row in reports.iterrows():
            question = Question.from_id(row['question'])
            data.append((question.text, row['agent'], int(row['score'] > 0)))
        
        meta_model, trainer = create_trained_meta_model(data)
        for question in test_questions:
            top_agents: list[Agent] = forc_match(meta_model, trainer, question, collection)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

    elif baseline == "semantic":
        query_times = []
        for question in tqdm(test_questions[:100], desc="Evaluating Semantic"):
            start_time = time.time()
            top_agents: list[Agent] = semantic_match(question, top_k=1, collection=collection)
            query_time = (time.time() - start_time) * 1000
            query_times.append(query_time)


            top_agent = Agent.from_id(top_agents[0].id, collection=collection)
            score = top_agent.grade(question)
            scores.append(score)

        print(scores)

        avg_query_time = np.mean(query_times)
        print(f"  Semantic: Avg query time: {avg_query_time:.2f}ms ({len(query_times)} queries)")

    elif baseline == "bm25":
        bm25 = init_bm25(collection)
        for question in test_questions:
            top_agents: list[Agent] = bm25_match(bm25, question, collection)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

    elif baseline == "rerank":
        data: list[RerankData] = []
        for _, row in reports.iterrows():
            question = Question.from_id(row['question'])
            target_agent = Agent.from_id(row['agent'], collection=collection)
            data.append((question.text, target_agent.agent_card, row['score']))

        reranker = create_trained_reranker(data)
        for question in test_questions:
            top_agents: list[Agent] = rerank_match(reranker, question, collection)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

    elif baseline == "ltr":
        data: list[LTRData] = []
        for _, row in reports.iterrows():
            agent = Agent.from_id(int(row['agent']), collection=collection)
            question = Question.from_id(int(row['question']))
            data.append((agent, question, row['score']))

        print("Init LTR model...")
        cluster_data, model = init_ltr(data)
        query_times = []
        for question in tqdm(test_questions[:100], desc="Evaluating LTR"):
            start_time = time.time()
            top_agents: list[Agent] = ltr_match(data, cluster_data, model, question, collection)
            query_time = (time.time() - start_time) * 1000
            query_times.append(query_time)

            top_agent = Agent.from_id(top_agents[0].id, collection=collection)
            score = top_agent.grade(question)
            scores.append(score)

        avg_query_time = np.mean(query_times)
        print(f"  LTR: Avg query time: {avg_query_time:.2f}ms ({len(query_times)} queries)")

    elif baseline == "regressive_ltr":
        data: list[LTRData] = []
        for _, row in reports.iterrows():
            agent = Agent.from_id(int(row['agent']), collection=collection)
            question = Question.from_id(int(row['question']))
            data.append((agent, question, row['score']))

        print("Init Regressive LTR model...")
        cluster_data, model = init_regressive_ltr(data)
        query_times = []
        for question in tqdm(test_questions[:100], desc="Evaluating Regressive LTR"):
            start_time = time.time()
            top_agents: list[Agent] = regressive_ltr_match(data, cluster_data, model, question, collection)
            query_time = (time.time() - start_time) * 1000
            query_times.append(query_time)

            top_agent = Agent.from_id(top_agents[0].id, collection=collection)
            score = top_agent.grade(question)
            scores.append(score)

        print(scores)

        avg_query_time = np.mean(query_times)
        print(f"  Regressive LTR: Avg query time: {avg_query_time:.2f}ms ({len(query_times)} queries)")

    elif baseline == "q_semantic":
        for question in tqdm(test_questions[:100], desc="Evaluating Q-Semantic"):
            top_agents: list[Agent] = semantic_match(question, top_k=16, collection=collection)
            # Compute mean embedding for each agent
            mean_embeddings = []
            for agent in top_agents:
                rel_data = [row['question'] for _, row in reports.iterrows() if row['agent'] == agent.id]
                if len(rel_data) == 0:
                    mean_embedding = Agent.from_id(agent.id, collection=collection).embedding
                else:
                    question_embeddings = [Question.from_id(int(qid)).embedding for qid in rel_data]
                    mean_embedding = np.mean(question_embeddings, axis=0)
                mean_embeddings.append(mean_embedding)

            # Compute cosine similarities and re-sort top_agents
            question_emb = question.embedding
            def cosine_sim(a, b):
                a = np.asarray(a)
                b = np.asarray(b)
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            agent_sims = [
                (agent, cosine_sim(question_emb, mean_emb))
                for agent, mean_emb in zip(top_agents, mean_embeddings)
            ]
            agent_sims_sorted = sorted(agent_sims, key=lambda x: x[1], reverse=True)
            top_agents = [agent for agent, _ in agent_sims_sorted]

            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)
    elif baseline == "set_transformer":
        data: list[SetTransformerData] = []
        for _, row in reports.iterrows():
            agent = Agent.from_id(int(row['agent']), collection=collection)
            question = Question.from_id(int(row['question']))
            data.append((agent, question, row['score']))

        model = init_set_transformer(data)
        for question in tqdm(test_questions, desc="Evaluating SetTransformer"):
            top_agents: list[Agent] = set_transformer_match(data, model, question, collection)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

    elif baseline == "oracle":
        for question in test_questions:
            top_agents: list[Agent] = oracle.rank_agent_ids(question.id, top_k=1)
            score = oracle.get_score(question.id, top_agents[0])
            scores.append(score)

    elif baseline == "topk_oracle":
        for question in test_questions:
            semantic_agents: list[Agent] = semantic_match(question, top_k=64, collection=collection)
            semantic_agent_ids = [agent.id for agent in semantic_agents]
            oracle_ranked = oracle.rank_agent_ids(question.id, top_k=len(semantic_agent_ids))

            best_agent_id = None
            for agent_id in oracle_ranked:
                if agent_id in semantic_agent_ids:
                    best_agent_id = agent_id
                    break

            if best_agent_id is None:
                best_agent_id = semantic_agent_ids[0]

            score = oracle.get_score(question.id, best_agent_id)
            scores.append(score)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    avg_score = sum(scores) / len(scores)
    pct_positive = sum(1 for score in scores if score > 0) / len(scores) * 100

    return avg_score, pct_positive


def main():
    parser = argparse.ArgumentParser(description="Evaluate baselines for agent search")
    parser.add_argument("baseline",
                       choices=["forc", "semantic", "bm25", "rerank", "ltr", "regressive_ltr", "set_transformer", "q_semantic", "oracle", "topk_oracle"],
                       help="Baseline to evaluate")
    args = parser.parse_args()

    test_questions = load_test_questions()
    oracle = TestOracle()

    # Evaluate the selected baseline
    reports = load_data("data/new_reports.csv")
    # Group reports by 'question', shuffle groups, select first 10000 groups, then combine
    # question_groups = list(reports.groupby('question'))
    # np.random.shuffle(question_groups)
    # selected_groups = question_groups
    # reports = pd.concat([group for _, group in selected_groups], ignore_index=True)

    collection = "agents"
    avg_score, pct_positive = evaluate_baseline(
        args.baseline, test_questions, oracle, reports, collection
    )
    print(f"{args.baseline.upper()}: {avg_score:.4f} (positive: {pct_positive:.1f}%)")

if __name__ == "__main__":
    main()

