import argparse
import pandas as pd
from tqdm import tqdm
from agentsearch.utils.eval import load_test_questions, load_all_attacks, TestOracle
from agentsearch.baselines.semantic import semantic_match
from agentsearch.baselines.bm25 import bm25_match, init_bm25
from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question
from agentsearch.baselines.rerank import create_trained_reranker, rerank_match, RerankData
from agentsearch.baselines.forc import create_trained_meta_model, forc_match, FORCData
from agentsearch.baselines.lambdamart import init_lambdamart, lambdamart_match, LambdaMARTData
from agentsearch.baselines.ltr import init_ltr, ltr_match, LTRData

def evaluate_baseline(baseline: str, agent_store: AgentStore, test_questions: list[Question],
                     oracle: TestOracle, graph_df: pd.DataFrame):
    """Evaluate a specific baseline and return results."""
    scores = []

    if baseline == "forc":
        data: list[FORCData] = []
        for _, row in graph_df.iterrows():
            question = Question.from_id(row['question'], shallow=True)
            data.append((question.question, row['target_agent'], int(row['score'] > 0)))
        
        meta_model, trainer = create_trained_meta_model(data)
        for question in test_questions:
            top_agents: list[Agent] = forc_match(meta_model, trainer, agent_store, question)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

    elif baseline == "semantic":
        for question in test_questions:
            top_agents: list[Agent] = semantic_match(agent_store, question)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

    elif baseline == "bm25":
        bm25 = init_bm25(agent_store)
        for question in test_questions:
            top_agents: list[Agent] = bm25_match(bm25, agent_store, question)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

    elif baseline == "rerank":
        data: list[RerankData] = []
        for _, row in graph_df.iterrows():
            question = Question.from_id(row['question'], shallow=True)
            target_agent = agent_store.from_id(row['target_agent'], shallow=True)
            data.append((question.question, target_agent.agent_card, row['score']))

        reranker = create_trained_reranker(data)
        for question in test_questions:
            top_agents: list[Agent] = rerank_match(reranker, agent_store, question)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

    elif baseline == "lambdamart":
        data: list[LambdaMARTData] = []
        for _, row in graph_df.iterrows():
            question = Question.from_id(row['question'], shallow=True)
            if row['source_agent'] == 0:
                source_agent = Agent.make_dummy()
            else:
                source_agent = agent_store.from_id(row['source_agent'], shallow=True)
            target_agent = agent_store.from_id(row['target_agent'], shallow=True)
            data.append((source_agent, target_agent, question, row['score']))

        finder = init_lambdamart(data)
        for question in tqdm(test_questions, desc="Evaluating LambdaMART"):
            top_agents: list[Agent] = lambdamart_match(finder, agent_store, question)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

    elif baseline == "ltr":
        data: list[LTRData] = []
        for _, row in graph_df.iterrows():
            agent = agent_store.from_id(int(row['target_agent']), shallow=False)
            question = Question.from_id(int(row['question']), shallow=False)
            data.append((agent, question, row['score']))

        cluster_data, model = init_ltr(data)
        for question in tqdm(test_questions, desc="Evaluating LTR"):
            top_agents: list[Agent] = ltr_match(data, cluster_data, model, agent_store, question)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    avg_score = sum(scores) / len(scores)
    pct_positive = sum(1 for score in scores if score > 0) / len(scores) * 100

    return avg_score, pct_positive


def main():
    parser = argparse.ArgumentParser(description="Evaluate baselines for agent search")
    parser.add_argument("baseline", nargs="?", default="all",
                       choices=["forc", "semantic", "bm25", "rerank", "ltr", "all"],
                       help="Baseline to evaluate (default: all)")

    args = parser.parse_args()

    agent_store = AgentStore(use_llm_agent_card=False)
    test_questions = load_test_questions()
    oracle = TestOracle()

    # Available baselines
    all_baselines = ["forc", "semantic", "bm25", "rerank", "ltr"]
    baselines_to_run = [args.baseline] if args.baseline != "all" else all_baselines
    print(baselines_to_run)

    for attack_volume, graph_df in load_all_attacks():
        print(f"Attack volume: {attack_volume}")

        # Evaluate selected baselines
        for baseline in baselines_to_run:
            avg_score, pct_positive = evaluate_baseline(
                baseline, agent_store, test_questions, oracle, graph_df
            )
            print(f"{baseline.upper()}: {avg_score:.4f} (positive: {pct_positive:.1f}%)")

        break  # Only process first attack volume


if __name__ == "__main__":
    main()

