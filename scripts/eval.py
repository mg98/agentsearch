from agentsearch.utils.eval import load_test_questions, load_all_attacks, TestOracle
from agentsearch.baselines.semantic import semantic_match
from agentsearch.baselines.bm25 import bm25_match, init_bm25
from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question
from agentsearch.baselines.rerank import create_trained_reranker, rerank_match
from agentsearch.baselines.forc import create_trained_meta_model, forc_match

if __name__ == "__main__":
    agent_store = AgentStore(use_llm_agent_card=True)
    test_questions = load_test_questions()
    oracle = TestOracle()

    for attack_volume, graph_df in load_all_attacks():
        # Transform graph_df to list[str, str, float] format (question text, target agent card, score)
        data = []
        for _, row in graph_df.iterrows():
            question = Question.from_id(row['question'], shallow=True)
            target_agent = agent_store.from_id(row['target_agent'], shallow=True)
            data.append((question.question, target_agent.agent_card, row['score']))

        scores = []
        forc = create_trained_meta_model(data)
        for question in test_questions:
            top_agents: list[Agent] = forc_match(forc, agent_store, question)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        pct_positive = sum(1 for score in scores if score > 0) / len(scores) * 100
        print(f"FORC: {avg_score:.4f} (positive: {pct_positive:.1f}%)")

        avg_score = sum(scores) / len(scores)

        scores = []
        for question in test_questions:
            top_agents: list[Agent] = semantic_match(agent_store, question)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        pct_positive = sum(1 for score in scores if score > 0) / len(scores) * 100
        print(f"Semantic: {avg_score:.4f} (positive: {pct_positive:.1f}%)")
        
        scores = []
        bm25 = init_bm25(agent_store)
        for question in test_questions:
            top_agents: list[Agent] = bm25_match(bm25, agent_store, question)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        pct_positive = sum(1 for score in scores if score > 0) / len(scores) * 100
        print(f"BM25: {avg_score:.4f} (positive: {pct_positive:.1f}%)")
        
        scores = []
        reranker = create_trained_reranker(data)
        for question in test_questions:
            top_agents: list[Agent] = rerank_match(reranker, agent_store, question)
            score = oracle.get_score(question.id, top_agents[0].id)
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        pct_positive = sum(1 for score in scores if score > 0) / len(scores) * 100
        print(f"BM25: {avg_score:.4f} (positive: {pct_positive:.1f}%)")

        break

