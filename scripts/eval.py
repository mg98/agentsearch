from agentsearch.utils.eval import load_test_questions, load_all_attacks, TestOracle
from agentsearch.baselines.semantic import semantic_match
from agentsearch.baselines.bm25 import bm25_match
from agentsearch.dataset.agents import AgentStore, Agent


if __name__ == "__main__":
    agent_store = AgentStore(use_llm_agent_card=True)
    test_questions = load_test_questions()
    oracle = TestOracle()

    for attack_volume, graph_df in load_all_attacks():
        print(f"Attack volume: {attack_volume}")
        for question in test_questions:
            top_agents: list[Agent] = semantic_match(agent_store, question)
            real_top_agents: list[Agent] = oracle.rank_agents(agent_store, question)
            # Get the intersection of agent IDs
            top_agent_ids = {agent.id for agent in top_agents}
            real_top_agent_ids = {agent.id for agent in real_top_agents}
            intersection = top_agent_ids & real_top_agent_ids
            print(f"Intersection cardinality: {len(intersection)}")
        break

