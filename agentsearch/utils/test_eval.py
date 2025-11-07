from agentsearch.utils.eval import Oracle, load_test_questions
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question

def test_oracle():
    """Test Oracle by comparing its rankings with actual agent grades for a question."""
    oracle = Oracle()
    test_questions = load_test_questions()

    question = test_questions[12]
    print(f"Testing with question {question.id}: {question.text}")

    oracle_top_agents = oracle.match(question, collection="agents")
    print(f"\nOracle's top {len(oracle_top_agents)} agents:")

    prev_grade = float('inf')
    is_descending = True

    for i, agent in enumerate(oracle_top_agents):
        full_agent = Agent.from_id(agent.id, collection="agents")
        grade = full_agent.grade(question)
        print(f"  {i+1}. Agent {agent.id} ({agent.name}): grade = {grade}")

        assert grade <= prev_grade, f"Grade {grade} is higher than previous grade {prev_grade}"
        prev_grade = grade

    print(f"\nOracle ranking is in descending order by grade: {is_descending}")

if __name__ == "__main__":
    test_oracle()
