from agentsearch.utils.eval import Oracle, load_test_questions
from agentsearch.dataset.agents import AgentStore
from agentsearch.dataset.questions import Question

def test_oracle():
    """Test Oracle by comparing its rankings with actual agent grades for a question."""
    # Initialize components
    oracle = Oracle()
    agent_store = AgentStore(use_llm_agent_card=False)
    test_questions = load_test_questions()
    
    # Pick an arbitrary question
    question = test_questions[12]
    print(f"Testing with question {question.id}: {question.text}")
    
    # Get Oracle's top agents for this question
    oracle_top_agents = oracle.match(agent_store, question)
    print(f"\nOracle's top {len(oracle_top_agents)} agents:")
    
    # Grade each agent and check if grades are in descending order
    prev_grade = float('inf')
    is_descending = True
    
    for i, agent in enumerate(oracle_top_agents):
        # Load the agent with full data to enable grading
        full_agent = agent_store.from_id(agent.id, shallow=True)
        grade = full_agent.grade(question)
        print(f"  {i+1}. Agent {agent.id} ({agent.name}): grade = {grade}")
        
        # Check if current grade is lower or equal to previous grade
        assert grade <= prev_grade, f"Grade {grade} is higher than previous grade {prev_grade}"
        prev_grade = grade
    
    print(f"\nOracle ranking is in descending order by grade: {is_descending}")

if __name__ == "__main__":
    test_oracle()
