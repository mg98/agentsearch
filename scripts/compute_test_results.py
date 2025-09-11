import pandas as pd
from tqdm import tqdm
from agentsearch.dataset.agents import AgentStore
from agentsearch.dataset.questions import Question

# Load test question IDs
with open('data/test_qids.txt', 'r') as f:
    test_qids = [int(qid) for qid in f.read().strip().split(',')]

print(f"Loaded {len(test_qids)} test question IDs")

# Initialize agent store
agent_store = AgentStore(use_llm_agent_card=False)
agents = agent_store.all(shallow=True)
print(f"Loaded {len(agents)} agents")

questions = [Question.from_id(qid, shallow=False) for qid in test_qids]
print(f"Loaded {len(questions)} questions")

# Prepare results list
results = []

# Process each question
for question in tqdm(questions, desc="Processing questions"):
    # Get the question
    print(f"Processing question {question.id}")
    
    # For each agent, compute confidence score
    for agent in tqdm(agents, desc="Processing agents"):
        print(f"Processing agent {agent.id}")
        confidence = agent.get_confidence(question)
        print(f"Agent {agent.id} confidence: {confidence}")
        
        results.append({
            'question': question.id,
            'agent': agent.id,
            'score': confidence
        })

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv('data/tests.csv', index=False)
print(f"Saved {len(results_df)} results to data/tests.csv")