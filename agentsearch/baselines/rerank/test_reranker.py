"""
Simple test script for the BERT cross-encoder re-ranking implementation
"""

from agentsearch.dataset.agents import AgentStore
from agentsearch.dataset.questions import Question
from agentsearch.baselines.bhopale.reranker import BERTCrossEncoderReranker
import sys


def test_reranker():
    """Test the re-ranking functionality with a single question"""
    
    print("Testing BERT Cross-Encoder Re-ranking...")
    print("-" * 50)
    
    # Initialize components
    print("Initializing agent store and reranker...")
    agent_store = AgentStore(use_llm_agent_card=False)
    reranker = BERTCrossEncoderReranker()
    
    # Load test question IDs
    try:
        with open("data/test_qids.txt", "r") as f:
            test_qids = [int(qid.strip()) for qid in f.read().split(',')]
        
        # Test with first question
        qid = test_qids[0]
        print(f"\nTesting with question ID: {qid}")
        
    except FileNotFoundError:
        # Fallback to a default question if test file not found
        print("Test file not found, using default question ID: 1")
        qid = 1
    
    # Get question
    question = Question.from_id(qid)
    print(f"Question: {question.question[:100]}...")
    print(f"True Agent ID: {question.agent_id}")
    
    # Initial retrieval
    print("\nPerforming initial retrieval (top 100)...")
    initial_matches = agent_store.match_by_qid(qid, top_k=100)
    
    print(f"Initial top 5 agents:")
    for i, match in enumerate(initial_matches[:5], 1):
        print(f"  {i}. Agent {match.agent.id}: {match.agent.name} (score: {match.similarity_score:.4f})")
    
    # Check if true agent is in initial results
    initial_ranks = [m.agent.id for m in initial_matches]
    if question.agent_id in initial_ranks:
        initial_rank = initial_ranks.index(question.agent_id) + 1
        print(f"\nTrue agent found at rank {initial_rank} in initial retrieval")
    else:
        print(f"\nTrue agent NOT found in initial top 100")
    
    # Re-rank
    print("\nRe-ranking with BERT cross-encoder...")
    reranked_matches = reranker.rerank_with_agents(
        query=question.question,
        agent_matches=initial_matches,
        top_k=10,
        show_progress=True
    )
    
    print(f"\nRe-ranked top 5 agents:")
    for i, match in enumerate(reranked_matches[:5], 1):
        print(f"  {i}. Agent {match.agent.id}: {match.agent.name} (score: {match.similarity_score:.4f})")
    
    # Check if true agent is in re-ranked results
    reranked_ranks = [m.agent.id for m in reranked_matches]
    if question.agent_id in reranked_ranks:
        reranked_rank = reranked_ranks.index(question.agent_id) + 1
        print(f"\nTrue agent found at rank {reranked_rank} in re-ranked results")
        
        if question.agent_id in initial_ranks[:10]:
            improvement = initial_rank - reranked_rank
            if improvement > 0:
                print(f"✅ Improved by {improvement} positions!")
            elif improvement == 0:
                print(f"➖ Same position")
            else:
                print(f"⚠️  Dropped by {-improvement} positions")
    else:
        print(f"\nTrue agent NOT found in re-ranked top 10")
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")


if __name__ == "__main__":
    try:
        test_reranker()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)