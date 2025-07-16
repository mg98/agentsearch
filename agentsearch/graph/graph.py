import torch
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import pickle
import os
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from agentsearch.dataset import agents
from agentsearch.agent import eval
from agentsearch.graph.types import GraphData, TrustGNN
from agentsearch.graph.training import train_model, evaluate_and_predict, predict_top_targets
from agentsearch.graph.utils import visualize_graph
from agentsearch.utils.globals import EMBEDDING_DIM
import sys

random.seed(42)

Edge = tuple[int, int, np.ndarray, int] # (source_node_idx, target_node_idx, query_embedding, trust_score)

if __name__ == '__main__':

    edges: list[Edge] = []

    # Get all agents and create ID to index mapping
    all_agents = Agent.all_from_cluster("artificial intelligence", 16)
    assert len(all_agents) == 16
    agent_id_to_index = {agent.id: idx for idx, agent in enumerate(all_agents)}
    print(agent_id_to_index.keys())
    
    i = 0
    all_questions = Question.all(from_agents=[agent.id for agent in all_agents])
    random.shuffle(all_questions)
    questions = all_questions[:len(all_agents)*4]

    print(f"Found {len(questions)} questions")
    
    if not os.path.exists('data/graph.pkl'):
        for question in questions:
            i += 1
            print(f"Processing question {i}: {question.question}")
            print("-"*100)

            # evaluate top-k agents
            agent_network = [agent.id for agent in all_agents if agent.id != question.agent_id]
            print(f"Agent network size: {len(agent_network)}")
            
            matches = list(agents.match_by_qid(question.id, 3, whitelist=agent_network))
            print(f"Found {len(matches)} matches for question {question.id}")
            
            for match in matches:
                print(match.agent.name)
                print(match.agent.scholar_url)
                print(match.similarity_score)

                answer = match.agent.ask(question.question)
                print("Answer:", answer)

                if answer == "I don't know":
                    # hotfix to make evaluation more efficient
                    grade = 0
                    reason = "-"
                else:
                    grade, reason = eval.grade_answer(question.question, answer)
                print("Grade:", grade)
                print("Reason:", reason)
                print("-"*100)

                # Use graph node indices instead of agent CSV IDs
                # Select random index that's not the match agent's index
                available_indices = [idx for idx in agent_id_to_index.values() if idx != agent_id_to_index[match.agent.id]]
                source_idx = random.choice(available_indices)
                target_idx = agent_id_to_index[match.agent.id]
                assert question.embedding is not None
                assert len(question.embedding) == EMBEDDING_DIM
                edges.append((source_idx, target_idx, question.embedding, grade))

        # Use the same agents list for embeddings
        node_embeddings = [agent.embedding for agent in all_agents]
        node_embeddings = np.array(node_embeddings)
        node_embeddings = torch.from_numpy(node_embeddings).float()
        num_nodes = len(node_embeddings)

        assert node_embeddings.shape[1] == EMBEDDING_DIM
        num_edges = len(edges)
        
        print(f"Total edges created: {num_edges}")
        assert num_edges > 0

        source_ids = [edge[0] for edge in edges]
        target_ids = [edge[1] for edge in edges]
        query_embeddings = [edge[2] for edge in edges]
        trust_scores = [edge[3] for edge in edges]
        
        edge_index = torch.tensor([source_ids, target_ids], dtype=torch.long)
        edge_trust_scores = torch.tensor(trust_scores, dtype=torch.float).unsqueeze(1)  # Shape: (num_edges, 1)
        
        edge_query_embeddings = torch.tensor(np.stack(query_embeddings), dtype=torch.float)

        data = GraphData(
            x=node_embeddings,
            edge_index=edge_index,
            edge_trust_score=edge_trust_scores,
            edge_query_embedding=edge_query_embeddings
        )

        
        # Save graph data to pickle file
        with open('data/graph.pkl', 'wb') as f:
            pickle.dump({
                'data': data,
                'agent_id_to_index': agent_id_to_index,
                'num_nodes': num_nodes,
                'num_edges': num_edges
            }, f)
        print("Saved graph data to data/graph.pkl")

    else:
        with open('data/graph.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
            data = loaded_data['data']
            agent_id_to_index = loaded_data['agent_id_to_index'] 
            num_nodes = loaded_data['num_nodes']
            num_edges = loaded_data['num_edges']

    print("Graph Data Summary:")
    print(data)
    print(f"Agent ID to Index mapping created for {len(agent_id_to_index)} agents")
    print(f"Node indices range: 0-{num_nodes-1}")

    # Instantiate the model
    model = TrustGNN(
        hidden_channels=128,
        node_out_channels=64,
    )
    print("\nModel Architecture:")
    print(model)

    # 1. Visualize the initial graph
    # visualize_graph(data, title=f"Graph with {num_nodes} Nodes and {data.num_edges} Edges")

    # 2. Show predictions with the untrained model
    evaluate_and_predict(model, data, title="Predictions Before Training")

    # 3. Train the model
    train_model(model, data, epochs=400)

    # 5. Show predictions with the trained model
    evaluate_and_predict(model, data, title="Predictions After Training")

    # Evaluate another question for comparison
    print("-"*100)
    for i in range(len(all_agents)):
        q = all_questions[len(questions)+i]
        print(f"Predicting top targets for question \"{q.question}\"")
        targets = predict_top_targets(model, data, source_idx=0, question=q, top_k=1)

        target_agent = all_agents[targets[0][0]]
        grade, _ = eval.grade_answer(q.question, target_agent.ask(q.question))
        print(f"PREDICTED:\t{target_agent.name} {grade}")

        agent_ids = [agent.id for agent in all_agents if agent.id != q.agent_id]
        match = agents.match_by_qid(q.id, 1, whitelist=agent_ids)[0]
        grade, _ = eval.grade_answer(q.question, match.agent.ask(q.question))
        print(f"MATCHED:\t{match.agent.name} {grade}")

        print("-" * 50)