import random
import pickle
from colorama import init, Fore, Style
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from agentsearch.dataset import agents
from agentsearch.graph.types import GraphData, TrustGNN
from agentsearch.graph.utils import visualize_graph
from tqdm import tqdm

# Initialize colorama
init(autoreset=True)

random.seed(42)

NUM_AGENTS = 256
CORE_NUM_QUESTIONS = 64
NUM_QUESTIONS = 16
K_MATCHES = 8

if __name__ == '__main__':
    all_agents = Agent.all()
    random.shuffle(all_agents)

    all_questions = Question.all()
    random.shuffle(all_questions)

    total_questions_asked = CORE_NUM_QUESTIONS + (NUM_QUESTIONS*K_MATCHES) * NUM_QUESTIONS
    assert len(all_questions) >= total_questions_asked, "Not enough questions" 

    graph_data = GraphData(all_agents)
    core_agent = all_agents[0]

    direct_target_agents = []

    for question in tqdm(all_questions[:CORE_NUM_QUESTIONS], desc="Core agent asking questions"):
        print(f"\n{Fore.GREEN}Core agent asking question {Style.BRIGHT}\"{question.question}\"{Style.RESET_ALL}")
        # Ensure question embedding is loaded
        if question.embedding is None:
            question.load_embedding()
        
        matches = agents.match_by_qid(question.id, K_MATCHES, whitelist=[a.id for a in all_agents if a.id != core_agent.id])
        assert len(matches) == K_MATCHES
        for match in matches:
            # answer = match.agent.ask(question.question)
            # grade, _ = eval.grade_answer(question.question, answer)
            grade = match.agent.ask(question.question)
            print(f"{core_agent.name} asked {match.agent.name}: {grade}")
            graph_data.add_edge(core_agent, match.agent, question.embedding, grade)
            direct_target_agents.append(match.agent)

    questions_index = CORE_NUM_QUESTIONS

    for agent in tqdm(direct_target_agents, desc="Target agents"):
        for question in tqdm(all_questions[questions_index:questions_index+NUM_QUESTIONS], desc="Agent asking questions"):
            print(f"\n{Fore.GREEN}Agent {agent.name} asking question {Style.BRIGHT}\"{question.question}\"{Style.RESET_ALL}")
            # Ensure question embedding is loaded
            if question.embedding is None:
                question.load_embedding()
                
            matches = agents.match_by_qid(question.id, K_MATCHES, whitelist=[a.id for a in all_agents if a.id != agent.id])
            assert len(matches) == K_MATCHES
            for match in matches:
                # answer = match.agent.ask(question.question)
                # grade, _ = eval.grade_answer(question.question, answer)
                grade = match.agent.ask(question.question)
                print(f"{agent.name} asked {match.agent.name}: {grade}")
                graph_data.add_edge(agent, match.agent, question.embedding, grade)
        questions_index += NUM_QUESTIONS
    
    
    with open('data/graph.pkl', 'wb') as f:
        pickle.dump({
            'graph': graph_data,
            'questions': all_questions,
            'total_questions_asked': total_questions_asked,
            'core_agent': core_agent,
        }, f)
    print(f"\n{Fore.GREEN}Saved graph data to data/graph.pkl{Style.RESET_ALL}")

    visualize_graph(graph_data, title=f"Graph")