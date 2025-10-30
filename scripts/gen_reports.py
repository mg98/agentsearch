import random
from colorama import init, Fore, Style
from agentsearch.dataset.agents import Agent, AgentStore
from agentsearch.dataset.questions import Question
from tqdm import tqdm
import argparse
import csv
from itertools import cycle

init(autoreset=True)
random.seed(42)
K_MATCHES = 16

# Define a list of colors to rotate through for question grouping
QUESTION_COLORS = cycle([
    Fore.YELLOW,
    Fore.GREEN,
    Fore.MAGENTA,
    Fore.BLUE,
    Fore.RED,
    Fore.CYAN,
    Fore.LIGHTBLUE_EX,
    Fore.LIGHTGREEN_EX,
    Fore.LIGHTMAGENTA_EX,
    Fore.LIGHTYELLOW_EX,
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate agent-question interaction reports')
    parser.add_argument('--job-id', type=int, default=0, help='Job ID for parallel processing (0-indexed)')
    parser.add_argument('--job-count', type=int, default=1, help='Total number of parallel jobs')
    args = parser.parse_args()

    print(f"{Fore.CYAN}Generating reports for job {args.job_id+1} of {args.job_count}{Style.RESET_ALL}")

    agent_store = AgentStore(use_llm_agent_card=False)
    all_agents = agent_store.all(shallow=True)
    random.shuffle(all_agents)

    all_questions = Question.all()
    random.shuffle(all_questions)

    test_questions = all_questions[:1000]
    questions = all_questions[1000:]

    if args.job_id == 0:
        test_qids = [str(q.id) for q in test_questions]
        with open('data/test_qids.txt', 'w') as f:
            f.write(','.join(test_qids))
        print(f"{Fore.CYAN}Wrote {len(test_qids)} test question IDs to data/test_qids.txt{Style.RESET_ALL}")

    reports_file = f'data/reports_{args.job_id}.csv'
    questions = [q for idx, q in enumerate(questions) if idx % args.job_count == args.job_id]

    for question in tqdm(questions, desc="Generating reports"):
        # Get the next color from the cycle for this question
        question_color = next(QUESTION_COLORS)
        
        matches = agent_store.match_by_qid(question.id, K_MATCHES)
        for match in matches:
            trust_score = match.agent.grade(question)
            print(f"{question_color}Asking question {question.id} to {match.agent.name}: {trust_score}{Style.RESET_ALL}")
            report_row = [match.agent.id, question.id, trust_score]

            with open(reports_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(report_row)

            if trust_score > 0:
                break
