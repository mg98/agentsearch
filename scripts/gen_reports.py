import random
from colorama import init, Fore, Style
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from tqdm import tqdm
import argparse
import csv
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

    all_agents = Agent.all(collection="agents")
    random.shuffle(all_agents)

    all_questions = Question.all()
    random.shuffle(all_questions)

    test_questions = all_questions[:1000]
    questions = all_questions[1000:11000]

    if args.job_id == 0:
        test_qids = [str(q.id) for q in test_questions]
        with open('data/test_qids.txt', 'w') as f:
            f.write(','.join(test_qids))
        print(f"{Fore.CYAN}Wrote {len(test_qids)} test question IDs to data/test_qids.txt{Style.RESET_ALL}")

    reports_file = f'data/new_reports_{args.job_id}.csv'
    questions = [q for idx, q in enumerate(questions) if idx % args.job_count == args.job_id]

    file_lock = threading.Lock()

    def process_question(question: Question):
        question_color = next(QUESTION_COLORS)
        matches = Agent.match(question, K_MATCHES, collection="agents")

        for match in matches:
            score = match.agent.grade(question)
            print(f"{question_color}Asking question {question.id} to {match.agent.name}: {score}{Style.RESET_ALL}")

            with file_lock:
                with open(reports_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([match.agent.id, question.id, score])

            if score > 0:
                break

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_question, question) for question in questions]

        for future in tqdm(as_completed(futures), total=len(questions), desc="Generating reports"):
            try:
                future.result()
            except Exception as e:
                print(f"{Fore.RED}Error processing question: {e}{Style.RESET_ALL}")
