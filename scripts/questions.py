import json
import time
import os
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, Field
import pandas as pd
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.papers import Paper
from concurrent.futures import ProcessPoolExecutor
from functools import partial

DEBUG = False

client = OpenAI()

class Question(BaseModel):
    question: str = Field(None, title="Question")

SYSTEM_PROMPT = """
You are generating knowledge-testing questions from scientific texts.

Given the text below, write **one simple question** that could be **answered using the scientific knowledge, findings, or reasoning** contained in the text — as if the reader already understands the paper’s domain.

Avoid questions about:
- the structure or purpose of the paper
- what the authors studied or concluded
- generic “what is the main finding” type questions

Focus on:
- mechanisms
- experimental observations
- theoretical implications
- relationships between concepts
"""

def _make_task(paper):
    return {
        "custom_id": f"paper-{paper.id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4.1-nano",
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": paper.extract_text()
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Question",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string"
                            }
                        },
                        "required": ["question"],
                        "additionalProperties": False
                    }
                }
            },
        }
    }

def create_batch_file(papers: list[Paper], filename="data/batch_questions.jsonl"):
    """Create a JSONL file with batch requests for all papers (using multicore processing)"""
    from multiprocessing import cpu_count

    tasks = []
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        # tqdm does not natively handle as_completed, but we can wrap the enumerate with tqdm for progress bar
        task_iter = executor.map(_make_task, papers)
        tasks = list(tqdm(task_iter, total=len(papers), desc="Creating batch requests"))

    with open(filename, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

    print(f"Created batch file with {len(tasks)} requests: {filename}")
    return filename

def upload_batch_file(filename):
    """Upload the batch file to OpenAI"""
    print(f"Uploading batch file: {filename}")
    
    with open(filename, 'rb') as f:
        batch_file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    print(f"File uploaded successfully. File ID: {batch_file.id}")
    return batch_file.id

def create_batch_job(file_id):
    """Create a batch job with the uploaded file"""
    print(f"Creating batch job with file ID: {file_id}")
    
    batch_job = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    print(f"Batch job created successfully. Job ID: {batch_job.id}")
    return batch_job.id

def monitor_batch_job(batch_id):
    """Monitor the batch job status until completion"""
    print(f"Monitoring batch job: {batch_id}")
    
    status = "validating"
    while status not in ("completed", "failed", "cancelled", "expired"):
        time.sleep(30)  # Wait 30 seconds between checks
        
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Batch Status: {status}")
        
        if status == "failed" or status == "cancelled":
            print("Batch job failed!")
            if batch_response.errors:
                for error in batch_response.errors.data:
                    print(f"Error: {error.code} - {error.message}")
            return None
    
    print(f"Batch job completed with status: {status}")
    return batch_response

def retrieve_batch_results(batch_response):
    """Retrieve and process the batch results"""
    if not batch_response.output_file_id:
        print("No output file available")
        return []

    print(f"Retrieving results from file: {batch_response.output_file_id}")

    file_response = client.files.content(batch_response.output_file_id)
    raw_responses = file_response.text.strip().split('\n')

    results = []
    for raw_response in raw_responses:
        json_response = json.loads(raw_response)

        custom_id = json_response['custom_id']
        parts = custom_id.split('-')
        paper_idx = parts[1]

        if json_response.get('error'):
            print(f"Error for paper {paper_idx}: {json_response['error']}")
            continue

        response_content = json_response['response']['body']['choices'][0]['message']['content']
        try:
            question_data = json.loads(response_content)
            question = question_data.get('question', '').replace('\n', ' ').strip()

            if question:
                paper = Paper.from_id(paper_idx)
                results.append({
                    'agent_id': paper.agent_id,
                    'paper_id': paper.id,
                    'question': question
                })

        except json.JSONDecodeError as e:
            print(f"Error parsing response for paper {paper_idx}: {e}")
            continue

    return results

def save_results_to_csv(results):
    """Save the results to CSV file"""
    df = pd.DataFrame(results)
    df.to_csv('data/questions.csv', index_label='question_id', escapechar='\\')
    print(f"Saved {len(results)} questions to data/questions.csv")

    matched = df['agent_id'].notna().sum()
    print(f"\nSummary:")
    print(f"- Total papers processed: {len(results)}")
    print(f"- Total questions generated: {len(results)}")
    print(f"- Questions matched to agents: {matched} ({matched/len(results)*100:.1f}%)")

def main():
    """Main function to orchestrate the batch processing"""
    print("Starting batch question generation...")

    # deduped list of papers
    agents = Agent.all()
    papers: list[Paper] = list({paper.id: paper for agent in agents for paper in agent.papers}.values())

    print(f"Loaded {len(papers)} papers")

    BATCH_SIZE = 3000
    num_batches = (len(papers) + BATCH_SIZE - 1) // BATCH_SIZE
    results = []

    for batch_num in range(num_batches):
        batch_papers = papers[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
        batch_filename = create_batch_file(batch_papers, filename=f"data/batch_questions_{batch_num}.jsonl")
        try:
            file_id = upload_batch_file(batch_filename)
            batch_id = create_batch_job(file_id)
            batch_response = monitor_batch_job(batch_id)

            if batch_response and batch_response.status == "completed":
                batch_results = retrieve_batch_results(batch_response)
                results.extend(batch_results)
                print(f"\nCompleted batch {batch_num+1}/{num_batches} ({len(batch_results)} questions)")
            
                os.remove(batch_filename)
                print(f"Cleaned up batch file: {batch_filename}")
            else:
                print(f"Batch {batch_num+1} processing failed or was cancelled")

        except Exception as e:
            print(f"Error during batch {batch_num+1} processing: {e}")

    save_results_to_csv(results)

if __name__ == "__main__":
    main()
