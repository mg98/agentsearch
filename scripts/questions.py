import json
import time
import os
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, Field
import pandas as pd

DEBUG = False

client = OpenAI()

class QuestionList(BaseModel):
    questions: list[str] = Field(None, title="Questions")

question_template = """
Generate 5 questions which could have been answered in the paper "{paper}". 
Make sure to include all context needed. 
Do not enumerate the questions, just list them line-by-line.
"""

def create_batch_file(paper_titles: list[str], filename="data/batch_questions.jsonl"):
    """Create a JSONL file with batch requests for all papers"""
    tasks = []

    for idx, paper_title in enumerate(paper_titles):
        task = {
            "custom_id": f"paper-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": question_template.format(paper=paper_title)
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "QuestionList",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "questions": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["questions"],
                            "additionalProperties": False
                        }
                    }
                },
            }
        }
        tasks.append(task)

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
        time.sleep(30)  # Wait 60 seconds between checks
        
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

def retrieve_batch_results(batch_response, paper_titles: list[str]):
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
        paper_idx = int(parts[1])

        if json_response.get('error'):
            print(f"Error for paper {paper_idx}: {json_response['error']}")
            continue

        response_content = json_response['response']['body']['choices'][0]['message']['content']
        try:
            questions_data = json.loads(response_content)
            questions = questions_data.get('questions', [])
            questions = [q.replace('\n', '').strip() for q in questions if q.strip()]

            results.append({
                'paper_title': paper_titles[paper_idx],
                'questions': questions
            })

        except json.JSONDecodeError as e:
            print(f"Error parsing response for paper {paper_idx}: {e}")
            continue

    return results

def save_results_to_csv(results):
    """Save the results to CSV file"""
    # Load papers and agents to create mapping
    papers_df = pd.read_csv('data/papers.csv')
    agents_df = pd.read_csv('data/agents.csv')

    # Create mappings
    title_to_author = dict(zip(papers_df['title'], papers_df['author']))
    name_to_id = dict(zip(agents_df['name'], agents_df['id']))

    data = []

    for result in results:
        paper_title = result['paper_title']
        questions = result['questions']

        # Get agent_id from paper title
        author = title_to_author.get(paper_title)
        agent_id = name_to_id.get(author) if author else None

        for question in questions:
            data.append({
                'agent_id': int(agent_id) if agent_id is not None else None,
                'question': question
            })

    df = pd.DataFrame(data)
    df.to_csv('data/questions.csv', index_label='question_id')
    print(f"Saved {len(data)} questions to data/questions.csv")

    matched = df['agent_id'].notna().sum()
    print(f"\nSummary:")
    print(f"- Total papers processed: {len(results)}")
    print(f"- Total questions generated: {len(data)}")
    print(f"- Questions matched to agents: {matched} ({matched/len(data)*100:.1f}%)")
    print(f"- Average questions per paper: {len(data) / len(results):.1f}")

def main():
    """Main function to orchestrate the batch processing"""
    print("Starting batch question generation...")

    with open('paper_titles.txt', 'r', encoding='utf-8') as f:
        paper_titles = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(paper_titles)} paper titles")

    batch_filename = create_batch_file(paper_titles)

    try:
        file_id = upload_batch_file(batch_filename)
        batch_id = create_batch_job(file_id)
        batch_response = monitor_batch_job(batch_id)

        if batch_response and batch_response.status == "completed":
            results = retrieve_batch_results(batch_response, paper_titles)
            save_results_to_csv(results)

            print("\nBatch processing completed successfully!")

        else:
            print("Batch processing failed or was cancelled")

    except Exception as e:
        print(f"Error during batch processing: {e}")

    finally:
        if os.path.exists(batch_filename):
            os.remove(batch_filename)
            print(f"Cleaned up batch file: {batch_filename}")

if __name__ == "__main__":
    main()
