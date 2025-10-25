import json
import time
import os
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, Field
import pandas as pd
from agentsearch.dataset.agents import Agent, AgentStore
from typing import List

DEBUG = False

client = OpenAI()

class QuestionList(BaseModel):
    questions: List[str] = Field(None, title="Questions")

question_template = """
You are a scientist and expert in the following domains: {expertise}.

Generate 10 questions in English language that an expert in these domains would be able to answer.
Do not enumerate the questions, just list them.
"""

def create_batch_file(agents: list[Agent], filename="data/batch_questions.jsonl"):
    """Create a JSONL file with batch requests for all categories"""
    tasks = []
    
    for agent in agents:
        task = {
            "custom_id": f"agent-{agent.id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": question_template.format(
                            expertise=agent.agent_card
                        )
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
                # "temperature": 0.6,
                # "max_tokens": 1000
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

def retrieve_batch_results(batch_response):
    """Retrieve and process the batch results"""
    if not batch_response.output_file_id:
        print("No output file available")
        return []
    
    print(f"Retrieving results from file: {batch_response.output_file_id}")
    
    # Download the results file
    file_response = client.files.content(batch_response.output_file_id)
    raw_responses = file_response.text.strip().split('\n')
    
    # Process the results
    results = []
    for raw_response in raw_responses:
        json_response = json.loads(raw_response)
        
        # Extract custom_id to get category info
        custom_id = json_response['custom_id']
        # custom_id format: "category-{idx}-{category.code}"
        parts = custom_id.split('-')
        agent_id = int(parts[1])
        
        # Check if the request was successful
        if json_response.get('error'):
            print(f"Error for agent {agent_id}: {json_response['error']}")
            continue
        
        # Extract the questions from the response
        response_content = json_response['response']['body']['choices'][0]['message']['content']
        try:
            questions_data = json.loads(response_content)
            questions = questions_data.get('questions', [])
            
            # Clean up questions
            questions = [q.replace('\n', '').strip() for q in questions if q.strip()]
            
            results.append({
                'agent_id': agent_id,
                'questions': questions
            })
            
        except json.JSONDecodeError as e:
            print(f"Error parsing response for agent {agent_id}: {e}")
            continue
    
    return results

def save_results_to_csv(results):
    """Save the results to CSV file"""
    data = []
    
    for result in results:
        agent_id = result['agent_id']
        questions = result['questions']
        
        for question in questions:
            data.append({
                'agent_id': agent_id,
                'question': question
            })
    
    df = pd.DataFrame(data)
    df.to_csv('data/questions.csv', index=True)
    print(f"Saved {len(data)} questions to data/questions.csv")
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Total agents processed: {len(results)}")
    print(f"- Total questions generated: {len(data)}")
    print(f"- Average questions per agent: {len(data) / len(results):.1f}")

def main():
    """Main function to orchestrate the batch processing"""
    print("Starting batch question generation...")
    
    agent_store = AgentStore(use_llm_agent_card=False)
    agents = agent_store.all(shallow=True)
    batch_filename = create_batch_file(agents)
    
    try:
        file_id = upload_batch_file(batch_filename)
        batch_id = create_batch_job(file_id)
        batch_response = monitor_batch_job(batch_id)
        
        if batch_response and batch_response.status == "completed":
            results = retrieve_batch_results(batch_response)
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
