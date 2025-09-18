import sys
import csv
import os
import json
import time
from tqdm import tqdm
import openai
from openai import OpenAI
import tiktoken
from multiprocessing import Pool, cpu_count
from agentsearch.dataset.agents import Agent, AgentStore, agents_df

def get_agent_publications(agent: Agent) -> str:
    """
    Read and concatenate all PDF publications for a specific agent.
    
    Args:
        agent: The Agent object
    
    Returns:
        Concatenated string of all publication content
    """
    agent.load_papers()
    
    if not agent.papers:
        print(f"No papers found for agent {agent.id}")
        return ""
    
    all_publications = []
    
    for paper in agent.papers:
        if not paper.exists():
            print(f"Paper file not found: {paper.path}")
            continue
            
        try:
            chunks = paper.make_chunks()
            if chunks:
                pdf_content = "\n\n".join([chunk.page_content for chunk in chunks])
                all_publications.append(pdf_content)
        except Exception as e:
            print(f"Error processing {paper.path}: {e}")
            continue
    
    if all_publications:
        return "\n\n---NEXT PAPER---\n\n".join(all_publications)
    else:
        return ""

def create_batch_request(agent_id: str, publications: str) -> dict:
    """
    Create a batch request for the OpenAI API.
    
    Args:
        agent_id: The agent ID to use as custom_id
        publications: Concatenated string of all publications
    
    Returns:
        Dictionary containing the batch request in OpenAI format
    """
    system_message = """
    You are an expert at analyzing academic publications and creating concise summaries.
    Based on the complete set of publications provided, create one comprehensive summary of the researcher's work.
    Write the summary in the style of an abstract that informs other researchers about this researcher's work.
    Do not refer to the researcher directly, just describe the work.
    """
    
    return {
        "custom_id": str(agent_id),  # Convert to string as required by OpenAI
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4.1-mini-2025-04-14",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": publications}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
    }


def process_agent(agent_data):
    """
    Worker function to process a single agent's publications.
    Returns tuple of (agent_id, agent_name, batch_request, error, was_truncated)
    """
    agent_id, use_llm_agent_card = agent_data
    
    try:
        # Create agent from store
        from agentsearch.dataset.agents import AgentStore
        agent_store = AgentStore(use_llm_agent_card=use_llm_agent_card)
        agent = agent_store.from_id(agent_id, shallow=True)
        
        # Get publications
        publications = get_agent_publications(agent)
        
        if not publications:
            return (agent.id, agent.name, None, "No publications found", False)
        
        # Truncate if needed
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(publications)
        was_truncated = False
        
        token_limit = 990_000
        if len(tokens) > token_limit:
            was_truncated = True
            truncated_tokens = tokens[:token_limit]
            publications = encoding.decode(truncated_tokens)
        
        # Create batch request
        batch_request = create_batch_request(agent.id, publications)
        
        return (agent.id, agent.name, batch_request, None, was_truncated)
        
    except Exception as e:
        return (agent_id, None, None, str(e), False)


def main():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    batch_requests_file = "batch_requests.jsonl"
    csv_file = "data/agentcards.csv"
    
    print("Preparing batch requests...")
    
    # Get all agent IDs
    agent_store = AgentStore(use_llm_agent_card=False)
    agent_ids = [agent.id for agent in agent_store.all(shallow=True)]
    
    # Prepare data for parallel processing
    agent_data = [(agent_id, False) for agent_id in agent_ids]
    
    # Process agents in parallel
    num_workers = min(cpu_count(), 8)  # Cap at 8 workers to avoid overwhelming the system
    print(f"Processing {len(agent_ids)} agents using {num_workers} parallel workers...")
    
    batch_requests = []
    agent_info = {}
    truncated_count = 0
    
    with Pool(processes=num_workers) as pool:
        # Process agents in parallel with progress bar
        results = list(tqdm(
            pool.imap(process_agent, agent_data),
            total=len(agent_data),
            desc="Processing publications"
        ))
    
    # Process results
    for agent_id, agent_name, batch_request, error, was_truncated in results:
        if error:
            if agent_name:
                print(f"Warning: {agent_name} (ID: {agent_id}): {error}")
            agent_info[agent_id] = {"name": agent_name, "error": error}
        else:
            if was_truncated:
                truncated_count += 1
                print(f"Truncated publications for {agent_name}")
            
            if batch_request:
                batch_requests.append(batch_request)
                agent_info[agent_id] = {"name": agent_name, "error": None}
    
    print(f"\nTruncated {truncated_count} agents out of {len(agent_ids)} agents ({truncated_count/len(agent_ids)*100:.2f}%)")
    
    if not batch_requests:
        print("No valid batch requests to process")
        return
    
    # Write batch requests to JSONL file
    with open(batch_requests_file, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"Created {len(batch_requests)} batch requests")
    
    # Upload batch requests file
    print("Uploading batch requests...")
    with open(batch_requests_file, 'rb') as file:
        batch_input_file = client.files.create(
            file=file,
            purpose="batch"
        )
    
    # Create batch job
    print("Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    print(f"Batch job created: {batch_job.id}")
    print("Waiting for batch job to complete...")
    
    while batch_job.status not in ['completed', 'failed', 'cancelled', 'expired']:
        time.sleep(30)  # Wait 30 seconds before checking again
        batch_job = client.batches.retrieve(batch_job.id)
        print(f"Batch job status: {batch_job.status}")
    
    print(f"Job finished with status: {batch_job.status}")
    
    if batch_job.status != 'completed':
        print(f"Batch job failed with status: {batch_job.status}")
        if batch_job.errors:
            print(f"Errors: {batch_job.errors}")
        return
    
    # Download results
    result_file_id = batch_job.output_file_id
    results_content = client.files.content(result_file_id).read()
    
    # Parse results and write to CSV
    successful_count = 0
    failed_count = 0
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # writer.writerow(["agent_id", "agent_card"])
        
        # Process results
        for line in results_content.decode('utf-8').strip().split('\n'):
            result = json.loads(line)
            agent_id = int(result['custom_id'])  # Convert back to int for lookup
            
            if 'response' in result and result['response'] and 'body' in result['response']:
                # Success case
                response_body = result['response']['body']
                if 'choices' in response_body and response_body['choices']:
                    content = response_body['choices'][0]['message']['content']
                    writer.writerow([agent_id, content])
                    successful_count += 1
                    agent_name = agent_info[agent_id]["name"]
                    print(f"Generated card for {agent_name}")
                else:
                    error_msg = "No content in response"
                    writer.writerow([agent_id, f"ERROR: {error_msg}"])
                    failed_count += 1
                    agent_name = agent_info[agent_id]["name"]
                    print(f"Failed to generate card for {agent_name}: {error_msg}")
            else:
                # Error case
                error_msg = result.get('error', {}).get('message', 'Unknown error')
                writer.writerow([agent_id, f"ERROR: {error_msg}"])
                failed_count += 1
                agent_name = agent_info[agent_id]["name"]
                print(f"Failed to generate card for {agent_name}: {error_msg}")
        
        # Add agents with no publications
        for agent_id, info in agent_info.items():
            if info["error"]:
                writer.writerow([agent_id, f"ERROR: {info['error']}"])
                failed_count += 1
    
    print(f"\nSuccessfully generated {successful_count} agent cards")
    print(f"Failed: {failed_count}")
    print(f"Saved to {csv_file}")
    
    # Clean up local file
    os.remove(batch_requests_file)

if __name__ == "__main__":
    main()