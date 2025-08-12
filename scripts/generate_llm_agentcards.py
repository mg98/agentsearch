import sys
import csv
import os
import json
import time
from tqdm import tqdm
from google import genai
from google.genai import types
from google.genai.types import JobState
from agentsearch.dataset.agents import Agent, agents_df

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
    Create a batch request for the Gemini API.
    
    Args:
        agent_id: The agent ID to use as key
        publications: Concatenated string of all publications
    
    Returns:
        Dictionary containing the batch request
    """
    system_message = """
    You are an expert at analyzing academic publications and creating concise summaries.
    Based on the complete set of publications provided, create one comprehensive summary of the researcher's work.
    Write the summary in the style of an abstract that informs other researchers about this researcher's work.
    Do not refer to the researcher directly, just describe the work.
    """
    
    return {
        "key": agent_id,
        "request": {
            "contents": [
                {
                    "parts": [
                        {"text": system_message},
                        {"text": publications}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3
            }
        }
    }


def main():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    client = genai.Client(api_key=api_key)
    
    batch_requests_file = "batch_requests.jsonl"
    csv_file = "data/agentcards.csv"
    
    print("Preparing batch requests...")
    
    # Collect all batch requests
    batch_requests = []
    agent_info = {}
    
    for agent_id in tqdm(agents_df.index[:100], desc="Collecting publications"):
        agent = Agent.from_id(agent_id, shallow=True)
        publications = get_agent_publications(agent)
        
        if not publications:
            print(f"Warning: No publications found for {agent.name} (ID: {agent_id})")
            agent_info[agent_id] = {"name": agent.name, "error": "No publications found"}
            continue
        
        # Check if publications are too long (Gemini has token limits)
        # Approximate: 1 token ≈ 4 characters
        if len(publications) > 4000000:  # ~1M tokens
            # Truncate to fit within limits
            publications = publications[:4000000]
            print(f"Truncated publications for {agent.name} due to length")
        
        # Create batch request
        batch_request = create_batch_request(agent_id, publications)
        batch_requests.append(batch_request)
        agent_info[agent_id] = {"name": agent.name, "error": None}
    
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
    uploaded_file = client.files.upload(
        file=batch_requests_file,
        config=types.UploadFileConfig(display_name='my-batch-requests', mime_type='jsonl')
    )
    
    # Create batch job
    print("Creating batch job...")
    batch_job = client.batches.create(
        model="gemini-2.5-flash-lite",
        src=uploaded_file.name
    )
    
    print(f"Batch job created: {batch_job.name}")
    print("Waiting for batch job to complete...")
    
    while True:
        if batch_job.state.name in ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED']:
            break
        time.sleep(30)  # Wait 30 seconds before checking again
        batch_job = client.batches.get(name=batch_job.name)
        print(f"Batch job status: {batch_job.state}")
    
    print(f"Job finished with state: {batch_job.state.name}")
    
    results_content = client.files.download(file=batch_job.dest.file_name)
    
    # Parse results and write to CSV
    successful_count = 0
    failed_count = 0
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "agent_card"])
        
        # Process results
        for line in results_content.decode('utf-8').strip().split('\n'):
            result = json.loads(line)
            agent_id = result['key']
            
            if 'response' in result and 'candidates' in result['response']:
                # Success case
                content = result['response']['candidates'][0]['content']['parts'][0]['text']
                writer.writerow([agent_id, content])
                successful_count += 1
                agent_name = agent_info[agent_id]["name"]
                print(f"Generated card for {agent_name}")
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