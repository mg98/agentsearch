import os
import csv
import json
import time
from tqdm import tqdm
import openai
from openai import OpenAI
import tiktoken
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
        return ""

    all_publications = []

    for paper in agent.papers:
        if not paper.exists():
            continue

        chunks = paper.make_chunks()
        if chunks:
            pdf_content = "\n\n".join([chunk.page_content for chunk in chunks])
            all_publications.append(pdf_content)

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
        "custom_id": str(agent_id),
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
    agents = agent_store.all(shallow=True)
    encoding = tiktoken.encoding_for_model("gpt-4")

    batch_requests = []
    agent_names = {}
    
    # Process each agent
    # print(f"Processing {len(agents)} agents...")
    # for agent in tqdm(agents, desc="Processing publications"):
    #     publications = get_agent_publications(agent)

    #     if not publications:
    #         continue

    #     # Truncate if needed
    #     tokens = encoding.encode(publications)
    #     token_limit = 990_000
    #     if len(tokens) > token_limit:
    #         truncated_tokens = tokens[:token_limit]
    #         publications = encoding.decode(truncated_tokens)

    #     # Create batch request
    #     batch_request = create_batch_request(agent.id, publications)
    #     batch_requests.append(batch_request)
    #     agent_names[agent.id] = agent.name

    # if not batch_requests:
    #     print("No valid batch requests to process")
    #     return

    # # Write batch requests to JSONL file
    # with open(batch_requests_file, 'w', encoding='utf-8') as f:
    #     for request in batch_requests:
    #         f.write(json.dumps(request) + '\n')

    # print(f"Created {len(batch_requests)} batch requests")
    # import sys; sys.exit()

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
        time.sleep(30)
        batch_job = client.batches.retrieve(batch_job.id)
        print(f"Batch job status: {batch_job.status}")

    print(f"Job finished with status: {batch_job.status}")

    if batch_job.status != 'completed':
        print(f"Batch job failed with status: {batch_job.status}")
        return

    # Download results
    result_file_id = batch_job.output_file_id
    results_content = client.files.content(result_file_id).read()

    # Parse results and write to CSV
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Process results
        for line in results_content.decode('utf-8').strip().split('\n'):
            result = json.loads(line)
            agent_id = int(result['custom_id'])

            if 'response' in result and result['response'] and 'body' in result['response']:
                response_body = result['response']['body']
                if 'choices' in response_body and response_body['choices']:
                    content = response_body['choices'][0]['message']['content']
                    writer.writerow([agent_id, content])
                    print(f"Generated card for {agent_names.get(agent_id, agent_id)}")

    print(f"Saved to {csv_file}")

    # Clean up
    if os.path.exists(batch_requests_file):
        os.remove(batch_requests_file)


if __name__ == "__main__":
    main()