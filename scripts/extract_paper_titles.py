#!/usr/bin/env python3

from pathlib import Path
import time
import pandas as pd
from agentsearch.apis import s2
import requests


def get_paper_titles_batch(paper_ids: list[str]) -> dict[str, str]:
    """Get paper titles for multiple papers using batch API."""
    try:
        response = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            params={'fields': 'title'},
            json={"ids": paper_ids},
            timeout=60
        )
        if response.status_code == 200:
            results = {}
            for paper in response.json():
                if paper and paper.get('paperId') and paper.get('title'):
                    results[paper['paperId']] = paper['title']
            return results
        else:
            print(f"  Warning: Batch request failed: {response.status_code}", flush=True)
            return {}
    except Exception as e:
        print(f"  Error in batch request: {e}", flush=True)
        return {}


def main():
    papers_base_dir = Path('papers/pdf')

    print("Collecting paper IDs...", flush=True)
    paper_agent_mapping = []

    agent_dirs = sorted([d for d in papers_base_dir.iterdir() if d.is_dir()])

    for agent_dir in agent_dirs:
        agent_id = agent_dir.name
        pdf_files = list(agent_dir.glob('*.pdf'))
        for pdf_file in pdf_files:
            paper_agent_mapping.append({
                'agent_id': agent_id,
                'paper_id': pdf_file.stem
            })

    all_paper_ids = list(dict.fromkeys([item['paper_id'] for item in paper_agent_mapping]))
    print(f"Found {len(all_paper_ids)} unique papers across {len(agent_dirs)} agents", flush=True)

    batch_size = 500
    titles_dict_all = {}

    for i in range(0, len(all_paper_ids), batch_size):
        batch = all_paper_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_paper_ids) + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} papers)...", flush=True)

        titles_dict = get_paper_titles_batch(batch)
        titles_dict_all.update(titles_dict)

        print(f"  Retrieved {len(titles_dict)} titles from this batch", flush=True)

        time.sleep(1)

    rows = []
    for item in paper_agent_mapping:
        paper_id = item['paper_id']
        if paper_id in titles_dict_all:
            rows.append({
                'agent_id': item['agent_id'],
                'title': titles_dict_all[paper_id]
            })

    df = pd.DataFrame(rows)

    output_file = Path('data/paper_titles.csv')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\nExtracted {len(rows)} paper-agent pairs with titles out of {len(paper_agent_mapping)} total", flush=True)
    print(f"Success rate: {100 * len(rows) / len(paper_agent_mapping):.1f}%", flush=True)
    print(f"Output written to: {output_file}", flush=True)


if __name__ == "__main__":
    main()
