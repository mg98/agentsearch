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
            # headers={'x-api-key': s2.S2_API_KEY},
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
    all_paper_ids = []

    agent_dirs = sorted([d for d in papers_base_dir.iterdir() if d.is_dir()])

    for agent_dir in agent_dirs:
        pdf_files = list(agent_dir.glob('*.pdf'))
        for pdf_file in pdf_files:
            all_paper_ids.append(pdf_file.stem)

    all_paper_ids = list(dict.fromkeys(all_paper_ids))
    print(f"Found {len(all_paper_ids)} unique papers across {len(agent_dirs)} agents", flush=True)

    batch_size = 500
    all_titles = []

    for i in range(0, len(all_paper_ids), batch_size):
        batch = all_paper_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_paper_ids) + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} papers)...", flush=True)

        titles_dict = get_paper_titles_batch(batch)

        for paper_id in batch:
            if paper_id in titles_dict:
                all_titles.append(titles_dict[paper_id])

        print(f"  Retrieved {len(titles_dict)} titles from this batch", flush=True)

        time.sleep(1)

    unique_titles = list(dict.fromkeys(all_titles))

    output_file = Path('paper_titles.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for title in unique_titles:
            f.write(f"{title}\n")

    print(f"\nExtracted {len(all_titles)} paper titles ({len(unique_titles)} unique) out of {len(all_paper_ids)} total papers", flush=True)
    print(f"Success rate: {100 * len(all_titles) / len(all_paper_ids):.1f}%", flush=True)
    print(f"Output written to: {output_file}", flush=True)


if __name__ == "__main__":
    main()
