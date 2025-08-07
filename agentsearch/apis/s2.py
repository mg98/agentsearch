import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

S2_API_KEY = os.environ.get('S2_API_KEY')
if S2_API_KEY is None:
    raise ValueError("S2_API_KEY environment variable must be set")

# See: https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_get_paper
ExternalPaperIDs = list[dict[str, str]]
Paper = tuple[str, ExternalPaperIDs]

def match_profile(agent_row: pd.Series) -> str:
    # Query Semantic Scholar API to find matching authors
    name = agent_row['name']
    # Remove Dr/Dr. prefix if present
    name = name.replace('Dr.', '').replace('Dr ', '').strip()
    # Remove any suffix after a comma (e.g. ", MBA")
    name = name.split(',')[0].strip()
    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/author/search",
        params={'query': name, 'fields': 'citationCount'},
        headers={'x-api-key': S2_API_KEY},
        timeout=30
    )
    assert response.status_code == 200, f"failed to find s2 profile for {name}: {response.status_code}"
    data = response.json()
    if 'next' in data:
        raise Exception("Semantic Scholar API returned more than 100 results")
    
    # Find the candidate with citation count closest to agent's citation count
    candidates = data['data']
    closest_match = None
    min_diff = float('inf')
    
    for candidate in candidates:
        candidate_citations = candidate.get('citationCount', 0)
        if candidate_citations is None:
            continue
            
        diff = abs(candidate_citations - agent_row['citation_count'])
        if diff < min_diff:
            min_diff = diff
            closest_match = candidate
    
    assert closest_match is not None, "no match found"
    return closest_match['authorId']


def get_papers(s2_profile_id: str) -> list[Paper]:
    """
    Retrieves all papers from a Semantic Scholar profile. 
    Returns a list of S2 paper IDs and their external IDs, e.g., DOI or ArXiv (keys not guaranteed).
    """
    results = []
    data = {"next": 0}

    while 'next' in data:
        response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/author/{s2_profile_id}/papers",
            params={'fields': 'externalIds', 'offset': data['next']},
            headers={'x-api-key': S2_API_KEY},
            timeout=30
        )
        assert response.status_code == 200, f"failed to get papers for {s2_profile_id}: {response.status_code}"
        data = response.json()
        results.extend([(result['paperId'], result['externalIds']) for result in data['data']])

    return results

