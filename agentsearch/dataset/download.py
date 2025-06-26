import requests
import os
import shutil

def download_paper(paper_id: str, export_path: str):
    """
    Download paper source (tar.gz) and save it to the export path.

    Args:
        paper_id (str): The arXiv ID of the paper to download.
        export_path (str): The path to save the paper to.

    Returns:
    """
    response = requests.get(f"https://arxiv.org/src/{paper_id}", stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download: HTTP {response.status_code}")
        
    if response.headers.get('Content-Type') != 'application/gzip':
        print(f"Paper {paper_id} has no source: {response.headers.get('Content-Type')}")
        return
    
    content_disposition = response.headers.get('Content-Disposition')
    if not content_disposition:
        raise Exception("No Content-Disposition header found")
        
    filename = content_disposition.split('filename=')[-1].strip('"')
    if not filename:
        raise Exception("No filename found in Content-Disposition header")
    
    if not os.path.exists(export_path):
        os.makedirs(export_path)
        
    with open(os.path.join(export_path, filename), 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                
    return filename

def get_paper_ids(author_id: str) -> list[str]:
    """
    Get arXiv paper IDs for an author using Semantic Scholar (S2) API.

    Args:
        author_id (str): The ID of the author in S2.

    Returns:
        list[str]: A list of arXiv paper IDs.
    """
    url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}?fields=papers.externalIds"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch author papers: HTTP {response.status_code}")
        
    data = response.json()
    paper_ids = []
    for paper in data["papers"]:
        if paper.get("externalIds") and paper["externalIds"].get("ArXiv"):
            paper_ids.append(paper["externalIds"]["ArXiv"])
            
    return paper_ids

if __name__ == "__main__":
    
    author_ids = [
        1800677,    # Johan Pouwelse
        1394550477,    # Rowdy Chotkan
        2265490493  # Maarten de Rijke
    ]

    shutil.rmtree("papers/src", ignore_errors=True)

    for author_id in author_ids:
        paper_ids = get_paper_ids(author_id)
        if len(paper_ids) == 0:
            print(f"No papers found for author {author_id}")
            continue
        
        for paper_id in paper_ids:
            print(author_id, paper_id)
            download_paper(paper_id, f"papers/src/{author_id}")
