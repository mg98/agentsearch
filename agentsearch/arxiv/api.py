import requests
import os
from xml.etree import ElementTree
import re
from agentsearch.arxiv.type import ArxivPaper, ArxivCategory

def find_papers(agent_id: int, agent_name: str) -> list[ArxivPaper]:
    """
    Find all publications for an author.
    """
    items_per_page = 100
    start = 0
    total_results = 1
    papers = []

    while total_results > start:
        response = requests.get(f'http://export.arxiv.org/api/query?search_query=au:"{agent_name}"&max_results={items_per_page}&start={start}')
        if response.status_code != 200:
            raise Exception(f"Failed to find publications: HTTP {response.status_code}")
        
        root = ElementTree.fromstring(response.content)
        namespaces = {'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'}
        total_results = root.find('opensearch:totalResults', namespaces)
        if total_results is None:
            raise Exception("Failed to parse XML: totalResults not found")
        total_results = int(total_results.text)
        print(f"Found {total_results} publications for {agent_name}")

        for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
            abs_url = entry.find('.//{http://www.w3.org/2005/Atom}id').text
            print(abs_url)
            arxiv_id = abs_url[len("http://arxiv.org/abs/"):]
            title = entry.find('.//{http://www.w3.org/2005/Atom}title').text

            primary_category = entry.find('.//{http://arxiv.org/schemas/atom}primary_category').get('term')
            categories = [primary_category]
            for category in entry.findall('.//{http://www.w3.org/2005/Atom}category'):
                cat_term = category.get('term')
                if cat_term not in categories:
                    categories.append(cat_term)

            categories = [ArxivCategory.from_code(code) for code in categories]
            papers.append(ArxivPaper(arxiv_id, agent_id, title, categories))
        
        start += items_per_page

    return papers

def download_src(paper_id: str, export_path: str):
    """
    Download paper source (tar.gz) and save it to the export path.

    Args:
        paper_id (str): The arXiv ID of the paper to download.
        export_path (str): The path to save the paper to.

    Returns:
    """
    response = requests.get(f"https://arxiv.org/src/{paper_id}", stream=True)
    if response.status_code == 403:
        print(f"Source not public for {paper_id} (403)")
        return
    if response.status_code != 200:
        print(f"Failed to download {paper_id}: HTTP {response.status_code}")
        return
        
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