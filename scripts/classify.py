import os
import requests
from urllib.parse import quote
from xml.etree import ElementTree
from glob import glob
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class ArxivPaper:
    id: str
    categories: list[str] = field(default_factory=list)

    @property
    def primary_category(self) -> str:
        return self.categories[0]

def extract_categories_from_xml(xml_content: bytes) -> dict[str, list[str]]:
    """
    Extract categories from the XML content of an arXiv API response.
    
    Args:
        xml_content (str): The XML content of an arXiv API response
        
    Returns:
        dict: Mapping of paper IDs to their categories (primary category first)
    """

    root = ElementTree.fromstring(xml_content)
    
    categories = {}
    for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
        paper_id = entry.find('.//{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
        
        primary_category = entry.find('.//{http://arxiv.org/schemas/atom}primary_category').get('term')
        all_categories = [primary_category]  # Primary category first
        
        # Get all other categories
        for category in entry.findall('.//{http://www.w3.org/2005/Atom}category'):
            cat_term = category.get('term')
            if cat_term not in all_categories:
                all_categories.append(cat_term)
                
        categories[paper_id] = all_categories
        
    return categories

def classify_papers(paper_ids: set[str]) -> list[ArxivPaper]:
    """
    Get the arXiv categories for a list of paper IDs using the arXiv API.
    
    Args:
        paper_ids (set[str]): Set of arXiv paper IDs
        
    Returns:
        dict: Mapping of paper IDs to their categories
    """
    
    id_list = ','.join(quote(pid) for pid in paper_ids)
    response = requests.get(f"http://export.arxiv.org/api/query?id_list={id_list}&max_results=10000")
    if response.status_code != 200:
        raise Exception(f"ArXiv API request failed with status {response.status_code}")

    root = ElementTree.fromstring(response.content)
    
    papers = []
    for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
        paper_id = entry.find('.//{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
        
        primary_category = entry.find('.//{http://arxiv.org/schemas/atom}primary_category').get('term')
        all_categories = [primary_category]  # Primary category first
        
        # Get all other categories
        for category in entry.findall('.//{http://www.w3.org/2005/Atom}category'):
            cat_term = category.get('term')
            if cat_term not in all_categories:
                all_categories.append(cat_term)
                
        papers.append(ArxivPaper(paper_id, all_categories))
        
    return papers
    
if __name__ == "__main__":
    paper_ids = glob("papers/html/*/*.html")
    paper_ids = [paper.split("/")[-1] for paper in paper_ids]
    paper_ids = [paper[len("arXiv-"):len(paper)-len(".html")] for paper in paper_ids]
    paper_ids = set(paper_ids)
    print("PAPER IDS", len(paper_ids))

    papers: list[ArxivPaper] = []

    batch_size = 100
    for i in range(0, len(paper_ids), batch_size):
        batch = list(paper_ids)[i:i + batch_size]
        new_papers = classify_papers(batch)
        papers.extend(new_papers)
    
    print(papers)

   