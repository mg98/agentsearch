import os
import pandas as pd
import ast
from agentsearch.arxiv.type import ArxivCategory, ArxivPaper

PAPERS_PATH = 'data/papers.csv'

def load_papers() -> list[ArxivPaper]:
    df = pd.read_csv(PAPERS_PATH)
    
    # filter authors that have papers
    agent_ids = [int(f) for f in os.listdir('papers/html') if os.path.isdir(os.path.join('papers/html', f))]
    papers = papers[papers['agent_id'].isin(agent_ids)]

    df['categories'] = df['categories'].apply(ast.literal_eval)
    df['categories'] = df['categories'].apply(lambda x: [ArxivCategory(cat) for cat in x])

    papers = []
    for _, row in df.iterrows():
        paper = ArxivPaper(
            id=row['id'],
            agent_id=row['agent_id'],
            title=row['title'], 
            categories=row['categories']
        )
        papers.append(paper)

    return papers

def save_papers(papers: list[ArxivPaper]):
    papers_data = [
        {
            'id': p.id,
            'agent_id': p.agent_id, 
            'title': p.title,
            'categories': str([cat.code for cat in p.categories])
        } for p in papers
    ]
    
    df = pd.DataFrame(papers_data)
    df.to_csv(PAPERS_PATH, index=False)

def load_categories() -> list[ArxivCategory]:
    df = pd.read_csv('data/arxiv_category_taxonomy.csv')
    
    categories = []
    for _, row in df.iterrows():
        category = ArxivCategory.from_code(row['code'])
        categories.append(category)

    return categories