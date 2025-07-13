from dataclasses import dataclass
import pandas as pd
from ast import literal_eval
import os

arxiv_category_df = pd.read_csv('data/arxiv_category_taxonomy.csv')

@dataclass
class ArxivCategory:
    code: str
    name: str
    description: str

    @classmethod
    def from_code(cls, code: str) -> 'ArxivCategory':
        if code not in arxiv_category_df['code'].values:
            return None
        return cls(
            code=code, 
            name=arxiv_category_df.loc[arxiv_category_df['code'] == code, 'name'].values[0], 
            description=arxiv_category_df.loc[arxiv_category_df['code'] == code, 'description'].values[0]
            )

papers_df = pd.read_csv('data/papers.csv')
papers_df['categories'] = papers_df['categories'].apply(literal_eval)
papers_df['categories'] = papers_df['categories'].apply(lambda x: [ArxivCategory.from_code(cat) for cat in x])
papers_df['categories'] = papers_df['categories'].apply(lambda x: [cat for cat in x if cat is not None])
    
@dataclass
class ArxivPaper:
    id: str
    agent_id: int
    title: str
    categories: list[ArxivCategory]

    @property
    def primary_category(self) -> ArxivCategory:
        return self.categories[0]
    
    @property
    def src_url(self) -> str:
        return f"https://arxiv.org/src/{self.id}"
    
    @classmethod
    def from_id(cls, id: str) -> 'ArxivPaper':
        paper_df = papers_df[papers_df['id'] == id]
        if len(paper_df) == 0:
            raise ValueError(f"No paper found for ID: {id}")
        paper_df = paper_df.iloc[0]
        return cls(
            id=id,
            agent_id=paper_df['agent_id'],
            title=paper_df['title'],
            categories=paper_df['categories']
        )