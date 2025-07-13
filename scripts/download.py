import os
import shutil
import pandas as pd
from agentsearch.arxiv.api import find_papers, download_src

if __name__ == "__main__":

    authors = pd.read_csv("data/authors.csv", index_col=0)
    
    def process_author(id, author):
        papers = find_papers(id, author['name'])
        for paper in papers:
            if os.path.exists(f"papers/src/{id}/{paper.id}.tar.gz"):
                continue
            download_src(paper.id, f"papers/src/{id}")
        return papers
    
    # Create papers.csv file in write mode
    with open('data/papers.csv', 'w') as f:
        f.write('id,agent_id,title,categories\n')

        # author_papers = executor.map(process_author, 
        #                         authors.index,
        #                         authors.to_dict('records'))
        author_papers = []
        for id, author in authors.iterrows():
            papers = process_author(id, author)

            for p in papers:
                categories = str([cat.code for cat in p.categories])
                f.write(f"{p.id},{p.agent_id},{p.title},{categories}\n")