from dataclasses import dataclass
from agentsearch.arxiv.type import ArxivCategory

@dataclass
class ArxivPaper:
    id: str
    author: str
    title: str
    categories: list[ArxivCategory]

    @property
    def primary_category(self) -> ArxivCategory:
        return self.categories[0]
    
    @property
    def src_url(self) -> str:
        return f"https://arxiv.org/src/{self.id}"
    