import os
from dataclasses import dataclass
from agentsearch.utils.parse import chunk_pdf
from langchain_core.documents import Document

@dataclass
class Paper:
    id: str
    agent_id: int

    @property
    def path(self) -> str:
        return f'papers/pdf/{self.agent_id}/{self.id}.pdf'
    
    def exists(self) -> bool:
        return os.path.exists(self.path)

    def make_chunks(self) -> list[str]:
        chunks = chunk_pdf(self.path)
        return [Document(page_content=chunk) for chunk in chunks]
