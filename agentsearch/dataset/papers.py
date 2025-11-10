import os
from dataclasses import dataclass
from agentsearch.utils.parse import chunk_pdf
from langchain_core.documents import Document

BASE_DIR = "papers/pdf"
@dataclass
class Paper:
    id: str
    agent_id: int

    @property
    def path(self) -> str:
        return f'{BASE_DIR}/{self.agent_id}/{self.id}.pdf'

    @classmethod
    def from_id(cls, id: str) -> 'Paper':
        for agent_id_str in os.listdir(BASE_DIR):
            agent_dir = os.path.join(BASE_DIR, agent_id_str)
            if not os.path.isdir(agent_dir):
                continue
            candidate_path = os.path.join(agent_dir, f"{id}.pdf")
            if os.path.exists(candidate_path):
                agent_id = int(agent_id_str)
                return cls(id=id, agent_id=agent_id)
        raise FileNotFoundError(f"Could not find paper with id {id}")
    
    def exists(self) -> bool:
        return os.path.exists(self.path)

    def make_chunks(self) -> list[str]:
        chunks = chunk_pdf(self.path)
        return [Document(page_content=chunk) for chunk in chunks]

    def extract_text(self) -> str:
        chunks = chunk_pdf(self.path)
        return "\n\n".join(chunks)
