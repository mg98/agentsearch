import pandas as pd
from dataclasses import dataclass
from ast import literal_eval
import os
from agentsearch.agent.rag import retrieve, retrieve_with_embedding
import numpy as np
from agentsearch.dataset.papers import Paper
from agentsearch.utils.globals import embeddings
import warnings
from agentsearch.dataset.questions import Question
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agentsearch.utils.vector_store import load_index, search_collection, get_agent_embedding

I_DONT_KNOW_ANSWER = "I don't know"

SCORING_SYSTEM_PROMPT = """Given a user request and a search result, you must provide a score on an
integer scale of 0 to 3 with the following meanings:
3 = key, this search result contains relevant, diverse, informative and
correct answers to the user request; the user request can be fulfilled by
relying only on this search result.
2 = high relevance, this search result contains relevant, informative and
correct answers to the user request; however, it does not span diverse
perspectives, and including another perspective can help with a better
answer to the user request.
1 = minimal relevance, this search result contains relevant answers to the
user request. However, it is impossible to answer the user request based
solely on the search result.
0 = not relevant, this search result does not contain any relevant answer
to the user request."""

SCORING_USER_TEMPLATE = """Assume that you are collecting all the relevant search results to write a
final answer for the user request.
User Request:
A user typed the following request.
{request}
Result:
Consider the following search result:
—BEGIN Search Result CONTENT—
{result}
—END Search Result CONTENT—
Instructions:
Split this problem into steps:
Consider the underlying intent of the user request.
Measure how well the content matches a likely intent of the request (M)
Measure how trustworthy the search result is (T).
Consider the aspects above and the relative importance of each, and
decide on a final score (O).
Produce a JSON of scores without providing any reasoning. Example:{{"M": 2, "T": 1,"O": 1}}
Results:"""

ANSWER_TEMPLATE = """You are an AI assistant that answers questions based solely on the provided context.

Context:
{context}

Question: {query}

Answer the question based only on the information provided in the context above. If the context does not contain enough information to answer the question, say so clearly."""

agents_df = pd.read_csv('data/agents.csv', index_col=0)
if os.path.exists("papers/pdf"):
    agents_df = agents_df[agents_df.index.astype(str).isin(os.listdir("papers/pdf"))]
else:
    warnings.warn("no papers/pdf directory found")
agents_df['research_fields'] = agents_df['research_fields'].apply(literal_eval)
agents_df = agents_df[agents_df['research_fields'].apply(len) > 0]
agents_df = agents_df[agents_df['name'].notna()].copy()
agents_df = agents_df.sample(frac=1, random_state=42)

# Load LLM-generated agent cards
agentcards_df = pd.read_csv('data/agentcards.csv', index_col=0)
agentcards_df = agentcards_df.reindex(agents_df.index)

def num_sources_to_score(num_sources: int) -> float:
    if num_sources >= 100:
        return 1.0
    return np.log(num_sources + 1) / np.log(101)

class Agent:
    def __init__(self, id: int, name: str, citation_count: int, scholar_url: str,
                 agent_card: str, faculty: str, department: str, group: str,
                 collection: str = "agents"):
        self.id = id
        self.name = name
        self.citation_count = citation_count
        self.scholar_url = scholar_url
        self.agent_card = agent_card
        self.faculty = faculty
        self.department = department
        self.group = group
        self._collection = collection
        self._embedding = None
        self._papers = None

    @property
    def embedding(self) -> np.ndarray:
        if self._embedding is None:
            self._embedding = get_agent_embedding(self.id, self._collection)
        return self._embedding

    @property
    def papers(self) -> list[Paper]:
        if self._papers is None:
            papers_dir = f'papers/pdf/{self.id}'
            if not os.path.exists(papers_dir):
                self._papers = []
            else:
                paper_ids = [f[:-len('.pdf')] for f in os.listdir(papers_dir) if f.endswith('.pdf')]
                self._papers = [Paper(id=paper_id, agent_id=self.id) for paper_id in paper_ids]
        return self._papers

    @classmethod
    def from_id(cls, id: int, collection: str = "agents") -> 'Agent':
        """
        Load an agent by ID

        Args:
            id: Agent ID
            collection: FAISS collection name to use for embeddings
        """
        use_llm_agent_card = collection == "agents_with_llm_agent_cards"

        if use_llm_agent_card:
            agent_card = agentcards_df.loc[id, 'agent_card']
            if type(agent_card) != str:
                agent_card = ', '.join(agents_df.loc[id, 'research_fields'])
        else:
            agent_card = ', '.join(agents_df.loc[id, 'research_fields'])

        return cls(
            id=id,
            name=agents_df.loc[id, 'name'],
            citation_count=agents_df.loc[id, 'citation_count'],
            scholar_url=agents_df.loc[id, 'scholar_url'],
            agent_card=agent_card,
            faculty=agents_df.loc[id, 'faculty'],
            department=agents_df.loc[id, 'department'],
            group=agents_df.loc[id, 'group'],
            collection=collection
        )

    @classmethod
    def all(cls, collection: str = "agents") -> list['Agent']:
        """Get all agents"""
        agent_ids = agents_df.index.tolist()
        return [cls.from_id(id, collection) for id in agent_ids]

    @classmethod
    def all_from_cluster(cls, topic: str, size: int, collection: str = "agents") -> list['Agent']:
        """Get agents similar to a topic"""
        topic_embedding = np.array(embeddings.embed_query(topic))
        _, indices = search_collection(collection, topic_embedding, k=size)
        _, metadata = load_index(collection)

        agents = []
        for idx in indices:
            if idx != -1:
                agent_id = int(metadata['ids'][idx])
                agents.append(cls.from_id(agent_id, collection=collection))

        return agents

    @classmethod
    def match(cls, question: Question, top_k: int = 1, collection: str = "agents") -> list['AgentMatch']:
        """
        Match a question to the most similar agents using FAISS

        Args:
            question: The Question object to match
            top_k: Number of top matches to return
            collection: FAISS collection name to use

        Returns:
            List of agent matches
        """
        question_embedding = np.array([question.embedding]).astype('float32')
        index, metadata = load_index(collection)
        distances, indices = index.search(question_embedding, top_k)

        matches: list[AgentMatch] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            agent_id = int(metadata['ids'][idx])
            distance = float(distances[0][i])
            agent = cls.from_id(agent_id, collection=collection)
            matches.append(AgentMatch(
                agent=agent,
                distance=distance
            ))

        return matches


    def ask_string(self, question: str) -> str:
        sources = retrieve(self.id, question)
        if len(sources) == 0:
            return "I don't know"

        context = "\n\n".join([source.page_content for source in sources])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
        chain = prompt | llm

        response = chain.invoke({"context": context, "query": question})
        return response.content
    
    def count_sources(self, question: str) -> int:
        sources = retrieve(self.id, question)
        return len(sources)
    
    def has_sources(self, question: str) -> bool:
        sources = retrieve(self.id, question, k=1)
        return len(sources) > 0

    def ez_grade(self, question: Question) -> float:
        sources = retrieve_with_embedding(self.id, question.embedding, k=100)
        return num_sources_to_score(len(sources))

    def ask(self, question: Question) -> str:
        sources = retrieve_with_embedding(self.id, question.embedding, k=10)
        if len(sources) == 0:
            return I_DONT_KNOW_ANSWER

        context = "\n\n".join([source.page_content for source in sources])

        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
        prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
        chain = prompt | llm

        response = chain.invoke({"context": context, "query": question.text})
        return response.content

    def grade(self, question: Question) -> int:
        answer = self.ask(question)
        if answer == I_DONT_KNOW_ANSWER:
            return 0

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        scoring_messages = [
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": SCORING_USER_TEMPLATE.format(request=question.text, result=answer)}
        ]

        response = llm.invoke(scoring_messages)
        try:
            scores = json.loads(response.content)
            return scores.get("O", 0)
        except json.JSONDecodeError:
            warnings.warn(f"Failed to parse scores for agent {self.id} and question {question.id}")
            return 0

@dataclass
class AgentMatch:
    agent: Agent
    distance: float