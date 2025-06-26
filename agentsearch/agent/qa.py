from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from agentsearch.agent.rag import retrieve

DEBUG = False
model = OllamaLLM(model="llama3.2:3b")

template = """
You are a scientist answering questions based on provided sources. 
Use only the information from the sources below to answer the question. 
If the sources don't contain enough information to answer the question, say that you don't know.
Answer as if you are the author of those sources.

Sources:
{sources}

Question: {question}

Be clear and concise, get straight to the point.

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

author = input("Enter author (johan, rowdy, maarten): ")
author_id = {
    "johan": 1800677,
    "rowdy": 1394550477,
    "maarten": 2265490493
}[author]

while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    if question == "q":
        break
    print()
    
    sources = retrieve(author_id, question)
    
    if DEBUG:
        formatted_prompt = prompt.format(sources=sources, question=question)
        print("=== DEBUG: FORMATTED PROMPT ===")
        print(formatted_prompt)
        print("=== END DEBUG ===")
        print("\n")
    
    result = chain.invoke({"sources": sources, "question": question})
    print(result)