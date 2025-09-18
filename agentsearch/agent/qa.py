# from langchain_ollama.llms import OllamaLLM
# from langchain_openai import OpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from agentsearch.agent.rag import retrieve

# DEBUG = False
# # model = OllamaLLM(model="llama3.1:8b", temperature=0)

# model = OpenAI(
#     model="gpt-4.1-nano",
#     temperature=0,
#     max_retries=2,
# )

# template = """
# You are a scientist answering questions based on provided sources. 
# Use only the information from the sources below to answer the question. 

# SOURCES:
# {sources}

# QUESTION: {question}

# When answering, do not mention the sources.
# Answer as if you were the author of the sources.
# Be clear and concise, get straight to the point.
# If no sources are provided, orthe sources don't contain enough information to answer the question, say that you do not know, and nothing else.

# ANSWER:
# """
# prompt = ChatPromptTemplate.from_template(template)
# chain = prompt | model


# if __name__ == "__main__":
#     author = input("Enter author: ")

#     while True:
#         print("\n-------------------------------")
#         question = input("Ask your question (q to quit): ")
#         if question == "q":
#             break
#         print()
        
#         sources = retrieve(author, question)
        
#         if DEBUG:
#             formatted_prompt = prompt.format(sources=sources, question=question)
#             print("=== DEBUG: FORMATTED PROMPT ===")
#             print(formatted_prompt)
#             print("=== END DEBUG ===")
#             print("\n")
        
#         result = chain.invoke({"sources": sources, "question": question})
#         print(result)