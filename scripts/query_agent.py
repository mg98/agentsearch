import json
from colorama import Fore, Style
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agentsearch.agent.rag import retrieve

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

def main():
    query = input("Enter your query: ").strip()
    if not query:
        print("Query cannot be empty")
        return

    agent_id_str = input("Enter agent ID: ").strip()
    try:
        agent_id = int(agent_id_str)
    except ValueError:
        print(f"Invalid agent ID: {agent_id_str}")
        return

    print(f"\nRetrieving relevant chunks for agent {agent_id}...")
    try:
        documents = retrieve(agent_id, query)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # if not documents:
    #     print("No relevant chunks found for this query")
    #     return

    print(f"Found {len(documents)} relevant chunks\n")

    print(f"{Fore.LIGHTBLACK_EX}{'='*80}")
    print("CONTEXT CHUNKS:")
    print('='*80)
    for i, doc in enumerate(documents, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content)
    print('='*80 + Style.RESET_ALL)

    context = "\n\n".join([doc.page_content for doc in documents])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
    chain = prompt | llm

    print("\nGenerating answer...")
    response = chain.invoke({"context": context, "query": query})

    answer = response.content

    print("\nAnswer:")
    print(answer)

    print("\nEvaluating answer using FEB4RAG method...")
    scoring_messages = [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {"role": "user", "content": SCORING_USER_TEMPLATE.format(request=query, result=answer)}
    ]

    score_response = llm.invoke(scoring_messages)
    try:
        scores = json.loads(score_response.content)
        print(f"\nAnswer Scores: M={scores.get('M', 0)}, T={scores.get('T', 0)}, O={scores.get('O', 0)}")
    except json.JSONDecodeError:
        print("\nFailed to parse answer scores")

if __name__ == "__main__":
    main()
