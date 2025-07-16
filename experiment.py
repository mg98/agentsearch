import numpy as np
from agentsearch.dataset.questions import Question
from agentsearch.dataset import agents
from agentsearch.agent import eval

Edge = tuple[int, int, np.ndarray, int] # (source_id, target_id, query_embedding, trust_score)
edges: list[Edge] = []

for question in Question.all():
    print(question.question)
    print(question.category.code)
    print("-"*100)

    # evaluate top-5 agents
    for match in agents.match_by_qid(question.id, 5):
        print(match.agent.name)
        print(match.agent.scholar_url)
        print(match.similarity_score)

        answer = match.agent.ask(question.question)
        print("Answer:", answer)

        grade, reason = eval.grade_answer(question.question, answer)
        print("Grade:", grade)
        print("Reason:", reason)
        print("-"*100)

        edges.append((1, match.agent.id, question.embedding, grade))

    break