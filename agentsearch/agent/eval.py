from deepeval.metrics import DAGMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
    TaskNode,
)

# More specific criteria based on measurable elements
detail_assessment_node = NonBinaryJudgementNode(
    criteria=(
        "Classify the answer detail level based on the extracted statistics."
    ),
    children=[
        # VerdictNode(verdict="No facts, or claim that they do not know the answer", score=0),
        VerdictNode(verdict="Some facts but less than three examples", score=5),
        VerdictNode(verdict="Some facts and at least three examples", score=10),
    ],
)

extract_stats_node = TaskNode(
    instructions="""
    Count the number of unique factual statements that are relevant to the question. 
    Then, count the number of unique examples. 
    Examples include scenarios, solutions, algorithms, methods, challenges, benefits, and alike. 
    Output in format: 'Facts: X, Examples: Y'
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="Statistics",
    children=[detail_assessment_node]
)

info_provided_node = BinaryJudgementNode(
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    criteria="Does the answer provide actual information about the topic (rather than saying they don't know or can't answer)?",
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=extract_stats_node),
    ]
)

dag = DeepAcyclicGraph(root_nodes=[info_provided_node])

def grade_answer(question: str, answer: str, include_reason: bool = False) -> float:
    test_case = LLMTestCase(
        input=question,
        actual_output=answer
    )
    answer_quality = DAGMetric(name="Answer Quality", dag=dag, include_reason=include_reason, model="gpt-4.1-mini")
    answer_quality.measure(test_case)
    return answer_quality.score, answer_quality.reason
