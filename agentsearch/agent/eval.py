from deepeval.metrics import DAGMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
    TaskNode,
)

# First, extract objective measures
extract_stats_node = TaskNode(
    instructions="Count the number of sentences, paragraphs, examples, and key concepts mentioned in the answer. Output in format: 'Sentences: X, Paragraphs: Y, Examples: Z, Key concepts: V'",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="Answer Statistics",
    children=[]
)

# More specific criteria based on measurable elements
detail_assessment_node = NonBinaryJudgementNode(
    criteria="Based on the Answer Statistics, classify the answer detail level.",
    children=[
        VerdictNode(verdict="Simple: One paragraph, less than 5 sentences", score=6),
        VerdictNode(verdict="Nuanced: Multiple paragraphs and key concepts", score=8),
        VerdictNode(verdict="Extensive: Multiple examples, multiple paragraphs, at least 5 key concepts", score=10),
    ],
)

# Connect extract_stats_node to detail_assessment_node
extract_stats_node.children = [detail_assessment_node]

# Check if any information was provided
info_provided_node = BinaryJudgementNode(
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    criteria="Does the answer provide actual information about the topic (rather than saying they don't know or can't answer)?",
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=extract_stats_node),
    ]
)

dag = DeepAcyclicGraph(root_nodes=[info_provided_node])

def grade_answer(question: str, answer: str) -> float:
    test_case = LLMTestCase(
        input=question,
        actual_output=answer
    )
    answer_quality = DAGMetric(name="Answer Quality", dag=dag, model="gpt-4o")
    answer_quality.measure(test_case)
    return answer_quality.score, answer_quality.reason
