import pytest
from agentsearch.agent.eval import grade_answer

q1 = "How do current breakthroughs in swarm intelligence contribute to the development of collaborative MAV networks for search and rescue missions?"

@pytest.mark.parametrize("question, answer, expected_grade", [
    (q1, "I do not know the answer", 0.0),
    (q1, "The sources do not provide actual information about the topic", 0.0),
    (q1, "I do not know. The provided sources focus on the challenges of computation-in-memory (CIM) and edge computing, including non-idealities, energy efficiency, and security aspects, but they do not address interoperability between diverse IoT devices and platforms in a heterogeneous Edge Computing environment.", 0.0),

    # simple answers
    (q1, "Machine learning methods have been explored for obtaining accurate inverse dynamic models of robot manipulators, which can be useful in developing more resilient and adaptive structural designs under uncertain environmental conditions. However, these methods require large amounts of training data and several iterations for learning, making it challenging to generalize them to new situations.", 
     0.5),
     (q1, """The current breakthroughs in swarm intelligence contribute to the development of collaborative MAV (Micro Aerial Vehicle) networks for search and rescue missions by enabling autonomous agents to work together effectively. This is achieved through the realization of self-regulating cyber-physical systems that leverage collaborative awareness, allowing multiple heterogeneous agents to interact and coordinate with each other.

In particular, our research focuses on developing a distributed form of self-aware intelligence, which enhances system performance, reliability, and adaptation through hierarchical learning models and energy-efficient technologies. This enables MAVs to operate in dynamic outdoor settings, such as search and rescue missions, where human involvement is limited due to the complexity of the environment.""",
     0.5),

     # extensive answers
     (q1,
      """The current breakthroughs in swarm intelligence contribute to the development of collaborative MAV (Micro Aerial Vehicle) networks for search and rescue missions by enabling autonomous agents to work together effectively. This is achieved through the realization of self-regulating cyber-physical systems that leverage collaborative awareness, allowing multiple heterogeneous agents to interact and coordinate with each other.

In particular, our research focuses on developing a distributed form of self-aware intelligence, which enhances system performance, reliability, and adaptation through hierarchical learning models and energy-efficient technologies. This enables MAVs to operate in dynamic outdoor settings, such as search and rescue missions, where human involvement is limited due to the complexity of the environment.""",
     1.0)
])      
def test_grade_answer(question, answer, expected_grade):
    grade, reason = grade_answer(question, answer)
    print(reason)
    assert grade == expected_grade
