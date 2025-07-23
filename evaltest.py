from deepeval.test_case import LLMTestCase
from agentsearch.agent.eval import grade_answer


johans_answer = """
The proposed solutions include:

1. EigenTrust, which requires an initial notion of trust consisting of known trustworthy peers.
2. A mechanism based on indirect reciprocity, which takes into account indirect contributions and local connections to assess trustworthiness.

The latter approach is considered more advanced than tit-for-tat strategy and overcomes issues of misreporting and manipulation by Sybils. It also uses the MeritRank algorithm to assign subjective trust scores to nodes.

The implementation follows a similar approach to EigenTrust, but with an added restriction on locality, which reduces the amount of indirection in between contributing and reciprocating peers.
"""

rowdys_answer = """
Our proposed solution is based on an evolutionary mechanism referred to as indirect reciprocity, which takes into account indirect contributions when assessing the trustworthiness of nodes. This mechanism also accounts for the trustworthiness of peers with their local connections, making locality a necessary and sufficient condition in achieving sustained cooperation for peer-to-peer networks.
"""

maartens_answer = """
I don't know. The provided sources don't contain information on solving trust in a peer-to-peer network. They discuss concepts related to trust bias in recommender systems, specifically how to correct for it using affine estimators. However, they don't provide guidance on building or maintaining trust in peer-to-peer networks.
"""

for answer in [johans_answer, rowdys_answer, maartens_answer]:
    grade, reason = grade_answer("how to solve trust in a peer to peer network?", answer)
    print(grade)
    print(reason)