"""
LambdaMART baseline implementation for expert finding in community question answering.

Based on the TUEF paper: "Towards a Robust Expert Finding in Community Question Answering Platforms"
Uses LightGBM's LambdaMART implementation for learning-to-rank.
"""

import lightgbm as lgb
import numpy as np
from typing import Dict
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question, questions_df
import warnings
from tqdm import tqdm

LambdaMARTData = tuple[Agent, Agent, Question, float] # Source Agent, Target Agent, Question, Score

@dataclass
class LambdaMARTConfig:
    """Configuration for LambdaMART model following TUEF paper specifications."""
    # LightGBM LambdaMART parameters from paper
    objective: str = 'lambdarank'
    boosting_type: str = 'gbdt'
    metric: str = 'ndcg'
    num_leaves: int = 50  # Default from paper's hyperparameter range [50, 200]
    learning_rate: float = 0.1  # Default from paper's range [0.0001, 0.15]
    n_estimators: int = 100  # Default from paper's range [50, 150]
    max_depth: int = 10  # Default from paper's range [8, 15]
    min_data_in_leaf: int = 300  # Default from paper's range [150, 500]
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    verbose: int = -1
    random_state: int = 42

    # Feature extraction parameters
    max_features: int = 1000  # For TF-IDF vectorization
    test_size: float = 0.2
    validation_size: float = 0.2


class LambdaMARTExpertFinder:
    """
    LambdaMART-based expert finding system following the TUEF paper approach.

    This implementation adapts the TUEF methodology to work with the agentsearch
    dataset structure, using LightGBM's LambdaMART for learning-to-rank.
    """

    def __init__(self, config: LambdaMARTConfig = None):
        self.config = config or LambdaMARTConfig()
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_scaler = None
        self.feature_names = []
        # Pre-computed network statistics for fast feature extraction
        self.network_stats = {}

    def _precompute_network_stats(self, training_data: list[LambdaMARTData]):
        """Pre-compute network statistics for fast feature extraction."""
        print("Pre-computing network statistics...")

        # Initialize stats for each agent
        agent_stats = {}

        # Process training data once to compute all network metrics
        for source_agent, target_agent, _, score in tqdm(training_data, desc="Computing network stats"):
            # Initialize agent stats if not exists
            for agent in [source_agent, target_agent]:
                if agent.id not in agent_stats:
                    agent_stats[agent.id] = {
                        'in_degree_sources': set(),
                        'out_degree_targets': set(),
                        'scores_as_target': []
                    }

            # Track connections
            agent_stats[target_agent.id]['in_degree_sources'].add(source_agent.id)
            agent_stats[source_agent.id]['out_degree_targets'].add(target_agent.id)
            agent_stats[target_agent.id]['scores_as_target'].append(score)

        # Convert to final stats
        all_degrees = []
        for agent_id, stats in agent_stats.items():
            in_degree = len(stats['in_degree_sources'])
            out_degree = len(stats['out_degree_targets'])
            scores = stats['scores_as_target']

            self.network_stats[agent_id] = {
                'in_degree': in_degree,
                'out_degree': out_degree,
                'total_degree': in_degree + out_degree,
                'score_min': float(np.min(scores)) if scores else 0.0,
                'score_max': float(np.max(scores)) if scores else 0.0,
                'score_mean': float(np.mean(scores)) if scores else 0.0,
                'score_sum': float(np.sum(scores)) if scores else 0.0,
                'score_var': float(np.var(scores)) if scores else 0.0,
                'acceptance_ratio': float(sum(1 for s in scores if s > 0) / len(scores)) if scores else 0.0
            }
            all_degrees.append(in_degree + out_degree)

        # Compute max degree for eigenvector centrality approximation
        max_degree = max(all_degrees) if all_degrees else 1
        for agent_id in self.network_stats:
            self.network_stats[agent_id]['eigenvector_approx'] = float(
                self.network_stats[agent_id]['total_degree'] / max(max_degree, 1)
            )

        print(f"Pre-computed network stats for {len(self.network_stats)} agents")

    def extract_features(self, question: Question, agent: Agent, training_data: list[LambdaMARTData] = None) -> Dict[str, float]:
        """
        Extract features for question-agent pairs using specified feature set.

        Features:
        - TF, IDF, TF*IDF (min, max, mean, sum, variance)
        - BM25 score
        - Number of terms in query/agent card
        - Exact match and word match ratio
        - Network metrics (in_degree, out_degree, eigenvector)
        - Acceptance ratio and score statistics

        Args:
            question: Question object
            agent: Agent object
            training_data: List of LambdaMARTData for computing network features

        Returns:
            Dictionary of feature name -> value pairs
        """
        features = {}

        # Tokenize texts
        question_tokens = question.question.lower().split()
        agent_tokens = agent.agent_card.lower().split()

        # Basic text statistics
        features['num_terms_query'] = float(len(question_tokens))
        features['num_terms_agent'] = float(len(agent_tokens))

        # Exact match and word overlap
        question_words = set(question_tokens)
        agent_words = set(agent_tokens)

        features['is_exact_match'] = float(question.question.lower() in agent.agent_card.lower())

        if len(question_words) > 0:
            features['word_match_ratio'] = float(len(question_words & agent_words) / len(question_words))
        else:
            features['word_match_ratio'] = 0.0

        # TF-IDF features (meaningful query-document matching)
        if self.tfidf_vectorizer is not None:
            question_tfidf = self.tfidf_vectorizer.transform([question.question])
            agent_tfidf = self.tfidf_vectorizer.transform([agent.agent_card])

            question_vec = question_tfidf.toarray().flatten()
            agent_vec = agent_tfidf.toarray().flatten()

            # Cosine similarity between query and agent
            cosine_sim = np.dot(question_vec, agent_vec) / (np.linalg.norm(question_vec) * np.linalg.norm(agent_vec) + 1e-8)
            features['tfidf_cosine_similarity'] = float(cosine_sim)

            # TF-IDF overlap (element-wise product shows matching terms)
            tfidf_overlap = question_vec * agent_vec
            features['tfidf_overlap_sum'] = float(np.sum(tfidf_overlap))
            features['tfidf_overlap_max'] = float(np.max(tfidf_overlap))
            features['tfidf_overlap_mean'] = float(np.mean(tfidf_overlap))

            # Query TF-IDF characteristics (how distinctive are query terms)
            query_nonzero = question_vec[question_vec > 0]
            if len(query_nonzero) > 0:
                features['query_tfidf_max'] = float(np.max(query_nonzero))
                features['query_tfidf_mean'] = float(np.mean(query_nonzero))
                features['query_tfidf_sum'] = float(np.sum(query_nonzero))
                features['query_unique_terms'] = float(len(query_nonzero))
            else:
                features['query_tfidf_max'] = features['query_tfidf_mean'] = 0.0
                features['query_tfidf_sum'] = features['query_unique_terms'] = 0.0

            # Agent TF-IDF characteristics (how much content in agent card)
            agent_nonzero = agent_vec[agent_vec > 0]
            if len(agent_nonzero) > 0:
                features['agent_tfidf_max'] = float(np.max(agent_nonzero))
                features['agent_tfidf_mean'] = float(np.mean(agent_nonzero))
                features['agent_tfidf_sum'] = float(np.sum(agent_nonzero))
                features['agent_unique_terms'] = float(len(agent_nonzero))
            else:
                features['agent_tfidf_max'] = features['agent_tfidf_mean'] = 0.0
                features['agent_tfidf_sum'] = features['agent_unique_terms'] = 0.0

            # Coverage: what fraction of query terms appear in agent card
            if len(query_nonzero) > 0:
                matching_terms = np.sum(tfidf_overlap > 0)
                features['term_coverage'] = float(matching_terms / len(query_nonzero))
            else:
                features['term_coverage'] = 0.0

        # BM25 score (simplified implementation)
        if hasattr(self, 'bm25_scorer') and self.bm25_scorer is not None:
            features['bm25_score'] = float(self.bm25_scorer.get_scores(question_tokens)[agent.id] if agent.id < len(self.bm25_scorer.corpus) else 0.0)
        else:
            features['bm25_score'] = 0.0

        # Network features (using pre-computed stats for speed)
        if agent.id in self.network_stats:
            stats = self.network_stats[agent.id]
            features['in_degree'] = float(stats['in_degree'])
            features['out_degree'] = float(stats['out_degree'])
            features['score_min'] = stats['score_min']
            features['score_max'] = stats['score_max']
            features['score_mean'] = stats['score_mean']
            features['score_sum'] = stats['score_sum']
            features['score_var'] = stats['score_var']
            features['acceptance_ratio'] = stats['acceptance_ratio']
            features['eigenvector_approx'] = stats['eigenvector_approx']
        else:
            # Default values for agents not in training data
            features['in_degree'] = features['out_degree'] = 0.0
            features['score_min'] = features['score_max'] = 0.0
            features['score_mean'] = features['score_sum'] = features['score_var'] = 0.0
            features['acceptance_ratio'] = 0.0
            features['eigenvector_approx'] = 0.0

        return features

    def prepare_training_data(self, training_data: list[LambdaMARTData]) -> tuple:
        """
        Prepare training data in LightGBM LambdaMART format.

        Creates query-agent pairs with relevance labels and extracts features.
        Uses the score from LambdaMARTData as the relevance label.

        Args:
            training_data: List of LambdaMARTData for features and labels

        Returns:
            Tuple of (X, y, groups) where:
            - X: Feature matrix
            - y: Relevance scores from training data
            - groups: Number of candidates per question (for LambdaMART)
        """
        # Pre-compute network statistics for fast feature extraction
        self._precompute_network_stats(training_data)

        # Extract all unique agents and questions from training data
        all_agent_ids = set()
        agents_map = {}
        questions_map = {}
        for source_agent, target_agent, question, score in training_data:
            all_agent_ids.add(source_agent.id)
            all_agent_ids.add(target_agent.id)
            agents_map[source_agent.id] = source_agent
            agents_map[target_agent.id] = target_agent
            questions_map[question.id] = question

        all_agents = [agents_map[aid] for aid in all_agent_ids]
        questions = list(questions_map.values())

        # Fit TF-IDF vectorizer on all text data
        all_texts = []
        for question in questions:
            all_texts.append(question.question)
        for agent in all_agents:
            all_texts.append(agent.agent_card)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.tfidf_vectorizer.fit(all_texts)

        X_data = []
        y_data = []
        groups = []

        # Group training data by question
        question_groups = {}
        print("Grouping training data by question...")
        for source_agent, target_agent, question, score in tqdm(training_data, desc="Processing training samples"):
            if question.id not in question_groups:
                question_groups[question.id] = []
            question_groups[question.id].append((source_agent, target_agent, question, score))

        print("Extracting features...")
        for group_data in tqdm(question_groups.values(), desc="Processing question groups"):
            question_features = []
            question_labels = []

            # Extract features for each agent pair in this question group
            for source_agent, target_agent, question, score in group_data:
                features = self.extract_features(question, target_agent)
                question_features.append(list(features.values()))

                # Convert continuous score to binary relevance for LambdaMART
                # 0: negative/zero score, 1: positive score
                relevance = int(score > 0)
                question_labels.append(relevance)

            if not self.feature_names:
                # Store feature names from first question
                self.feature_names = list(self.extract_features(group_data[0][2], group_data[0][1]).keys())

            X_data.extend(question_features)
            y_data.extend(question_labels)
            groups.append(len(group_data))  # Number of candidates for this question

        X = np.array(X_data)
        y = np.array(y_data)

        # Scale features
        print("Scaling features...")
        self.feature_scaler = StandardScaler()
        X = self.feature_scaler.fit_transform(X)

        print(f"Prepared {len(X)} training samples with {X.shape[1]} features across {len(groups)} questions")
        return X, y, groups

    def train(self, training_data: list[LambdaMARTData]):
        """
        Train the LambdaMART model on question-agent pairs.

        Args:
            training_data: LambdaMARTData for features and labels
        """
        print("Preparing training data...")
        X, y, groups = self.prepare_training_data(training_data)

        # Split data while preserving groups
        train_X, val_X, train_y, val_y, train_groups, val_groups = self._split_grouped_data(
            X, y, groups, test_size=self.config.validation_size
        )

        print(f"Training on {len(train_groups)} questions, validating on {len(val_groups)} questions")

        # Create LightGBM datasets
        train_data = lgb.Dataset(train_X, label=train_y, group=train_groups)
        val_data = lgb.Dataset(val_X, label=val_y, group=val_groups, reference=train_data)

        # LambdaMART parameters from TUEF paper
        params = {
            'objective': self.config.objective,
            'boosting_type': self.config.boosting_type,
            'metric': self.config.metric,
            'num_leaves': self.config.num_leaves,
            'learning_rate': self.config.learning_rate,
            'max_depth': self.config.max_depth,
            'min_data_in_leaf': self.config.min_data_in_leaf,
            'feature_fraction': self.config.feature_fraction,
            'bagging_fraction': self.config.bagging_fraction,
            'bagging_freq': self.config.bagging_freq,
            'verbose': self.config.verbose,
            'random_state': self.config.random_state,
        }

        print("Training LambdaMART model...")
        with tqdm(total=self.config.n_estimators, desc="Training boosting rounds") as pbar:
            def log_evaluation_callback(env):
                pbar.update(1)
                if env.iteration % 20 == 0:
                    pbar.set_postfix({'iteration': env.iteration, 'valid_score': env.evaluation_result_list[-1][2]})

            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.config.n_estimators,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10),
                    log_evaluation_callback
                ]
            )

        print("Training completed!")

    def _split_grouped_data(self, X, y, groups, test_size=0.2):
        """Split data while preserving query groups."""
        # Calculate group boundaries
        group_boundaries = np.cumsum(groups)
        group_start_indices = np.concatenate([[0], group_boundaries[:-1]])

        # Split groups instead of individual samples
        num_groups = len(groups)
        group_indices = np.arange(num_groups)
        train_group_idx, val_group_idx = train_test_split(
            group_indices, test_size=test_size, random_state=self.config.random_state
        )

        # Get sample indices for each split
        train_indices = []
        val_indices = []

        for i in train_group_idx:
            start_idx = group_start_indices[i]
            end_idx = group_boundaries[i]
            train_indices.extend(range(start_idx, end_idx))

        for i in val_group_idx:
            start_idx = group_start_indices[i]
            end_idx = group_boundaries[i]
            val_indices.extend(range(start_idx, end_idx))

        train_X = X[train_indices]
        val_X = X[val_indices]
        train_y = y[train_indices]
        val_y = y[val_indices]
        train_groups = [groups[i] for i in train_group_idx]
        val_groups = [groups[i] for i in val_group_idx]

        return train_X, val_X, train_y, val_y, train_groups, val_groups

    def predict(self, question: Question, agents: list[Agent], training_data: list[LambdaMARTData] = None) -> list[tuple[Agent, float]]:
        """
        Predict relevance scores for question-agent pairs.

        Args:
            question: Question to find experts for
            agents: Candidate agents
            training_data: LambdaMARTData for network features

        Returns:
            List of (agent, score) tuples sorted by score (descending)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Extract features for all question-agent pairs
        features_list = []
        for agent in tqdm(agents, desc="Extracting features for prediction", disable=len(agents) < 100):
            features = self.extract_features(question, agent)
            features_list.append(list(features.values()))

        X = np.array(features_list)
        X = self.feature_scaler.transform(X)

        # Predict scores
        scores = self.model.predict(X)

        # Return agents sorted by score
        agent_scores = list(zip(agents, scores))
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        return agent_scores


def init_lambdamart(training_data: list[LambdaMARTData]) -> LambdaMARTExpertFinder:
    """
    Initialize and train a LambdaMART expert finder.

    Args:
        training_data: LambdaMARTData for features and labels

    Returns:
        Trained LambdaMARTExpertFinder instance
    """
    if not training_data:
        raise ValueError("Training data is required for LambdaMART")

    # Initialize and train model
    finder = LambdaMARTExpertFinder()
    finder.training_data = training_data  # Store for later use
    finder.train(training_data)

    return finder


def lambdamart_match(finder: LambdaMARTExpertFinder, agent_store: AgentStore, question: Question) -> list[Agent]:
    """
    Find expert agents for a question using trained LambdaMART model.

    Args:
        finder: Trained LambdaMARTExpertFinder instance
        agent_store: Store containing all agents
        question: Question to find experts for

    Returns:
        List of top 8 expert agents ranked by relevance
    """
    all_agents = agent_store.all(shallow=True)
    training_data = getattr(finder, 'training_data', None)
    agent_scores = finder.predict(question, all_agents, training_data)

    # Return top 8 agents
    top_agents = [agent for agent, _ in agent_scores[:8]]
    return top_agents