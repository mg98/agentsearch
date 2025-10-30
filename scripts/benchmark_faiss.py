import argparse
import faiss
import json
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from agentsearch.dataset.questions import Question
from agentsearch.utils.eval import load_test_questions
from agentsearch.utils.globals import THRESHOLD
from tqdm import tqdm

@dataclass
class RetrievalMetrics:
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    mrr: float
    query_time_ms: float
    num_queries: int

@dataclass
class BenchmarkResult:
    method: str
    metrics: RetrievalMetrics

    def __str__(self):
        m = self.metrics
        return (f"{self.method}:\n"
                f"  P@1: {m.precision_at_1:.4f} | P@5: {m.precision_at_5:.4f} | P@10: {m.precision_at_10:.4f}\n"
                f"  R@10: {m.recall_at_10:.4f} | MRR: {m.mrr:.4f}\n"
                f"  Avg query time: {m.query_time_ms:.2f}ms ({m.num_queries} queries)")

def load_faiss_index(index_path: str, meta_path: str = None):
    index = faiss.read_index(index_path)
    metadata = None
    if meta_path and Path(meta_path).exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    return index, metadata

def retrieve_unified(index: faiss.Index, metadata: dict, query_embeddings: np.ndarray,
                     target_agent_ids: list[int], k: int = 100) -> tuple[list[list[int]], float]:
    """
    Retrieve from unified index and filter by agent_id.
    Returns list of relevant indices per query and average query time.
    """
    query_embeddings = query_embeddings.astype('float32')

    start_time = time.time()
    distances, indices = index.search(query_embeddings, k * 10)
    query_time = (time.time() - start_time) * 1000 / len(query_embeddings)

    results = []
    for query_idx, target_agent_id in enumerate(target_agent_ids):
        relevant_indices = []
        for idx, distance in zip(indices[query_idx], distances[query_idx]):
            if idx == -1:
                break

            doc_metadata = metadata['metadatas'][idx]
            if doc_metadata.get('agent_id') == target_agent_id and distance < THRESHOLD:
                relevant_indices.append(idx)

            if len(relevant_indices) >= k:
                break

        results.append(relevant_indices)

    return results, query_time

def retrieve_individual(faiss_dir: Path, agent_id: int, query_embeddings: np.ndarray,
                       k: int = 100) -> tuple[list[list[int]], float]:
    """
    Retrieve from individual agent index.
    Returns list of relevant indices per query and average query time.
    """
    index_path = faiss_dir / f"agent_{agent_id}.bin"
    meta_path = faiss_dir / f"agent_{agent_id}_meta.json"

    if not index_path.exists():
        return [[] for _ in range(len(query_embeddings))], 0.0

    index, _ = load_faiss_index(str(index_path), str(meta_path))
    query_embeddings = query_embeddings.astype('float32')

    start_time = time.time()
    distances, indices = index.search(query_embeddings, k)
    query_time = (time.time() - start_time) * 1000 / len(query_embeddings)

    results = []
    for query_idx in range(len(query_embeddings)):
        relevant_indices = []
        for idx, distance in zip(indices[query_idx], distances[query_idx]):
            if idx == -1:
                break
            if distance < THRESHOLD:
                relevant_indices.append(idx)
        results.append(relevant_indices)

    return results, query_time

def compute_metrics(retrieved_docs: list[list[int]], ground_truth_counts: list[int]) -> RetrievalMetrics:
    """
    Compute retrieval metrics.
    retrieved_docs: list of lists of retrieved document indices per query
    ground_truth_counts: number of relevant documents per query
    """
    precisions_at_1 = []
    precisions_at_5 = []
    precisions_at_10 = []
    recalls_at_10 = []
    reciprocal_ranks = []
    query_times = []

    for retrieved, gt_count in zip(retrieved_docs, ground_truth_counts):
        num_retrieved = len(retrieved)

        precision_1 = 1.0 if num_retrieved >= 1 else 0.0
        precisions_at_1.append(precision_1)

        precision_5 = min(num_retrieved, 5) / 5.0
        precisions_at_5.append(precision_5)

        precision_10 = min(num_retrieved, 10) / 10.0
        precisions_at_10.append(precision_10)

        recall_10 = min(num_retrieved, 10) / gt_count if gt_count > 0 else 0.0
        recalls_at_10.append(recall_10)

        rr = 1.0 if num_retrieved > 0 else 0.0
        reciprocal_ranks.append(rr)

    return RetrievalMetrics(
        precision_at_1=np.mean(precisions_at_1),
        precision_at_5=np.mean(precisions_at_5),
        precision_at_10=np.mean(precisions_at_10),
        recall_at_10=np.mean(recalls_at_10),
        mrr=np.mean(reciprocal_ranks),
        query_time_ms=0.0,
        num_queries=len(retrieved_docs)
    )

def benchmark_retrieval(questions: list[Question], faiss_dir: str = "faiss",
                       k: int = 100, sample_size: int = None) -> dict[str, BenchmarkResult]:
    """
    Benchmark top-k retrieval on unified vs individual indices.
    """
    faiss_path = Path(faiss_dir)

    unified_index_path = faiss_path / "all_agents.bin"
    unified_meta_path = faiss_path / "all_agents_meta.json"

    if not unified_index_path.exists():
        raise FileNotFoundError(f"Unified index not found at {unified_index_path}")

    print(f"Loading unified index from {unified_index_path}...")
    unified_index, unified_metadata = load_faiss_index(
        str(unified_index_path), str(unified_meta_path)
    )
    print(f"Unified index loaded: {unified_index.ntotal} vectors")

    if sample_size:
        questions = questions[:sample_size]

    query_embeddings = np.array([q.embedding for q in questions])
    target_agent_ids = [q.agent_id for q in questions]

    print(f"\nBenchmarking with {len(questions)} questions...")
    print(f"Target k={k} documents per query\n")

    print("Computing ground truth counts from individual indices...")
    ground_truth_counts = []
    for question in tqdm(questions, desc="Ground truth"):
        if question.agent_id is None:
            ground_truth_counts.append(0)
            continue

        index_path = faiss_path / f"agent_{question.agent_id}.bin"
        if not index_path.exists():
            ground_truth_counts.append(0)
            continue

        index = faiss.read_index(str(index_path))
        ground_truth_counts.append(index.ntotal)

    print("\n=== UNIFIED INDEX RETRIEVAL ===")
    unified_retrieved, unified_time = retrieve_unified(
        unified_index, unified_metadata, query_embeddings, target_agent_ids, k
    )
    unified_metrics = compute_metrics(unified_retrieved, ground_truth_counts)
    unified_metrics.query_time_ms = unified_time

    print("\n=== INDIVIDUAL INDEX RETRIEVAL ===")
    individual_retrieved = []
    individual_times = []

    for question in tqdm(questions, desc="Individual retrieval"):
        if question.agent_id is None:
            individual_retrieved.append([])
            continue

        retrieved, query_time = retrieve_individual(
            faiss_path, question.agent_id, question.embedding.reshape(1, -1), k
        )
        individual_retrieved.append(retrieved[0])
        individual_times.append(query_time)

    individual_metrics = compute_metrics(individual_retrieved, ground_truth_counts)
    individual_metrics.query_time_ms = np.mean(individual_times) if individual_times else 0.0

    return {
        'unified': BenchmarkResult('Unified Index', unified_metrics),
        'individual': BenchmarkResult('Individual Indices', individual_metrics)
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS retrieval: unified vs individual indices")
    parser.add_argument("--k", type=int, default=100, help="Number of documents to retrieve per query")
    parser.add_argument("--sample", type=int, default=None, help="Sample size of questions to use (default: all)")
    parser.add_argument("--faiss-dir", type=str, default="faiss", help="Directory containing FAISS indices")

    args = parser.parse_args()

    print("Loading test questions...")
    questions = load_test_questions()

    valid_questions = [q for q in questions if q.agent_id is not None]
    print(f"Loaded {len(valid_questions)} test questions with agent IDs")

    results = benchmark_retrieval(valid_questions, args.faiss_dir, args.k, args.sample)

    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)

    for result in results.values():
        print(f"\n{result}")

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    unified_metrics = results['unified'].metrics
    individual_metrics = results['individual'].metrics

    speedup = individual_metrics.query_time_ms / unified_metrics.query_time_ms if unified_metrics.query_time_ms > 0 else float('inf')
    print(f"Query time speedup (unified vs individual): {speedup:.2f}x")

    print(f"\nPrecision@1 difference: {unified_metrics.precision_at_1 - individual_metrics.precision_at_1:+.4f}")
    print(f"Recall@10 difference: {unified_metrics.recall_at_10 - individual_metrics.recall_at_10:+.4f}")

if __name__ == "__main__":
    main()
