#!/usr/bin/env python3
"""
Retrieval Metrics for VTK RAG

Implements standard information retrieval metrics:
- Recall@k: Fraction of relevant docs in top-k
- nDCG@k: Normalized Discounted Cumulative Gain
- MRR: Mean Reciprocal Rank

Evaluates retrieval quality before LLM generation.
"""

import math
from typing import List, Set, Dict, Any
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Container for retrieval metrics"""
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_3: float
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'recall@1': self.recall_at_1,
            'recall@3': self.recall_at_3,
            'recall@5': self.recall_at_5,
            'recall@10': self.recall_at_10,
            'ndcg@3': self.ndcg_at_3,
            'ndcg@5': self.ndcg_at_5,
            'ndcg@10': self.ndcg_at_10,
            'mrr': self.mrr
        }


class RetrievalEvaluator:
    """
    Evaluate retrieval quality using standard IR metrics
    
    Measures how well the retrieval pipeline finds relevant documents
    before LLM generation.
    """
    
    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Recall@k
        
        Recall@k = (# relevant docs in top-k) / (# total relevant docs)
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of ground truth relevant document IDs
            k: Cutoff position
        
        Returns:
            Recall@k score [0.0, 1.0]
        """
        if not relevant_ids:
            return 0.0
        
        top_k = set(retrieved_ids[:k])
        found = top_k.intersection(relevant_ids)
        
        return len(found) / len(relevant_ids)
    
    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate nDCG@k (Normalized Discounted Cumulative Gain)
        
        DCG@k = sum(relevance / log2(position + 1))
        nDCG@k = DCG@k / IDCG@k
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of ground truth relevant document IDs
            k: Cutoff position
        
        Returns:
            nDCG@k score [0.0, 1.0]
        """
        if not relevant_ids:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in relevant_ids:
                # Binary relevance: 1 if relevant, 0 otherwise
                relevance = 1.0
                dcg += relevance / math.log2(i + 1)
        
        # Calculate IDCG (Ideal DCG - all relevant docs at top)
        idcg = 0.0
        for i in range(1, min(len(relevant_ids), k) + 1):
            idcg += 1.0 / math.log2(i + 1)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    def mrr(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Calculate MRR (Mean Reciprocal Rank) - single query
        
        RR = 1 / rank_of_first_relevant_doc
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of ground truth relevant document IDs
        
        Returns:
            Reciprocal rank [0.0, 1.0]
        """
        if not relevant_ids:
            return 0.0
        
        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / i
        
        return 0.0  # No relevant doc found
    
    def evaluate_query(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval for a single query
        
        Args:
            retrieved_ids: Retrieved document IDs (ordered)
            relevant_ids: Ground truth relevant IDs
        
        Returns:
            Dictionary of metrics
        """
        return {
            'recall@1': self.recall_at_k(retrieved_ids, relevant_ids, 1),
            'recall@3': self.recall_at_k(retrieved_ids, relevant_ids, 3),
            'recall@5': self.recall_at_k(retrieved_ids, relevant_ids, 5),
            'recall@10': self.recall_at_k(retrieved_ids, relevant_ids, 10),
            'ndcg@3': self.ndcg_at_k(retrieved_ids, relevant_ids, 3),
            'ndcg@5': self.ndcg_at_k(retrieved_ids, relevant_ids, 5),
            'ndcg@10': self.ndcg_at_k(retrieved_ids, relevant_ids, 10),
            'mrr': self.mrr(retrieved_ids, relevant_ids)
        }
    
    def evaluate_test_set(
        self,
        results: List[Dict[str, Any]]
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval over entire test set
        
        Args:
            results: List of dicts with 'retrieved_ids' and 'relevant_ids'
        
        Returns:
            Averaged metrics across all queries
        """
        all_metrics = []
        
        for result in results:
            metrics = self.evaluate_query(
                result['retrieved_ids'],
                set(result['relevant_ids'])
            )
            all_metrics.append(metrics)
        
        # Average across all queries
        n = len(all_metrics)
        if n == 0:
            return RetrievalMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        avg_metrics = {
            key: sum(m[key] for m in all_metrics) / n
            for key in all_metrics[0].keys()
        }
        
        return RetrievalMetrics(
            recall_at_1=avg_metrics['recall@1'],
            recall_at_3=avg_metrics['recall@3'],
            recall_at_5=avg_metrics['recall@5'],
            recall_at_10=avg_metrics['recall@10'],
            ndcg_at_3=avg_metrics['ndcg@3'],
            ndcg_at_5=avg_metrics['ndcg@5'],
            ndcg_at_10=avg_metrics['ndcg@10'],
            mrr=avg_metrics['mrr']
        )
    
    def print_metrics(self, metrics: RetrievalMetrics):
        """Print metrics in readable format"""
        print("\nRetrieval Metrics:")
        print("-" * 40)
        print(f"Recall@1:  {metrics.recall_at_1:.3f}")
        print(f"Recall@3:  {metrics.recall_at_3:.3f}")
        print(f"Recall@5:  {metrics.recall_at_5:.3f}")
        print(f"Recall@10: {metrics.recall_at_10:.3f}")
        print()
        print(f"nDCG@3:    {metrics.ndcg_at_3:.3f}")
        print(f"nDCG@5:    {metrics.ndcg_at_5:.3f}")
        print(f"nDCG@10:   {metrics.ndcg_at_10:.3f}")
        print()
        print(f"MRR:       {metrics.mrr:.3f}")


def main():
    """Test retrieval metrics"""
    print("=" * 80)
    print("Retrieval Metrics - Test")
    print("=" * 80)
    
    evaluator = RetrievalEvaluator()
    
    # Test case 1: Perfect retrieval
    print("\nTest 1: Perfect Retrieval")
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = {'doc1', 'doc2'}
    
    metrics = evaluator.evaluate_query(retrieved, relevant)
    print(f"  Recall@3: {metrics['recall@3']:.3f} (expected: 1.0)")
    print(f"  nDCG@3:   {metrics['ndcg@3']:.3f} (expected: 1.0)")
    print(f"  MRR:      {metrics['mrr']:.3f} (expected: 1.0)")
    
    # Test case 2: Relevant doc at position 3
    print("\nTest 2: Relevant at Position 3")
    retrieved = ['doc_a', 'doc_b', 'doc1', 'doc_c', 'doc_d']
    relevant = {'doc1'}
    
    metrics = evaluator.evaluate_query(retrieved, relevant)
    print(f"  Recall@3: {metrics['recall@3']:.3f} (expected: 1.0)")
    print(f"  Recall@1: {metrics['recall@1']:.3f} (expected: 0.0)")
    print(f"  MRR:      {metrics['mrr']:.3f} (expected: 0.333)")
    
    # Test case 3: Partial retrieval
    print("\nTest 3: Partial Retrieval")
    retrieved = ['doc1', 'doc_x', 'doc2', 'doc_y', 'doc_z']
    relevant = {'doc1', 'doc2', 'doc3'}  # doc3 not retrieved
    
    metrics = evaluator.evaluate_query(retrieved, relevant)
    print(f"  Recall@5: {metrics['recall@5']:.3f} (expected: 0.667)")
    print(f"  nDCG@5:   {metrics['ndcg@5']:.3f}")
    
    print("\n" + "=" * 80)
    print("✓ Tests complete")


if __name__ == '__main__':
    main()
