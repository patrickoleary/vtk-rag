"""
Unit tests for retrieval metrics (Recall@k, nDCG@k, MRR)

Tests verify mathematical correctness of IR metrics without requiring
Qdrant or real data. Fast, isolated tests for regression prevention.
"""

import sys
import unittest
from pathlib import Path

# Add evaluation module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'evaluation'))

from retrieval_metrics import RetrievalEvaluator


class TestRetrievalMetrics(unittest.TestCase):
    """Test retrieval metrics calculations"""
    
    def setUp(self):
        """Initialize evaluator for each test"""
        self.evaluator = RetrievalEvaluator()
    
    # ========================================================================
    # Recall@k Tests
    # ========================================================================
    
    def test_recall_perfect_retrieval(self):
        """Test Recall@k with perfect retrieval (all relevant in top-k)"""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = {'doc1', 'doc2'}
        
        metrics = self.evaluator.evaluate_query(retrieved, relevant)
        
        # Both relevant docs in top-3, so Recall@3 = 1.0
        self.assertEqual(metrics['recall@3'], 1.0)
        self.assertEqual(metrics['recall@5'], 1.0)
    
    def test_recall_partial_retrieval(self):
        """Test Recall@k with partial retrieval"""
        retrieved = ['doc1', 'doc_x', 'doc_y', 'doc2', 'doc_z']
        relevant = {'doc1', 'doc2', 'doc3'}  # doc3 not retrieved
        
        metrics = self.evaluator.evaluate_query(retrieved, relevant)
        
        # Found 2 out of 3 relevant docs
        self.assertAlmostEqual(metrics['recall@5'], 2/3, places=2)
    
    def test_recall_zero(self):
        """Test Recall@k when no relevant docs retrieved"""
        retrieved = ['doc_a', 'doc_b', 'doc_c']
        relevant = {'doc1', 'doc2', 'doc3'}
        
        metrics = self.evaluator.evaluate_query(retrieved, relevant)
        
        # No relevant docs found
        self.assertEqual(metrics['recall@3'], 0.0)
    
    def test_recall_at_different_k(self):
        """Test Recall increases as k increases"""
        retrieved = ['doc_x', 'doc1', 'doc_y', 'doc2', 'doc_z', 'doc3']
        relevant = {'doc1', 'doc2', 'doc3'}
        
        metrics = self.evaluator.evaluate_query(retrieved, relevant)
        
        # Recall should increase: 0 @ 1, 1/3 @ 3, 2/3 @ 5, 3/3 @ 10
        self.assertEqual(metrics['recall@1'], 0.0)
        self.assertAlmostEqual(metrics['recall@3'], 1/3, places=2)
        self.assertAlmostEqual(metrics['recall@5'], 2/3, places=2)
        self.assertEqual(metrics['recall@10'], 1.0)
    
    # ========================================================================
    # nDCG@k Tests
    # ========================================================================
    
    def test_ndcg_perfect_ranking(self):
        """Test nDCG@k with perfect ranking (all relevant at top)"""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = {'doc1', 'doc2'}
        
        metrics = self.evaluator.evaluate_query(retrieved, relevant)
        
        # Perfect ranking: nDCG = 1.0
        self.assertEqual(metrics['ndcg@3'], 1.0)
        self.assertEqual(metrics['ndcg@5'], 1.0)
    
    def test_ndcg_suboptimal_ranking(self):
        """Test nDCG@k with suboptimal ranking"""
        # Relevant at positions 3 and 5 (not ideal)
        retrieved = ['doc_a', 'doc_b', 'doc1', 'doc_c', 'doc2']
        relevant = {'doc1', 'doc2'}
        
        metrics = self.evaluator.evaluate_query(retrieved, relevant)
        
        # nDCG should be < 1.0 due to suboptimal ranking
        self.assertLess(metrics['ndcg@5'], 1.0)
        self.assertGreater(metrics['ndcg@5'], 0.0)
    
    # ========================================================================
    # MRR Tests
    # ========================================================================
    
    def test_mrr_first_position(self):
        """Test MRR when first relevant doc at position 1"""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc1'}
        
        metrics = self.evaluator.evaluate_query(retrieved, relevant)
        
        # First relevant at rank 1: MRR = 1/1 = 1.0
        self.assertEqual(metrics['mrr'], 1.0)
    
    def test_mrr_third_position(self):
        """Test MRR when first relevant doc at position 3"""
        retrieved = ['doc_a', 'doc_b', 'doc1', 'doc2']
        relevant = {'doc1', 'doc2'}
        
        metrics = self.evaluator.evaluate_query(retrieved, relevant)
        
        # First relevant at rank 3: MRR = 1/3 ≈ 0.333
        self.assertAlmostEqual(metrics['mrr'], 1/3, places=2)
    
    # ========================================================================
    # Aggregate Tests
    # ========================================================================
    
    def test_evaluate_test_set_averaging(self):
        """Test that test set evaluation averages metrics correctly"""
        results = [
            {
                'retrieved_ids': ['doc1', 'doc2', 'doc3'],
                'relevant_ids': ['doc1']
            },
            {
                'retrieved_ids': ['doc_a', 'doc_b', 'doc1'],
                'relevant_ids': ['doc1']
            }
        ]
        
        metrics = self.evaluator.evaluate_test_set(results)
        
        # Query 1: MRR = 1.0, Query 2: MRR = 0.333
        # Average MRR = (1.0 + 0.333) / 2 ≈ 0.667
        self.assertAlmostEqual(metrics.mrr, 0.667, places=2)
    
    def test_empty_relevant_set_edge_case(self):
        """Test handling of empty relevant set (edge case)"""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = set()  # No relevant docs
        
        metrics = self.evaluator.evaluate_query(retrieved, relevant)
        
        # All metrics should be 0.0 or undefined
        self.assertEqual(metrics['recall@3'], 0.0)
        self.assertEqual(metrics['mrr'], 0.0)


def run_tests():
    """Run all tests and print results"""
    print("=" * 80)
    print("Running Retrieval Metrics Tests")
    print("=" * 80)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRetrievalMetrics)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✅ All tests PASSED")
    else:
        print("❌ Some tests FAILED")
    print("=" * 80)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
