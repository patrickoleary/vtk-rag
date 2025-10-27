#!/usr/bin/env python3
"""
Task-Specific Retriever for VTK RAG System

Uses content-type separation to retrieve targeted chunks:
- CODE: Pythonic, self-contained examples
- EXPLANATION: Descriptions and tutorials
- API_DOC: Class and method documentation
- IMAGE: Visual results (optional)

This retriever dramatically reduces token usage by filtering chunks
by content type and metadata before retrieval.

Quality Boosting:
- Pythonic API examples: +20% score boost
- Modular imports: +15% score boost
- Combined (pythonic + modular): +35% total boost
This ensures the 971 gold-standard examples rank higher.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks for retrieval"""
    CODE_GENERATION = "code_generation"
    EXPLANATION = "explanation"
    API_LOOKUP = "api_lookup"
    MIXED = "mixed"


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    chunk_id: str
    content: str
    content_type: str
    source_type: str
    score: float
    metadata: Dict[str, Any]
    
    def estimate_tokens(self) -> int:
        """Estimate token count (4 chars per token)"""
        return len(self.content) // 4


@dataclass
class RetrievalConfig:
    """Configuration for retrieval strategy"""
    task_type: TaskType
    prefer_pythonic: bool = True
    prefer_self_contained: bool = True
    require_visualization: Optional[bool] = None
    complexity_level: Optional[str] = None
    vtk_classes: Optional[List[str]] = None
    category: Optional[str] = None
    class_name: Optional[str] = None


class TaskSpecificRetriever:
    """
    Retrieves chunks based on task type with content-type filtering
    
    Key features:
    - Content-type separation (code/explanation/api_doc/image)
    - Metadata filtering (pythonic, self-contained, complexity)
    - Quality-based score boosting (pythonic + modular preferred)
    - Token-efficient retrieval (85-95% reduction)
    - Configurable retrieval strategies
    
    Score Boosting:
    - Automatically boosts pythonic API examples (+20%)
    - Automatically boosts modular imports (+15%)
    - Combined boost: +35% for ideal examples
    """
    
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vtk_docs",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize task-specific retriever
        
        Args:
            qdrant_url: Qdrant server URL
            collection_name: Collection name
            embedding_model: Sentence transformer model
        """
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model)
    
    def retrieve_code(
        self,
        query: str,
        top_k: int = 3,
        prefer_pythonic: bool = True,
        prefer_self_contained: bool = True,
        require_visualization: Optional[bool] = None,
        complexity_level: Optional[str] = None,
        vtk_classes: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve CODE chunks
        
        Args:
            query: Search query
            top_k: Number of results
            prefer_pythonic: Prefer pythonic import style
            prefer_self_contained: Prefer examples without data files
            require_visualization: Filter by visualization presence
            complexity_level: Filter by complexity (simple/moderate/complex)
            vtk_classes: Filter by VTK classes used
        
        Returns:
            List of CODE chunks
        """
        query_vector = self.embedding_model.encode(query).tolist()
        
        # Build filter
        must_conditions = [
            {"key": "content_type", "match": {"value": "code"}}
        ]
        
        if prefer_self_contained:
            must_conditions.append(
                {"key": "metadata.requires_data_files", "match": {"value": False}}
            )
        
        if require_visualization is not None:
            must_conditions.append(
                {"key": "metadata.has_visualization", "match": {"value": require_visualization}}
            )
        
        if complexity_level:
            must_conditions.append(
                {"key": "metadata.complexity", "match": {"value": complexity_level}}
            )
        
        if vtk_classes:
            must_conditions.append(
                {"key": "metadata.vtk_classes", "match": {"any": vtk_classes}}
            )
        
        should_conditions = []
        if prefer_pythonic:
            should_conditions.append(
                {"key": "metadata.source_style", "match": {"value": "pythonic"}}
            )
        
        query_filter = {"must": must_conditions}
        if should_conditions:
            query_filter["should"] = should_conditions
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
        )
        
        return self._format_results(results)
    
    def retrieve_explanation(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve EXPLANATION chunks
        
        Args:
            query: Search query
            top_k: Number of results
            category: Filter by category (e.g., "GeometricObjects")
        
        Returns:
            List of EXPLANATION chunks
        """
        query_vector = self.embedding_model.encode(query).tolist()
        
        # Build filter
        must_conditions = [
            {"key": "content_type", "match": {"value": "explanation"}}
        ]
        
        if category:
            must_conditions.append(
                {"key": "metadata.category", "match": {"value": category}}
            )
        
        query_filter = {"must": must_conditions}
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
        )
        
        return self._format_results(results)
    
    def retrieve_api_doc(
        self,
        query: str,
        top_k: int = 3,
        class_name: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve API_DOC chunks
        
        Args:
            query: Search query
            top_k: Number of results
            class_name: Filter by VTK class (e.g., "vtkActor")
        
        Returns:
            List of API_DOC chunks
        """
        query_vector = self.embedding_model.encode(query).tolist()
        
        # Build filter
        must_conditions = [
            {"key": "content_type", "match": {"value": "api_doc"}}
        ]
        
        if class_name:
            must_conditions.append(
                {"key": "metadata.class_name", "match": {"value": class_name}}
            )
        
        query_filter = {"must": must_conditions}
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
        )
        
        return self._format_results(results)
    
    def retrieve_image(
        self,
        query: str,
        top_k: int = 3,
        image_type: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve IMAGE chunks (metadata only)
        
        Args:
            query: Search query
            top_k: Number of results
            image_type: Filter by type ("result" or "baseline")
        
        Returns:
            List of IMAGE chunks
        """
        query_vector = self.embedding_model.encode(query).tolist()
        
        # Build filter
        must_conditions = [
            {"key": "content_type", "match": {"value": "image"}}
        ]
        
        if image_type:
            must_conditions.append(
                {"key": "metadata.image_type", "match": {"value": image_type}}
            )
        
        query_filter = {"must": must_conditions}
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
        )
        
        return self._format_results(results)
    
    def retrieve_mixed(
        self,
        query: str,
        code_k: int = 2,
        explanation_k: int = 2,
        api_k: int = 1,
        **kwargs
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve mixed content types
        
        Args:
            query: Search query
            code_k: Number of CODE chunks
            explanation_k: Number of EXPLANATION chunks
            api_k: Number of API_DOC chunks
            **kwargs: Additional filters passed to retrieval methods
        
        Returns:
            Dict with keys: 'code', 'explanation', 'api_doc'
        """
        return {
            'code': self.retrieve_code(query, top_k=code_k, **kwargs),
            'explanation': self.retrieve_explanation(query, top_k=explanation_k),
            'api_doc': self.retrieve_api_doc(query, top_k=api_k)
        }
    
    def retrieve_with_config(
        self,
        query: str,
        config: RetrievalConfig,
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """
        Retrieve using a configuration object
        
        Args:
            query: Search query
            config: Retrieval configuration
            top_k: Number of results
        
        Returns:
            List of results
        """
        if config.task_type == TaskType.CODE_GENERATION:
            return self.retrieve_code(
                query,
                top_k=top_k,
                prefer_pythonic=config.prefer_pythonic,
                prefer_self_contained=config.prefer_self_contained,
                require_visualization=config.require_visualization,
                complexity_level=config.complexity_level,
                vtk_classes=config.vtk_classes
            )
        
        elif config.task_type == TaskType.EXPLANATION:
            return self.retrieve_explanation(
                query,
                top_k=top_k,
                category=config.category
            )
        
        elif config.task_type == TaskType.API_LOOKUP:
            return self.retrieve_api_doc(
                query,
                top_k=top_k,
                class_name=config.class_name
            )
        
        elif config.task_type == TaskType.MIXED:
            results = self.retrieve_mixed(query)
            # Flatten and return top_k
            all_results = []
            for result_list in results.values():
                all_results.extend(result_list)
            # Sort by score and return top_k
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:top_k]
        
        else:
            raise ValueError(f"Unknown task type: {config.task_type}")
    
    def _boost_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Apply metadata-based score boosting to prefer high-quality examples
        
        Boosts:
        - Pythonic API examples: +20%
        - Modular imports: +15%
        - Combined (pythonic + modular): +35% total
        
        This ensures the best examples rank higher in results.
        """
        boosted_results = []
        for result in results:
            boosted_score = result.score
            
            # Boost pythonic API examples (modern, idiomatic)
            if result.metadata.get('source_style') == 'pythonic':
                boosted_score *= 1.20
            
            # Boost modular imports (selective, best practice)
            if result.metadata.get('import_style') == 'modular':
                boosted_score *= 1.15
            
            # Create new result with boosted score
            boosted_results.append(RetrievalResult(
                chunk_id=result.chunk_id,
                content=result.content,
                content_type=result.content_type,
                source_type=result.source_type,
                score=boosted_score,
                metadata=result.metadata
            ))
        
        # Re-sort by boosted scores
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        return boosted_results
    
    def _format_results(self, qdrant_results) -> List[RetrievalResult]:
        """Convert Qdrant results to RetrievalResult objects"""
        results = []
        for result in qdrant_results:
            results.append(RetrievalResult(
                chunk_id=result.payload['chunk_id'],
                content=result.payload['content'],
                content_type=result.payload['content_type'],
                source_type=result.payload['source_type'],
                score=result.score,
                metadata=result.payload.get('metadata', {})
            ))
        
        # Apply score boosting based on quality indicators
        results = self._boost_scores(results)
        
        return results
    
    def retrieve_by_ids(self, chunk_ids: List[str]) -> List[RetrievalResult]:
        """
        Retrieve chunks by their IDs directly (no vector search)
        
        Args:
            chunk_ids: List of chunk IDs to fetch
        
        Returns:
            List of retrieval results in the order of chunk_ids
        """
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        
        # Fetch all requested chunks using scroll with filter
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="chunk_id",
                        match=MatchAny(any=chunk_ids)
                    )
                ]
            ),
            limit=len(chunk_ids),
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        results_by_id = {}
        for point in scroll_result[0]:  # scroll_result is (points, next_offset)
            chunk_id = point.payload['chunk_id']
            results_by_id[chunk_id] = RetrievalResult(
                chunk_id=chunk_id,
                content=point.payload['content'],
                content_type=point.payload['content_type'],
                source_type=point.payload['source_type'],
                score=1.0,  # No score for ID-based retrieval
                metadata=point.payload.get('metadata', {})
            )
        
        # Return in requested order
        return [results_by_id[cid] for cid in chunk_ids if cid in results_by_id]
    
    def estimate_total_tokens(self, results: List[RetrievalResult]) -> int:
        """Calculate total tokens for a list of results"""
        return sum(r.estimate_tokens() for r in results)
    
    def print_results_summary(self, results: List[RetrievalResult], title: str = "Results"):
        """Print a summary of retrieval results"""
        print(f"\n{'='*80}")
        print(f"{title}")
        print(f"{'='*80}")
        print(f"Total results: {len(results)}")
        print(f"Estimated tokens: {self.estimate_total_tokens(results)}")
        
        for i, result in enumerate(results, 1):
            # Check if this result was boosted
            is_pythonic = result.metadata.get('source_style') == 'pythonic'
            is_modular = result.metadata.get('import_style') == 'modular'
            boost_marker = ''
            if is_pythonic and is_modular:
                boost_marker = ' ✓✓ (pythonic+modular, +35% boost)'
            elif is_pythonic:
                boost_marker = ' ✓ (pythonic, +20% boost)'
            elif is_modular:
                boost_marker = ' ✓ (modular, +15% boost)'
            
            print(f"\n{i}. {result.chunk_id}{boost_marker}")
            print(f"   Score: {result.score:.4f}")
            print(f"   Type: {result.content_type} ({result.source_type})")
            print(f"   Tokens: ~{result.estimate_tokens()}")
            
            if result.content_type == 'code':
                print(f"   API Style: {result.metadata.get('source_style', 'N/A')}")
                print(f"   Import Style: {result.metadata.get('import_style', 'N/A')}")
                print(f"   Complexity: {result.metadata.get('complexity', 'N/A')}")
                print(f"   Requires Data: {result.metadata.get('requires_data_files', False)}")
            
            preview = result.content[:150].replace('\n', ' ')
            print(f"   Preview: {preview}...")


if __name__ == '__main__':
    # Quick test
    retriever = TaskSpecificRetriever()
    
    # Test code retrieval
    print("\nTesting CODE retrieval...")
    code_results = retriever.retrieve_code(
        "How to create a cylinder?",
        top_k=3,
        prefer_pythonic=True,
        prefer_self_contained=True
    )
    retriever.print_results_summary(code_results, "CODE Chunks")
    
    # Test explanation retrieval
    print("\nTesting EXPLANATION retrieval...")
    explanation_results = retriever.retrieve_explanation(
        "cylinder geometry",
        top_k=2
    )
    retriever.print_results_summary(explanation_results, "EXPLANATION Chunks")
    
    # Token comparison
    total_tokens = retriever.estimate_total_tokens(code_results + explanation_results)
    print(f"\n💡 Total tokens: {total_tokens}")
    print(f"   Old system: ~10,500 tokens (7 mixed chunks)")
    print(f"   Token reduction: ~{100 - (total_tokens / 10500 * 100):.0f}%")
