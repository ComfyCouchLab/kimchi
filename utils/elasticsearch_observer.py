"""
Elasticsearch Query Observability Tool

This tool provides detailed insights into Elasticsearch queries, helping you understand:
- What embeddings are being generated for queries
- How vector similarity search works
- Query performance and results quality
- Document structure and content analysis
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class QueryAnalysis:
    """Detailed analysis of a search query and its results."""
    query: str
    embedding_preview: List[float]
    embedding_dimensions: int
    results_count: int
    execution_time_ms: float
    top_scores: List[float]
    result_metadata: List[Dict[str, Any]]
    timestamp: str


class ElasticsearchObserver:
    """
    Advanced observability tool for Elasticsearch queries and results.
    
    Features:
    - Query embedding analysis
    - Result similarity scoring
    - Performance monitoring
    - Query comparison
    - Result quality assessment
    """
    
    def __init__(self, elasticsearch_connector):
        self.es_connector = elasticsearch_connector
        self.query_history: List[QueryAnalysis] = []
    
    def analyze_query(self, query: str, k: int = 5) -> QueryAnalysis:
        """
        Perform comprehensive analysis of a search query.
        
        Args:
            query: The search query
            k: Number of results to retrieve
            
        Returns:
            QueryAnalysis object with detailed insights
        """
        print(f"\n🔬 ELASTICSEARCH QUERY ANALYSIS")
        print(f"{'='*60}")
        print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📝 Query: '{query}'")
        print(f"🎯 Requested results: {k}")
        
        start_time = time.time()
        
        # Generate embedding for analysis
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(model=self.es_connector.embedding_model)
        
        print(f"\n🧮 EMBEDDING ANALYSIS:")
        query_embedding = embed_model.get_text_embedding(query)
        print(f"  📊 Dimensions: {len(query_embedding)}")
        print(f"  📈 Range: [{min(query_embedding):.4f}, {max(query_embedding):.4f}]")
        print(f"  🎯 Mean: {np.mean(query_embedding):.4f}")
        print(f"  📏 L2 Norm: {np.linalg.norm(query_embedding):.4f}")
        print(f"  🔢 First 10 values: {[round(x, 4) for x in query_embedding[:10]]}")
        
        # Perform search
        results = self.es_connector.search_documents(query, k=k, debug=False)
        
        execution_time = (time.time() - start_time) * 1000
        
        print(f"\n📊 RESULTS ANALYSIS:")
        print(f"  ⏱️  Execution time: {execution_time:.2f}ms")
        print(f"  📄 Results returned: {len(results)}")
        
        if results:
            scores = [r.get('score', 0) for r in results if r.get('score') != 'N/A']
            if scores:
                print(f"  🏆 Best score: {max(scores):.4f}")
                print(f"  📉 Worst score: {min(scores):.4f}")
                print(f"  📊 Average score: {np.mean(scores):.4f}")
                print(f"  📈 Score range: {max(scores) - min(scores):.4f}")
        
        # Analyze result content
        self._analyze_results_content(results)
        
        # Create analysis record
        analysis = QueryAnalysis(
            query=query,
            embedding_preview=query_embedding[:20],  # First 20 values
            embedding_dimensions=len(query_embedding),
            results_count=len(results),
            execution_time_ms=execution_time,
            top_scores=[r.get('score', 0) for r in results[:5] if r.get('score') != 'N/A'],
            result_metadata=[r.get('metadata', {}) for r in results],
            timestamp=datetime.now().isoformat()
        )
        
        self.query_history.append(analysis)
        
        return analysis
    
    def _analyze_results_content(self, results: List[Dict[str, Any]]) -> None:
        """Analyze the content and quality of search results."""
        if not results:
            print("  ❌ No results to analyze")
            return
        
        print(f"\n📝 CONTENT ANALYSIS:")
        
        # Analyze content lengths
        content_lengths = [len(r.get('content', '')) for r in results]
        print(f"  📏 Content lengths: min={min(content_lengths)}, max={max(content_lengths)}, avg={int(np.mean(content_lengths))}")
        
        # Analyze metadata
        metadata_keys = set()
        for result in results:
            if 'metadata' in result and result['metadata']:
                metadata_keys.update(result['metadata'].keys())
        
        print(f"  🏷️  Metadata fields found: {sorted(list(metadata_keys))}")
        
        # Show sample results
        print(f"\n📄 TOP 3 RESULTS:")
        for i, result in enumerate(results[:3]):
            print(f"\n  Result {i+1}:")
            print(f"    🏆 Score: {result.get('score', 'N/A')}")
            content = result.get('content', '')
            preview = content[:150] + "..." if len(content) > 150 else content
            print(f"    📝 Content: {preview}")
            
            if result.get('metadata'):
                key_metadata = {}
                for key in ['file_path', 'file_name', 'page_label', 'file_type']:
                    if key in result['metadata']:
                        key_metadata[key] = result['metadata'][key]
                if key_metadata:
                    print(f"    📁 Key metadata: {key_metadata}")
    
    def compare_queries(self, query1: str, query2: str, k: int = 5) -> Dict[str, Any]:
        """
        Compare two queries to understand how they differ in results.
        
        Args:
            query1: First query
            query2: Second query
            k: Number of results to compare
            
        Returns:
            Comparison analysis
        """
        print(f"\n🔄 QUERY COMPARISON ANALYSIS")
        print(f"{'='*60}")
        
        # Analyze both queries
        analysis1 = self.analyze_query(query1, k)
        analysis2 = self.analyze_query(query2, k)
        
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"  Query 1: '{query1}'")
        print(f"  Query 2: '{query2}'")
        print(f"  Results overlap: {self._calculate_overlap(analysis1, analysis2):.1%}")
        print(f"  Execution time diff: {abs(analysis1.execution_time_ms - analysis2.execution_time_ms):.2f}ms")
        
        # Embedding similarity
        embedding_similarity = np.dot(analysis1.embedding_preview, analysis2.embedding_preview)
        print(f"  Embedding similarity: {embedding_similarity:.4f}")
        
        return {
            "query1": analysis1,
            "query2": analysis2,
            "overlap_percentage": self._calculate_overlap(analysis1, analysis2),
            "embedding_similarity": embedding_similarity
        }
    
    def _calculate_overlap(self, analysis1: QueryAnalysis, analysis2: QueryAnalysis) -> float:
        """Calculate the overlap between two result sets."""
        # Simple content-based overlap calculation
        content1 = set()
        content2 = set()
        
        # This is a simplified approach - in a real system you'd use node IDs
        for meta in analysis1.result_metadata:
            if 'file_path' in meta:
                content1.add(meta['file_path'])
        
        for meta in analysis2.result_metadata:
            if 'file_path' in meta:
                content2.add(meta['file_path'])
        
        if not content1 and not content2:
            return 0.0
        
        overlap = len(content1.intersection(content2))
        total_unique = len(content1.union(content2))
        
        return overlap / total_unique if total_unique > 0 else 0.0
    
    def get_query_history(self) -> List[QueryAnalysis]:
        """Get the history of analyzed queries."""
        return self.query_history
    
    def export_analysis(self, filename: str = None) -> str:
        """
        Export query analysis history to JSON file.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"elasticsearch_query_analysis_{timestamp}.json"
        
        export_data = []
        for analysis in self.query_history:
            export_data.append({
                "query": analysis.query,
                "embedding_dimensions": analysis.embedding_dimensions,
                "results_count": analysis.results_count,
                "execution_time_ms": analysis.execution_time_ms,
                "top_scores": analysis.top_scores,
                "timestamp": analysis.timestamp
            })
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Analysis exported to: {filename}")
        return filename
    
    def suggest_improvements(self, analysis: QueryAnalysis) -> List[str]:
        """
        Suggest improvements based on query analysis.
        
        Args:
            analysis: QueryAnalysis to evaluate
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        if analysis.results_count == 0:
            suggestions.append("❌ No results found - try broader or different keywords")
        elif analysis.results_count < 3:
            suggestions.append("⚠️  Few results found - consider expanding the query or checking index content")
        
        if analysis.top_scores and max(analysis.top_scores) < 0.5:
            suggestions.append("📉 Low similarity scores - results may not be very relevant")
        
        if analysis.execution_time_ms > 1000:
            suggestions.append("⏱️  Slow query execution - consider optimizing index or reducing result count")
        
        if len(analysis.query.split()) < 2:
            suggestions.append("📝 Short query - try adding more specific terms for better results")
        
        if not suggestions:
            suggestions.append("✅ Query looks good - no obvious issues detected")
        
        return suggestions


def create_observability_demo(es_connector) -> None:
    """
    Create a demo showing observability features.
    
    Args:
        es_connector: ElasticsearchConnector instance
    """
    observer = ElasticsearchObserver(es_connector)
    
    print("🎯 ELASTICSEARCH OBSERVABILITY DEMO")
    print("="*50)
    
    # Sample queries for demonstration
    test_queries = [
        "What's EIS?",
        "How to deploy applications?",
        "Best practices for testing",
        "Docker configuration",
        "Security guidelines"
    ]
    
    print(f"\n🧪 Testing {len(test_queries)} sample queries...")
    
    for query in test_queries:
        analysis = observer.analyze_query(query, k=3)
        suggestions = observer.suggest_improvements(analysis)
        
        print(f"\n💡 SUGGESTIONS FOR '{query}':")
        for suggestion in suggestions:
            print(f"  {suggestion}")
        
        print("-" * 50)
    
    # Export results
    export_file = observer.export_analysis()
    
    print(f"\n✅ Demo completed!")
    print(f"📊 Analyzed {len(observer.get_query_history())} queries")
    print(f"📁 Results saved to: {export_file}")
