"""
Tests for QueryRouter functionality.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from query_router import QueryRouter, QueryType, RoutingDecision


class TestQueryRouter:
    """Test cases for QueryRouter."""
    
    @pytest.fixture
    def router(self):
        """Create a QueryRouter instance with mock API key."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            return QueryRouter()
    
    def test_router_initialization(self, router):
        """Test router initialization."""
        assert router.openai_api_key == 'test_key'
        assert hasattr(router, 'routing_examples')
        assert hasattr(router, 'mcp_tool_mapping')
    
    def test_router_initialization_no_api_key(self):
        """Test router initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key required"):
                QueryRouter()
    
    def test_get_routing_examples(self, router):
        """Test routing examples structure."""
        examples = router._get_routing_examples()
        
        assert 'live_data' in examples
        assert 'knowledge' in examples
        assert 'hybrid' in examples
        assert 'general' in examples
        
        # Check that each category has examples
        for category, example_list in examples.items():
            assert isinstance(example_list, list)
            assert len(example_list) > 0
            assert all(isinstance(ex, str) for ex in example_list)
    
    def test_get_mcp_tool_mapping(self, router):
        """Test MCP tool mapping structure."""
        mapping = router._get_mcp_tool_mapping()
        
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        
        # Check that all tools are strings
        for tool in mapping.values():
            assert isinstance(tool, str)
    
    @pytest.mark.asyncio
    async def test_route_query_success(self, router):
        """Test successful query routing."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '''
        {
            "query_type": "live_data",
            "confidence": 0.9,
            "reasoning": "User wants current commit data",
            "mcp_tools": ["list_commits"],
            "rag_topics": [],
            "priority": "mcp"
        }
        '''
        
        with patch('openai.ChatCompletion.acreate', return_value=mock_response):
            decision = await router.route_query("What are the recent commits?")
            
            assert isinstance(decision, RoutingDecision)
            assert decision.query_type == QueryType.LIVE_DATA
            assert decision.confidence == 0.9
            assert "list_commits" in decision.mcp_tools
    
    @pytest.mark.asyncio
    async def test_route_query_invalid_json(self, router):
        """Test routing with invalid JSON response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Invalid JSON"
        
        with patch('openai.ChatCompletion.acreate', return_value=mock_response):
            decision = await router.route_query("Test query")
            
            # Should fall back to general type
            assert decision.query_type == QueryType.GENERAL
            assert decision.confidence < 0.5
    
    @pytest.mark.asyncio
    async def test_route_query_api_error(self, router):
        """Test routing with API error."""
        with patch('openai.ChatCompletion.acreate', side_effect=Exception("API Error")):
            decision = await router.route_query("Test query")
            
            # Should fall back to general type
            assert decision.query_type == QueryType.GENERAL
            assert decision.confidence < 0.5
            assert "error" in decision.reasoning.lower()
    
    def test_query_type_enum(self):
        """Test QueryType enum values."""
        assert QueryType.LIVE_DATA.value == "live_data"
        assert QueryType.KNOWLEDGE.value == "knowledge"
        assert QueryType.HYBRID.value == "hybrid"
        assert QueryType.GENERAL.value == "general"
    
    def test_routing_decision_dataclass(self):
        """Test RoutingDecision dataclass."""
        decision = RoutingDecision(
            query_type=QueryType.LIVE_DATA,
            confidence=0.8,
            reasoning="Test reasoning",
            mcp_tools=["test_tool"],
            rag_topics=["test_topic"],
            priority="mcp"
        )
        
        assert decision.query_type == QueryType.LIVE_DATA
        assert decision.confidence == 0.8
        assert decision.reasoning == "Test reasoning"
        assert decision.mcp_tools == ["test_tool"]
        assert decision.rag_topics == ["test_topic"]
        assert decision.priority == "mcp"
    
    def test_classify_query_patterns(self, router):
        """Test classification of different query patterns."""
        examples = router._get_routing_examples()
        
        # Test each category has distinct patterns
        live_data_examples = examples['live_data']
        knowledge_examples = examples['knowledge']
        hybrid_examples = examples['hybrid']
        
        assert any('commit' in ex.lower() for ex in live_data_examples)
        assert any('issue' in ex.lower() for ex in live_data_examples)
        assert any('best practice' in ex.lower() for ex in knowledge_examples)
        assert any('ci/cd' in ex.lower() for ex in knowledge_examples)
        assert any('analyze' in ex.lower() for ex in hybrid_examples)
        assert any('recommend' in ex.lower() for ex in hybrid_examples)


class TestQueryRouterIntegration:
    """Integration tests for QueryRouter."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key required for integration tests")
    async def test_real_query_routing(self):
        """Test real query routing with OpenAI API."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            pytest.skip("OpenAI API key required")
        
        router = QueryRouter(api_key)
        
        # Test live data query
        decision = await router.route_query("What are the recent commits?")
        assert isinstance(decision, RoutingDecision)
        assert decision.query_type in [QueryType.LIVE_DATA, QueryType.HYBRID]
        
        # Test knowledge query
        decision = await router.route_query("How do I set up CI/CD?")
        assert isinstance(decision, RoutingDecision)
        assert decision.query_type in [QueryType.KNOWLEDGE, QueryType.HYBRID]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
