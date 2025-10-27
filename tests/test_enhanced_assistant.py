"""
Tests for EnhancedGitHubAssistant functionality.
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

from core.assistant import KimchiAssistant
from core.query_router import QueryType, RoutingDecision


class TestEnhancedGitHubAssistant:
    """Test cases for EnhancedGitHubAssistant."""
    
    @pytest.fixture
    def assistant(self):
        """Create an EnhancedGitHubAssistant instance."""
        return KimchiAssistant()
    
    def test_assistant_initialization(self, assistant):
        """Test assistant initialization."""
        assert assistant.query_router is None
        assert assistant.rag_connector is None
        assert assistant.mcp_client is None
        assert assistant.openai_client is None
        assert not assistant.is_mcp_available
        assert not assistant.is_rag_available
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, assistant):
        """Test successful initialization."""
        # Mock all dependencies
        with patch('core.assistant.QueryRouter') as mock_router_class, \
             patch('core.assistant.ElasticsearchConnector') as mock_rag_class, \
             patch('core.assistant.create_mcp_client') as mock_mcp_func, \
             patch('core.assistant.openai') as mock_openai, \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            
            # Configure mocks
            mock_router = Mock()
            mock_router_class.return_value = mock_router
            
            mock_rag = Mock()
            mock_rag_class.return_value = mock_rag
            
            mock_mcp = AsyncMock()
            mock_mcp_func.return_value = mock_mcp
            mock_mcp.connect = AsyncMock()
            
            mock_openai.OpenAI.return_value = Mock()
            
            await assistant.initialize()
            
            assert assistant.query_router == mock_router
            assert assistant.rag_connector == mock_rag
            assert assistant.mcp_client == mock_mcp
            assert assistant.is_rag_available
            assert assistant.is_mcp_available
    
    @pytest.mark.asyncio
    async def test_initialize_partial_failure(self, assistant):
        """Test initialization with partial component failure."""
        with patch('core.assistant.QueryRouter') as mock_router_class, \
             patch('core.assistant.ElasticsearchConnector', side_effect=Exception("RAG error")), \
             patch('core.assistant.create_mcp_client') as mock_mcp_func, \
             patch('core.assistant.openai') as mock_openai, \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            
            mock_router = Mock()
            mock_router_class.return_value = mock_router
            
            mock_mcp = AsyncMock()
            mock_mcp_func.return_value = mock_mcp
            mock_mcp.connect = AsyncMock()
            
            mock_openai.OpenAI.return_value = Mock()
            
            await assistant.initialize()
            
            assert assistant.query_router == mock_router
            assert assistant.rag_connector is None
            assert assistant.mcp_client == mock_mcp
            assert not assistant.is_rag_available
            assert assistant.is_mcp_available
    
    @pytest.mark.asyncio
    async def test_answer_query_mcp_only(self, assistant):
        """Test answering query using MCP only."""
        # Setup mocks
        mock_router = AsyncMock()
        mock_mcp = AsyncMock()
        mock_openai = Mock()
        
        assistant.query_router = mock_router
        assistant.mcp_client = mock_mcp
        assistant.openai_client = mock_openai
        assistant.is_mcp_available = True
        assistant.is_rag_available = False
        
        # Configure routing decision
        routing_decision = RoutingDecision(
            query_type=QueryType.LIVE_DATA,
            confidence=0.9,
            reasoning="Live data needed",
            mcp_tools=["list_commits"],
            rag_topics=[],
            priority="mcp"
        )
        mock_router.route_query.return_value = routing_decision
        
        # Configure MCP response
        mock_mcp.list_commits.return_value = "commit1\ncommit2"
        
        # Configure OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Synthesized response"
        mock_openai.chat.completions.create.return_value = mock_response
        
        result = await assistant.answer_question("What are recent commits?")
        
        assert result == "Synthesized response"
        mock_router.route_query.assert_called_once()
        mock_mcp.list_commits.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_answer_query_rag_only(self, assistant):
        """Test answering query using RAG only."""
        # Setup mocks
        mock_router = AsyncMock()
        mock_rag = Mock()
        mock_openai = Mock()
        
        assistant.query_router = mock_router
        assistant.rag_connector = mock_rag
        assistant.openai_client = mock_openai
        assistant.is_mcp_available = False
        assistant.is_rag_available = True
        
        # Configure routing decision
        routing_decision = RoutingDecision(
            query_type=QueryType.KNOWLEDGE,
            confidence=0.8,
            reasoning="Knowledge needed",
            mcp_tools=[],
            rag_topics=["ci/cd"],
            priority="rag"
        )
        mock_router.route_query.return_value = routing_decision
        
        # Configure RAG response
        mock_rag.search_documents.return_value = [
            {"content": "CI/CD best practices", "score": 0.9}
        ]
        
        # Configure OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Knowledge response"
        mock_openai.chat.completions.create.return_value = mock_response
        
        result = await assistant.answer_question("How to set up CI/CD?")
        
        assert result == "Knowledge response"
        mock_router.route_query.assert_called_once()
        mock_rag.search_documents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_answer_query_hybrid(self, assistant):
        """Test answering query using both RAG and MCP."""
        # Setup mocks
        mock_router = AsyncMock()
        mock_rag = Mock()
        mock_mcp = AsyncMock()
        mock_openai = Mock()
        
        assistant.query_router = mock_router
        assistant.rag_connector = mock_rag
        assistant.mcp_client = mock_mcp
        assistant.openai_client = mock_openai
        assistant.is_mcp_available = True
        assistant.is_rag_available = True
        
        # Configure routing decision
        routing_decision = RoutingDecision(
            query_type=QueryType.HYBRID,
            confidence=0.9,
            reasoning="Both sources needed",
            mcp_tools=["list_commits"],
            rag_topics=["best practices"],
            priority="hybrid"
        )
        mock_router.route_query.return_value = routing_decision
        
        # Configure responses
        mock_mcp.list_commits.return_value = "recent commits data"
        mock_rag.search_documents.return_value = [
            {"content": "Best practices content", "score": 0.8}
        ]
        
        # Configure OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hybrid response"
        mock_openai.chat.completions.create.return_value = mock_response
        
        result = await assistant.answer_question("Analyze recent commits for best practices")
        
        assert result == "Hybrid response"
        mock_router.route_query.assert_called_once()
        mock_mcp.list_commits.assert_called_once()
        mock_rag.search_documents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_answer_query_no_systems(self, assistant):
        """Test answering query when no systems are available."""
        # Setup minimal mock
        mock_router = AsyncMock()
        assistant.query_router = mock_router
        assistant.is_mcp_available = False
        assistant.is_rag_available = False
        
        routing_decision = RoutingDecision(
            query_type=QueryType.LIVE_DATA,
            confidence=0.8,
            reasoning="Live data needed",
            mcp_tools=["list_commits"],
            rag_topics=[],
            priority="mcp"
        )
        mock_router.route_query.return_value = routing_decision
        
        result = await assistant.answer_question("What are recent commits?")
        
        assert "not available" in result.lower() or "cannot" in result.lower()
    
    @pytest.mark.asyncio
    async def test_cleanup(self, assistant):
        """Test cleanup functionality."""
        mock_mcp = AsyncMock()
        assistant.mcp_client = mock_mcp
        
        await assistant.cleanup()
        
        mock_mcp.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, assistant):
        """Test error handling in answer_query."""
        # Setup mocks that will raise exceptions
        mock_router = AsyncMock()
        mock_router.route_query.side_effect = Exception("Routing error")
        assistant.query_router = mock_router
        
        result = await assistant.answer_question("Test query")
        
        assert "error" in result.lower() or "sorry" in result.lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
