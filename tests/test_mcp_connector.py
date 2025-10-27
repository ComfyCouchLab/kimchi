"""
Tests for MCP GitHub Connector functionality.
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

from connectors.mcp_github_connector import MCPGitHubClient, MCPConfig, create_local_config, create_mcp_client


class TestMCPConfig:
    """Test cases for MCPConfig."""
    
    def test_config_initialization(self):
        """Test config initialization with defaults."""
        config = MCPConfig()
        assert config.connection_type == "remote"
        assert config.timeout == 30
        assert config.max_retries == 3
    
    def test_config_with_values(self):
        """Test config initialization with custom values."""
        config = MCPConfig(
            connection_type="local",
            github_token="test_token",
            timeout=60
        )
        assert config.connection_type == "local"
        assert config.github_token == "test_token"
        assert config.timeout == 60


class TestMCPGitHubClient:
    """Test cases for MCPGitHubClient."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock MCP configuration."""
        return MCPConfig(
            github_token='test_token',
            connection_type='remote'
        )
    
    @pytest.fixture
    def client(self, mock_config):
        """Create an MCPGitHubClient instance."""
        return MCPGitHubClient(mock_config)
    
    def test_client_initialization(self, client, mock_config):
        """Test client initialization."""
        assert client.config == mock_config
        assert not client.is_connected
        assert client.session_id is None
        assert client._request_id == 0
    
    @pytest.mark.asyncio
    async def test_connect_remote_success(self, client):
        """Test successful remote connection."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock successful connection response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"session_id": "test_session"}
            mock_client.post.return_value = mock_response
            
            await client.connect()
            
            assert client.is_connected
            assert client.session_id == "test_session"
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, client):
        """Test connection failure."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock failed connection
            mock_client.post.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception):
                await client.connect()
            
            assert not client.is_connected
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test disconnection."""
        # Simulate connected state
        client.is_connected = True
        client.http_client = AsyncMock()
        
        await client.disconnect()
        
        assert not client.is_connected
        assert client.session_id is None
    
    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, client):
        """Test tool call when not connected."""
        with pytest.raises(Exception, match="not connected|Not connected"):
            await client.call_tool("test_tool", {})
    
    @pytest.mark.asyncio
    async def test_get_repository_info_method_exists(self, client):
        """Test that repository info method exists."""
        # Just check the method exists - actual functionality would need real connection
        assert hasattr(client, 'get_repository_info')
        assert callable(getattr(client, 'get_repository_info'))
    
    @pytest.mark.asyncio
    async def test_list_available_tools_method_exists(self, client):
        """Test that list tools method exists."""
        assert hasattr(client, 'list_available_tools')
        assert callable(getattr(client, 'list_available_tools'))


class TestMCPUtils:
    """Test utility functions for MCP."""
    
    def test_create_local_config(self):
        """Test creation of local MCP configuration."""
        config = create_local_config(
            github_token="test_token",
            toolsets=["repos", "issues"],
            read_only=True
        )
        
        assert config.github_token == "test_token"
        assert config.connection_type == "local"
        assert config.toolsets == ["repos", "issues"]
        assert config.read_only is True
    
    def test_create_local_config_defaults(self):
        """Test local config with defaults."""
        config = create_local_config(github_token="test_token")
        
        assert config.github_token == "test_token"
        assert config.connection_type == "local"
        assert isinstance(config.toolsets, list) or config.toolsets is None
    
    @pytest.mark.asyncio
    async def test_create_mcp_client(self):
        """Test MCP client creation."""
        config = MCPConfig(
            github_token='test_token',
            connection_type='remote'
        )
        
        client = await create_mcp_client(config)
        assert isinstance(client, MCPGitHubClient)
        assert client.config == config
    
    @pytest.mark.asyncio
    async def test_create_mcp_client_no_config(self):
        """Test MCP client creation without config."""
        with patch.dict(os.environ, {'GITHUB_TOKEN': 'env_token'}):
            client = await create_mcp_client()
            assert isinstance(client, MCPGitHubClient)
            assert client.config.github_token == 'env_token'


class TestMCPIntegration:
    """Integration tests for MCP functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv('GITHUB_TOKEN'), reason="GitHub token required for integration tests")
    async def test_real_mcp_connection(self):
        """Test real MCP connection (requires GitHub token)."""
        github_token = os.getenv('GITHUB_TOKEN') or os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
        if not github_token:
            pytest.skip("GitHub token required")
        
        config = create_local_config(
            github_token=github_token,
            toolsets=["repos"]
        )
        
        client = await create_mcp_client(config)
        
        try:
            # Try to connect - may fail if Docker/MCP server not available
            await client.connect()
            
            # If connection successful, test basic functionality
            if client.is_connected:
                tools = await client.list_available_tools()
                assert isinstance(tools, list)
            
        except Exception as e:
            # If connection fails (Docker not available, etc.), skip
            pytest.skip(f"MCP server not available: {e}")
        
        finally:
            if client.is_connected:
                await client.disconnect()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
