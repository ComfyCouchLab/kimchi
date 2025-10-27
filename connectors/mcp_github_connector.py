"""
MCP GitHub Connector - Handles communication with GitHub's official MCP server.

This module provides functionality to connect to and interact with GitHub's 
official MCP server for real-time repository data access.

Supports both remote (hosted) and local (Docker/binary) GitHub MCP server modes.
"""

import os
import asyncio
import json
import subprocess
import signal
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from dotenv import load_dotenv
import httpx
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MCPConfig:
    """Configuration for GitHub MCP server connection."""
    
    # Connection type: "remote" or "local"
    connection_type: str = "remote"
    
    # Remote server configuration
    remote_url: str = "https://api.githubcopilot.com/mcp/"
    
    # Local server configuration (Docker or binary)
    local_command: str = "docker"
    local_args: List[str] = None
    
    # Authentication
    github_token: Optional[str] = None
    
    # Server settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 2
    
    # GitHub MCP specific settings
    toolsets: List[str] = None  # e.g., ["repos", "issues", "pull_requests"]
    read_only: bool = False
    dynamic_toolsets: bool = False
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.local_args is None:
            self.local_args = [
                "run", "-i", "--rm", 
                "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github-mcp-server"
            ]
        
        if self.toolsets is None:
            self.toolsets = ["default"]  # GitHub MCP server default toolset


class MCPGitHubConnectorError(Exception):
    """Custom exception for MCP GitHub connector operations."""
    pass


class MCPGitHubClient:
    """
    Handles communication with GitHub's official MCP server for real-time data access.
    
    This connector supports both remote and local GitHub MCP server modes:
    - Remote: GitHub-hosted server at api.githubcopilot.com
    - Local: Docker container or binary with stdio protocol
    
    The connector is responsible for:
    - Connecting to GitHub MCP server (remote HTTP or local stdio)
    - Making real-time requests for repository data
    - Handling authentication and error management
    - Providing methods for common GitHub operations
    """
    
    def __init__(self, config: Optional[MCPConfig] = None):
        """
        Initialize the MCP GitHub client.
        
        Args:
            config: MCPConfig object. If None, will load from environment variables.
        """
        load_dotenv('.env')
        
        if config:
            self.config = config
        else:
            self.config = self._load_config_from_env()
        
        # Connection state
        self.is_connected = False
        self.session_id = None
        
        # For HTTP remote connections
        self.http_client = None
        
        # For local stdio connections
        self.process = None
        self.stdin = None
        self.stdout = None
        
        # Request ID counter for tracking requests
        self._request_id = 0
    
    def _load_config_from_env(self) -> MCPConfig:
        """Load configuration from environment variables."""
        connection_type = os.getenv('MCP_CONNECTION_TYPE', 'remote')
        remote_url = os.getenv('MCP_REMOTE_URL', 'https://api.githubcopilot.com/mcp/')
        github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN') or os.getenv('GITHUB_TOKEN')
        
        # Local server configuration
        local_command = os.getenv('MCP_LOCAL_COMMAND', 'docker')
        
        # Parse toolsets from environment
        toolsets_env = os.getenv('GITHUB_TOOLSETS', 'default')
        toolsets = [t.strip() for t in toolsets_env.split(',') if t.strip()]
        
        # Other settings
        timeout = int(os.getenv('MCP_TIMEOUT', '30'))
        max_retries = int(os.getenv('MCP_MAX_RETRIES', '3'))
        retry_delay = int(os.getenv('MCP_RETRY_DELAY', '2'))
        read_only = os.getenv('GITHUB_READ_ONLY', '').lower() in ('1', 'true', 'yes')
        dynamic_toolsets = os.getenv('GITHUB_DYNAMIC_TOOLSETS', '').lower() in ('1', 'true', 'yes')
        
        return MCPConfig(
            connection_type=connection_type,
            remote_url=remote_url,
            local_command=local_command,
            github_token=github_token,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            toolsets=toolsets,
            read_only=read_only,
            dynamic_toolsets=dynamic_toolsets
        )
    
    def _get_next_request_id(self) -> str:
        """Generate next request ID."""
        self._request_id += 1
        return f"req_{self._request_id}"
    
    async def connect(self) -> bool:
        """
        Connect to the GitHub MCP server.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            if self.config.connection_type == "remote":
                return await self._connect_remote()
            else:
                return await self._connect_local()
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self.is_connected = False
            return False
    
    async def _connect_remote(self) -> bool:
        """Connect to remote GitHub MCP server."""
        logger.info(f"Connecting to remote MCP server at {self.config.remote_url}")
        
        if not self.config.github_token:
            raise MCPGitHubConnectorError("GitHub token is required for remote MCP server")
        
        headers = {
            "Authorization": f"Bearer {self.config.github_token}",
            "Content-Type": "application/json"
        }
        
        self.http_client = httpx.AsyncClient(
            base_url=self.config.remote_url,
            headers=headers,
            timeout=self.config.timeout
        )
        
        # Test connection by listing tools
        try:
            response = await self.http_client.post("/tools/list", json={
                "jsonrpc": "2.0",
                "id": self._get_next_request_id(),
                "method": "tools/list",
                "params": {}
            })
            
            if response.status_code == 200:
                self.is_connected = True
                logger.info("Successfully connected to remote MCP server")
                return True
            else:
                logger.error(f"Remote connection failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Remote connection test failed: {e}")
            return False
    
    async def _connect_local(self) -> bool:
        """Connect to local GitHub MCP server via stdio."""
        logger.info("Starting local MCP server process")
        
        if not self.config.github_token:
            raise MCPGitHubConnectorError("GitHub token is required for local MCP server")
        
        # Prepare environment variables
        env = os.environ.copy()
        env["GITHUB_PERSONAL_ACCESS_TOKEN"] = self.config.github_token
        
        if self.config.toolsets and self.config.toolsets != ["default"]:
            env["GITHUB_TOOLSETS"] = ",".join(self.config.toolsets)
        
        if self.config.read_only:
            env["GITHUB_READ_ONLY"] = "1"
        
        if self.config.dynamic_toolsets:
            env["GITHUB_DYNAMIC_TOOLSETS"] = "1"
        
        # Start the MCP server process
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.config.local_command,
                *self.config.local_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            self.stdin = self.process.stdin
            self.stdout = self.process.stdout
            
            # Send initialization message
            init_message = {
                "jsonrpc": "2.0",
                "id": self._get_next_request_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {
                        "name": "kimchi-rag",
                        "version": "1.0.0"
                    },
                    "capabilities": {}
                }
            }
            
            await self._send_local_message(init_message)
            response = await self._receive_local_message()
            
            if "error" in response:
                logger.error(f"Local server initialization failed: {response['error']}")
                return False
            
            # Send initialized notification
            initialized_message = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            
            await self._send_local_message(initialized_message)
            
            self.is_connected = True
            logger.info("Successfully connected to local MCP server")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start local MCP server: {e}")
            return False
    
    async def _send_local_message(self, message: Dict[str, Any]):
        """Send a message to local MCP server via stdin."""
        if not self.stdin:
            raise MCPGitHubConnectorError("Local server not connected")
        
        message_str = json.dumps(message) + "\n"
        self.stdin.write(message_str.encode())
        await self.stdin.drain()
    
    async def _receive_local_message(self) -> Dict[str, Any]:
        """Receive a message from local MCP server via stdout."""
        if not self.stdout:
            raise MCPGitHubConnectorError("Local server not connected")
        
        line = await self.stdout.readline()
        if not line:
            raise MCPGitHubConnectorError("Local server connection closed")
        
        return json.loads(line.decode().strip())
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.config.connection_type == "remote" and self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        elif self.config.connection_type == "local" and self.process:
            if self.stdin:
                self.stdin.close()
                await self.stdin.wait_closed()
            
            # Terminate the process gracefully
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            
            self.process = None
            self.stdin = None
            self.stdout = None
        
        self.is_connected = False
        logger.info("Disconnected from MCP server")
    
    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to the MCP server.
        
        Args:
            method: The MCP method to call
            params: Parameters for the method
            
        Returns:
            Response from the server
            
        Raises:
            MCPGitHubConnectorError: If request fails
        """
        if not self.is_connected:
            raise MCPGitHubConnectorError("Not connected to MCP server")
        
        if self.config.connection_type == "remote":
            return await self._send_remote_request(method, params)
        else:
            return await self._send_local_request(method, params)
    
    async def _send_remote_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to remote MCP server."""
        request_id = self._get_next_request_id()
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }
        
        try:
            # For remote server, we post to the appropriate endpoint
            endpoint = f"/{method.replace('/', '_')}"
            response = await self.http_client.post(endpoint, json=message)
            
            if response.status_code != 200:
                raise MCPGitHubConnectorError(f"HTTP {response.status_code}: {response.text}")
            
            response_data = response.json()
            
            if "error" in response_data:
                raise MCPGitHubConnectorError(f"MCP request failed: {response_data['error']}")
            
            return response_data.get("result", {})
            
        except httpx.TimeoutException:
            raise MCPGitHubConnectorError(f"Request timeout after {self.config.timeout} seconds")
        except Exception as e:
            raise MCPGitHubConnectorError(f"Remote request failed: {e}")
    
    async def _send_local_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to local MCP server."""
        request_id = self._get_next_request_id()
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }
        
        try:
            await self._send_local_message(message)
            response = await asyncio.wait_for(
                self._receive_local_message(),
                timeout=self.config.timeout
            )
            
            if "error" in response:
                raise MCPGitHubConnectorError(f"MCP request failed: {response['error']}")
            
            return response.get("result", {})
            
        except asyncio.TimeoutError:
            raise MCPGitHubConnectorError(f"Request timeout after {self.config.timeout} seconds")
        except Exception as e:
            raise MCPGitHubConnectorError(f"Local request failed: {e}")
    
    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools from the MCP server.
        
        Returns:
            List of available tools and their descriptions
        """
        try:
            result = await self._send_request("tools/list", {})
            return result.get("tools", [])
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []
    
    async def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Get repository information via MCP.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository information
        """
        params = {
            "name": "get_repository",
            "arguments": {
                "owner": owner,
                "repo": repo
            }
        }
        
        return await self._send_request("tools/call", params)
    
    async def get_file_content(self, owner: str, repo: str, file_path: str, ref: str = "main") -> Dict[str, Any]:
        """
        Get file content from repository via MCP.
        
        Args:
            owner: Repository owner
            repo: Repository name
            file_path: Path to the file
            ref: Branch/tag/commit reference (default: main)
            
        Returns:
            File content and metadata
        """
        params = {
            "name": "get_file_contents",
            "arguments": {
                "owner": owner,
                "repo": repo,
                "path": file_path,
                "ref": ref
            }
        }
        
        return await self._send_request("tools/call", params)
    
    async def list_repository_files(self, owner: str, repo: str, path: str = "", ref: str = "main") -> Dict[str, Any]:
        """
        List files in repository directory via MCP.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: Directory path (empty for root)
            ref: Branch/tag/commit reference (default: main)
            
        Returns:
            List of files and directories
        """
        params = {
            "name": "get_file_contents",
            "arguments": {
                "owner": owner,
                "repo": repo,
                "path": path,
                "ref": ref
            }
        }
        
        return await self._send_request("tools/call", params)
    
    async def get_recent_commits(self, owner: str, repo: str, ref: str = "main", per_page: int = 10) -> Dict[str, Any]:
        """
        Get recent commits from repository via MCP.
        
        Args:
            owner: Repository owner
            repo: Repository name
            ref: Branch/tag/commit reference (default: main)
            per_page: Number of commits to retrieve
            
        Returns:
            List of recent commits
        """
        params = {
            "name": "list_commits",
            "arguments": {
                "owner": owner,
                "repo": repo,
                "sha": ref,
                "per_page": per_page
            }
        }
        
        return await self._send_request("tools/call", params)
    
    async def get_repository_issues(self, owner: str, repo: str, state: str = "open", per_page: int = 10) -> Dict[str, Any]:
        """
        Get repository issues via MCP.
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: Issue state (open, closed, all)
            per_page: Number of issues to retrieve
            
        Returns:
            List of issues
        """
        params = {
            "name": "list_issues",
            "arguments": {
                "owner": owner,
                "repo": repo,
                "state": state,
                "per_page": per_page
            }
        }
        
        return await self._send_request("tools/call", params)
    
    async def get_pull_requests(self, owner: str, repo: str, state: str = "open", per_page: int = 10) -> Dict[str, Any]:
        """
        Get repository pull requests via MCP.
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: PR state (open, closed, all)
            per_page: Number of PRs to retrieve
            
        Returns:
            List of pull requests
        """
        params = {
            "name": "list_pull_requests",
            "arguments": {
                "owner": owner,
                "repo": repo,
                "state": state,
                "per_page": per_page
            }
        }
        
        return await self._send_request("tools/call", params)
    
    async def search_repository_code(self, query: str, per_page: int = 10) -> Dict[str, Any]:
        """
        Search code across GitHub via MCP.
        
        Args:
            query: Search query (can include repo: qualifier)
            per_page: Number of results to retrieve
            
        Returns:
            Search results
        """
        params = {
            "name": "search_code",
            "arguments": {
                "q": query,
                "per_page": per_page
            }
        }
        
        return await self._send_request("tools/call", params)
    
    async def search_repository_code_in_repo(self, owner: str, repo: str, query: str, per_page: int = 10) -> Dict[str, Any]:
        """
        Search code in specific repository via MCP.
        
        Args:
            owner: Repository owner
            repo: Repository name
            query: Search query
            per_page: Number of results to retrieve
            
        Returns:
            Search results
        """
        search_query = f"repo:{owner}/{repo} {query}"
        return await self.search_repository_code(search_query, per_page)
    
    async def get_current_user(self) -> Dict[str, Any]:
        """
        Get information about the authenticated user via MCP.
        
        Returns:
            User information
        """
        params = {
            "name": "get_me",
            "arguments": {}
        }
        
        return await self._send_request("tools/call", params)
    
    async def create_issue(self, owner: str, repo: str, title: str, body: str = "", labels: List[str] = None) -> Dict[str, Any]:
        """
        Create a new issue via MCP.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Issue title
            body: Issue body (optional)
            labels: List of label names (optional)
            
        Returns:
            Created issue information
        """
        arguments = {
            "owner": owner,
            "repo": repo,
            "title": title
        }
        
        if body:
            arguments["body"] = body
        
        if labels:
            arguments["labels"] = labels
        
        params = {
            "name": "issue_write",
            "arguments": arguments
        }
        
        return await self._send_request("tools/call", params)
    
    async def create_pull_request(self, owner: str, repo: str, title: str, head: str, base: str, body: str = "") -> Dict[str, Any]:
        """
        Create a pull request via MCP.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: PR title
            head: Branch or commit SHA to merge from
            base: Branch to merge into
            body: PR body (optional)
            
        Returns:
            Created PR information
        """
        arguments = {
            "owner": owner,
            "repo": repo,
            "title": title,
            "head": head,
            "base": base
        }
        
        if body:
            arguments["body"] = body
        
        params = {
            "name": "create_pull_request",
            "arguments": arguments
        }
        
        return await self._send_request("tools/call", params)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.is_connected:
            asyncio.run(self.disconnect())


# Utility functions for easier usage
async def create_mcp_client(config: Optional[MCPConfig] = None) -> MCPGitHubClient:
    """
    Create and connect an MCP GitHub client.
    
    Args:
        config: Optional MCP configuration
        
    Returns:
        Connected MCPGitHubClient instance
        
    Raises:
        MCPGitHubConnectorError: If connection fails
    """
    client = MCPGitHubClient(config)
    
    if not await client.connect():
        raise MCPGitHubConnectorError("Failed to connect to MCP server")
    
    return client


async def test_mcp_connection(config: Optional[MCPConfig] = None) -> bool:
    """
    Test MCP server connection.
    
    Args:
        config: Optional MCP configuration
        
    Returns:
        True if connection test successful, False otherwise
    """
    try:
        client = await create_mcp_client(config)
        tools = await client.list_available_tools()
        await client.disconnect()
        
        logger.info(f"MCP connection test successful. Available tools: {len(tools)}")
        return True
        
    except Exception as e:
        logger.error(f"MCP connection test failed: {e}")
        return False


# Configuration helper functions
def create_remote_config(github_token: str, toolsets: List[str] = None, read_only: bool = False) -> MCPConfig:
    """
    Create configuration for remote GitHub MCP server.
    
    Args:
        github_token: GitHub personal access token
        toolsets: List of toolsets to enable (default: ["default"])
        read_only: Whether to enable read-only mode
        
    Returns:
        MCPConfig for remote server
    """
    return MCPConfig(
        connection_type="remote",
        github_token=github_token,
        toolsets=toolsets or ["default"],
        read_only=read_only
    )


def create_local_config(github_token: str, toolsets: List[str] = None, read_only: bool = False, use_docker: bool = True) -> MCPConfig:
    """
    Create configuration for local GitHub MCP server.
    
    Args:
        github_token: GitHub personal access token
        toolsets: List of toolsets to enable (default: ["default"])
        read_only: Whether to enable read-only mode
        use_docker: Whether to use Docker (True) or binary (False)
        
    Returns:
        MCPConfig for local server
    """
    if use_docker:
        local_command = "docker"
        local_args = [
            "run", "-i", "--rm", 
            "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server"
        ]
    else:
        local_command = "github-mcp-server"
        local_args = ["stdio"]
    
    return MCPConfig(
        connection_type="local",
        local_command=local_command,
        local_args=local_args,
        github_token=github_token,
        toolsets=toolsets or ["default"],
        read_only=read_only
    )
