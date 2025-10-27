"""
Connectors package for handling different data sources and destinations.
"""

from .github_connector import GitHubConnector, GitHubConfig, GitHubConnectorError
from .mcp_github_connector import MCPGitHubClient, MCPConfig, MCPGitHubConnectorError
from .elasticsearch_connector import (
    ElasticsearchConnector, 
    ElasticsearchConfig, 
    ElasticsearchConnectorError,
    DocumentProcessor,
    ParserConfig
)

__all__ = [
    'GitHubConnector', 
    'GitHubConfig', 
    'GitHubConnectorError',
    'MCPGitHubClient',
    'MCPConfig',
    'MCPGitHubConnectorError',
    'ElasticsearchConnector', 
    'ElasticsearchConfig', 
    'ElasticsearchConnectorError',
    'DocumentProcessor',
    'ParserConfig'
]
