"""
Custom exceptions for the Kimchi GitHub Assistant.
"""


class KimchiError(Exception):
    """Base exception for all Kimchi-related errors."""
    pass


class ConfigurationError(KimchiError):
    """Raised when configuration is invalid or incomplete."""
    pass


class ConnectorError(KimchiError):
    """Raised when connector operations fail."""
    pass


class QueryRoutingError(KimchiError):
    """Raised when query routing fails."""
    pass


class MCPError(ConnectorError):
    """Raised when MCP operations fail."""
    pass


class RAGError(ConnectorError):
    """Raised when RAG operations fail."""
    pass


class AIServiceError(KimchiError):
    """Raised when AI service operations fail."""
    pass
