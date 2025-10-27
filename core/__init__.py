"""
Core components of the Kimchi GitHub Assistant.

This package contains the main business logic and core functionality.
"""

from .assistant import KimchiAssistant
from .query_router import QueryRouter, QueryType, RoutingDecision

__all__ = ['KimchiAssistant', 'QueryRouter', 'QueryType', 'RoutingDecision']
