"""
Smart backend routing and selection system.

This module provides intelligent backend selection based on:
- Circuit complexity analysis
- Backend capabilities and performance
- Cost optimization
- Multi-objective scoring
"""

from .smart_router import (
    SmartBackendRouter,
    BackendScore,
    RoutingStrategy,
    create_smart_router
)

__all__ = [
    'SmartBackendRouter',
    'BackendScore',
    'RoutingStrategy',
    'create_smart_router',
]
