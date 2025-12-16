"""
Entanglement Registry
Manages entangled groups of related entities with automatic correlation propagation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np


@dataclass
class EntangledGroup:
    """Represents a group of entangled entities"""

    group_id: str
    entity_ids: Set[str]
    correlation_strength: float  # 0.0 to 1.0
    metadata: Dict = field(default_factory=dict)


class EntanglementRegistry:
    """
    Tracks and manages entangled relationships between entities
    """

    def __init__(self):
        self.groups: Dict[str, EntangledGroup] = {}
        self.entity_to_groups: Dict[str, Set[str]] = {}  # entity_id -> group_ids

    def create_entangled_group(
        self, group_id: str, entity_ids: List[str], correlation_strength: float = 0.85
    ) -> EntangledGroup:
        """
        Create a new entangled group

        Args:
            group_id: Unique identifier for the group
            entity_ids: List of entity IDs to entangle
            correlation_strength: Strength of correlation (0-1)

        Returns:
            Created EntangledGroup
        """
        if not 0.0 <= correlation_strength <= 1.0:
            raise ValueError("Correlation strength must be between 0 and 1")

        group = EntangledGroup(
            group_id=group_id, entity_ids=set(entity_ids), correlation_strength=correlation_strength
        )

        self.groups[group_id] = group

        # Update entity-to-group mapping
        for entity_id in entity_ids:
            if entity_id not in self.entity_to_groups:
                self.entity_to_groups[entity_id] = set()
            self.entity_to_groups[entity_id].add(group_id)

        return group

    def update_entity(self, entity_id: str, new_data: np.ndarray) -> List[str]:
        """
        Update an entity and propagate changes to entangled partners

        Args:
            entity_id: ID of entity being updated
            new_data: New vector data

        Returns:
            List of affected entity IDs (entangled partners)
        """
        affected = []

        # Find all groups this entity belongs to
        group_ids = self.entity_to_groups.get(entity_id, set())

        for group_id in group_ids:
            group = self.groups[group_id]

            # All other entities in this group are affected
            partners = group.entity_ids - {entity_id}
            affected.extend(partners)

        return list(set(affected))  # Remove duplicates

    def get_entangled_partners(self, entity_id: str) -> List[str]:
        """
        Get all entities entangled with given entity

        Args:
            entity_id: Entity to find partners for

        Returns:
            List of entangled partner IDs
        """
        partners = set()

        group_ids = self.entity_to_groups.get(entity_id, set())

        for group_id in group_ids:
            group = self.groups[group_id]
            partners.update(group.entity_ids - {entity_id})

        return list(partners)

    def measure_correlation(self, entity_a: str, entity_b: str) -> Optional[float]:
        """
        Measure correlation strength between two entities

        Args:
            entity_a: First entity ID
            entity_b: Second entity ID

        Returns:
            Correlation strength (0-1), or None if not entangled
        """
        groups_a = self.entity_to_groups.get(entity_a, set())
        groups_b = self.entity_to_groups.get(entity_b, set())

        # Find common groups
        common_groups = groups_a & groups_b

        if not common_groups:
            return None

        # Return maximum correlation strength across common groups
        max_correlation = 0.0
        for group_id in common_groups:
            group = self.groups[group_id]
            max_correlation = max(max_correlation, group.correlation_strength)

        return max_correlation

    def add_entity_to_group(self, group_id: str, entity_id: str):
        """
        Add an entity to an existing entangled group

        Args:
            group_id: Group to add to
            entity_id: Entity to add
        """
        if group_id not in self.groups:
            raise ValueError(f"Group {group_id} does not exist")

        group = self.groups[group_id]
        group.entity_ids.add(entity_id)

        if entity_id not in self.entity_to_groups:
            self.entity_to_groups[entity_id] = set()
        self.entity_to_groups[entity_id].add(group_id)

    def remove_entity_from_group(self, group_id: str, entity_id: str):
        """
        Remove an entity from an entangled group

        Args:
            group_id: Group to remove from
            entity_id: Entity to remove
        """
        if group_id in self.groups:
            group = self.groups[group_id]
            group.entity_ids.discard(entity_id)

            # Remove group if empty
            if not group.entity_ids:
                del self.groups[group_id]

        if entity_id in self.entity_to_groups:
            self.entity_to_groups[entity_id].discard(group_id)

            # Clean up if no groups
            if not self.entity_to_groups[entity_id]:
                del self.entity_to_groups[entity_id]

    def get_group(self, group_id: str) -> Optional[EntangledGroup]:
        """
        Retrieve an entangled group by ID

        Args:
            group_id: Group identifier

        Returns:
            EntangledGroup or None if not found
        """
        return self.groups.get(group_id)

    def get_groups_for_entity(self, entity_id: str) -> List[EntangledGroup]:
        """
        Get all groups an entity belongs to

        Args:
            entity_id: Entity identifier

        Returns:
            List of EntangledGroups
        """
        group_ids = self.entity_to_groups.get(entity_id, set())
        return [self.groups[gid] for gid in group_ids if gid in self.groups]
