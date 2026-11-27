"""The aggregations, by id, in the order every list and every page shows them.

To add one, write its function in ``aggregations.py`` and add it to ``BY_ID`` there.
"""

from __future__ import annotations

from ba.aggregations.aggregations import BY_ID
from ba.utils.registry import Registry

AGGREGATIONS = Registry("aggregations")
for aggregation_id, aggregate in BY_ID.items():
    AGGREGATIONS.register(aggregation_id, aggregate)
