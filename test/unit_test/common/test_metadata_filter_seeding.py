"""How ``meta_filter`` seeds its result set from the first condition.

``meta_filter`` starts from ``doc_ids = None`` and seeds the set with the first
condition's matches, whatever they are. The cases here are the ones where that
seeding is the whole behaviour under test: a leading condition that matches nothing
must leave an "and" empty rather than let a later condition seed it, and a filter
list with nothing in it must match nothing rather than everything.

``test_metadata_filter_operators.py`` covers the same seeding for a leading
condition on a key that *exists* but matches no value; these add the key that no
chunk carries at all, a run of conditions that all match nothing, and the empty
filter list.
"""

from common.metadata_utils import meta_filter


def test_and_with_a_leading_condition_on_an_absent_key():
    # a key no chunk carries at all seeds the "and" empty, it does not fall through
    metas = {"status": {"active": ["doc1"]}}
    filters = [
        {"key": "owner", "op": "=", "value": "alice"},
        {"key": "status", "op": "=", "value": "active"},
    ]

    assert meta_filter(metas, filters) == []


def test_and_with_every_condition_matching_nothing():
    # the whole leading run of empty conditions must stay empty, whichever one would have seeded
    metas = {"owner": {"alice": ["doc1"]}, "status": {"active": ["doc1"]}}
    filters = [
        {"key": "owner", "op": "=", "value": "bob"},
        {"key": "missing", "op": "=", "value": "x"},
        {"key": "status", "op": "=", "value": "active"},
    ]

    assert meta_filter(metas, filters) == []


def test_no_conditions():
    # no condition at all yields no chunks
    metas = {"owner": {"alice": ["doc1"]}}

    assert meta_filter(metas, []) == []
