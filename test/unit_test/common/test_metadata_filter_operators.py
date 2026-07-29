from common.metadata_utils import meta_filter


def test_contains():
    # returns chunk where the metadata contains the value
    metas = {"version": {"hello earth": ["doc1"], "hello mars": ["doc2"]}}
    filters = [{"key": "version", "op": "contains", "value": "earth"}]

    assert meta_filter(metas, filters) == ["doc1"]


def test_not_contains():
    # returns chunk where the metadata does not contain the value
    metas = {"version": {"hello earth": ["doc1"], "hello mars": ["doc2"]}}
    filters = [{"key": "version", "op": "not contains", "value": "earth"}]

    assert meta_filter(metas, filters) == ["doc2"]


def test_in_operator():
    # returns chunk where the metadata is in the value
    metas = {"status": {"active": ["doc1"], "pending": ["doc2"], "done": ["doc3"]}}
    filters = [{"key": "status", "op": "in", "value": "active,pending"}]

    assert set(meta_filter(metas, filters)) == {"doc1", "doc2"}


def test_not_in_operator():
    # returns chunk where the metadata is not in the value
    metas = {"status": {"active": ["doc1"], "pending": ["doc2"], "done": ["doc3"]}}
    filters = [{"key": "status", "op": "not in", "value": "active,pending"}]

    assert meta_filter(metas, filters) == ["doc3"]


def test_start_with():
    # returns chunk where the metadata starts with the value
    metas = {"name": {"prefix_value": ["doc1"], "other": ["doc2"]}}
    filters = [{"key": "name", "op": "start with", "value": "pre"}]

    assert meta_filter(metas, filters) == ["doc1"]


def test_end_with():
    # returns chunk where the metadata ends with the value
    metas = {"file": {"report.pdf": ["doc1"], "image.png": ["doc2"]}}
    filters = [{"key": "file", "op": "end with", "value": ".pdf"}]

    assert meta_filter(metas, filters) == ["doc1"]


def test_empty():
    # returns chunk where the metadata is empty
    metas = {"notes": {"": ["doc1"], "non-empty": ["doc2"]}}
    filters = [{"key": "notes", "op": "empty", "value": ""}]

    assert meta_filter(metas, filters) == ["doc1"]


def test_not_empty():
    # returns chunk where the metadata is not empty
    metas = {"notes": {"": ["doc1"], "non-empty": ["doc2"]}}
    filters = [{"key": "notes", "op": "not empty", "value": ""}]

    assert meta_filter(metas, filters) == ["doc2"]


def test_equal():
    # returns chunk where the metadata is equal to the value
    metas = {"score": {"5": ["doc1"], "6": ["doc2"]}}
    filters = [{"key": "score", "op": "=", "value": "5"}]

    assert meta_filter(metas, filters) == ["doc1"]


def test_not_equal():
    # returns chunk where the metadata is not equal to the value
    metas = {"score": {"5": ["doc1"], "6": ["doc2"]}}
    filters = [{"key": "score", "op": "≠", "value": "5"}]

    assert meta_filter(metas, filters) == ["doc2"]


def test_greater_than():
    # returns chunk where the metadata is greater than the value
    metas = {"score": {"10": ["doc1"], "2": ["doc2"]}}
    filters = [{"key": "score", "op": ">", "value": "5"}]

    assert meta_filter(metas, filters) == ["doc1"]


def test_less_than():
    # returns chunk where the metadata is less than the value
    metas = {"score": {"10": ["doc1"], "2": ["doc2"]}}
    filters = [{"key": "score", "op": "<", "value": "5"}]

    assert meta_filter(metas, filters) == ["doc2"]


def test_greater_than_or_equal():
    # returns chunk where the metadata is greater than or equal to the value
    metas = {"score": {"5": ["doc1"], "6": ["doc2"], "4": ["doc3"]}}
    filters = [{"key": "score", "op": "≥", "value": "5"}]

    assert set(meta_filter(metas, filters)) == {"doc1", "doc2"}


def test_less_than_or_equal():
    # returns chunk where the metadata is less than or equal to the value
    metas = {"score": {"5": ["doc1"], "6": ["doc2"], "4": ["doc3"]}}
    filters = [{"key": "score", "op": "≤", "value": "5"}]

    assert set(meta_filter(metas, filters)) == {"doc1", "doc3"}


def test_and_of_several_conditions():
    # returns only the chunks matching every condition
    metas = {"owner": {"alice": ["doc1", "doc2"]}, "status": {"active": ["doc1"], "done": ["doc2"]}}
    filters = [
        {"key": "owner", "op": "=", "value": "alice"},
        {"key": "status", "op": "=", "value": "active"},
    ]

    assert meta_filter(metas, filters) == ["doc1"]


def test_and_with_a_leading_condition_that_matches_nothing():
    # an "and" whose first condition matches nothing is empty, it does not fall through to the second one
    metas = {"owner": {"alice": ["doc1"]}, "status": {"active": ["doc1"]}}
    filters = [
        {"key": "owner", "op": "=", "value": "bob"},
        {"key": "status", "op": "=", "value": "active"},
    ]

    assert meta_filter(metas, filters) == []


def test_and_with_a_leading_condition_on_an_absent_key():
    # same, for a key no chunk carries at all
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


def test_or_of_several_conditions():
    # returns the chunks matching any condition
    metas = {"owner": {"alice": ["doc1"], "bob": ["doc2"]}, "status": {"done": ["doc3"]}}
    filters = [
        {"key": "owner", "op": "=", "value": "alice"},
        {"key": "status", "op": "=", "value": "done"},
    ]

    assert set(meta_filter(metas, filters, "or")) == {"doc1", "doc3"}


def test_or_with_a_leading_condition_that_matches_nothing():
    # an empty condition contributes nothing to an "or" but must not discard the others
    metas = {"owner": {"alice": ["doc1"]}, "status": {"done": ["doc3"]}}
    filters = [
        {"key": "owner", "op": "=", "value": "bob"},
        {"key": "status", "op": "=", "value": "done"},
    ]

    assert meta_filter(metas, filters, "or") == ["doc3"]


def test_no_conditions():
    # no condition at all yields no chunks
    metas = {"owner": {"alice": ["doc1"]}}

    assert meta_filter(metas, []) == []
