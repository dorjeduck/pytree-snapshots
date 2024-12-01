import pytest
from snapshot_manager.utils.ranked_list import (
    RankedList, RankedListAddResult, OrderPolicy
)


def test_ranked_list_basic():
    """Test basic functionality of RankedList."""
    def cmp_func(a, b) -> int:
        return b - a

    ranked_list = RankedList(cmp=cmp_func,max_items=3)
    
    # Add items in random order
    for i in range(5):
        if i < 3:
            assert ranked_list.add(i) == RankedListAddResult.SUCCESS
        else:
            assert ranked_list.add(i) == RankedListAddResult.NOT_QUALIFIED
    
    items = ranked_list.get_items()
    assert len(items) == 3
    assert items == [0, 1, 2]


def test_ranked_list_duplicates():
    """Test handling of duplicate items."""
    def cmp_func(a, b) -> int:
        return a - b
    
    ranked_list = RankedList(cmp_func)
    
    # Add original item
    assert ranked_list.add(100) == RankedListAddResult.SUCCESS
    
    # Try adding duplicate (same object)
    assert ranked_list.add(100) == RankedListAddResult.EXISTS


def test_ranked_list_update_cmp():
    """Test updating comparison function."""
    def cmp_asc(a, b) -> int:
        return a - b
        
    def cmp_desc(a, b) -> int:
        return b - a
    
    ranked_list = RankedList(cmp_asc)
    
    # Add items in ascending order
    for i in range(3):
        ranked_list.add(i)
    
    # Initially sorted ascending
    assert ranked_list.get_items() == [2, 1, 0]
    
    # Update to descending order
    ranked_list.update_cmp(cmp_desc)
    assert ranked_list.get_items() == [0, 1, 2]


def test_equal_ranked_items_oldest_first():
    """Test ordering of equal-ranked items with OLDEST_FIRST policy."""
    def cmp_func(a, b) -> int:
        return a["score"] - b["score"]
    
    ranked_list = RankedList(cmp_func, order_policy=OrderPolicy.OLDEST_FIRST)
    
    # Add items with equal scores
    items = [
        {"score": 100, "id": "a"},
        {"score": 100, "id": "b"},
        {"score": 100, "id": "c"},
    ]
    
    # First two should be added
    assert ranked_list.add(items[0]) == RankedListAddResult.SUCCESS
    assert ranked_list.add(items[1]) == RankedListAddResult.SUCCESS
    
    # Third should be added but maintain oldest-first order
    assert ranked_list.add(items[2]) == RankedListAddResult.SUCCESS
    
    result = ranked_list.get_items()
    assert len(result) == 3
    assert [item["id"] for item in result] == ["a", "b", "c"]


def test_equal_ranked_items_newest_first():
    """Test ordering of equal-ranked items with NEWEST_FIRST policy."""
    def cmp_func(a, b) -> int:
        return a["score"] - b["score"]
    
    ranked_list = RankedList(cmp_func, order_policy=OrderPolicy.NEWEST_FIRST)
    
    # Add items with equal scores
    items = [
        {"score": 100, "id": "a"},
        {"score": 100, "id": "b"},
        {"score": 100, "id": "c"},
    ]
    
    # Items should be added in reverse order
    for item in items:
        assert ranked_list.add(item) == RankedListAddResult.SUCCESS
    
    result = ranked_list.get_items()
    assert len(result) == 3
    assert [item["id"] for item in result] == ["c", "b", "a"]


def test_ranked_list_remove():
    """Test removing items from ranked list."""
    def cmp_func(str1, str2) -> int:
        if str1 < str2:
            return -1
        elif str1 > str2:
            return 1
        else:
            return 0

    
    ranked_list = RankedList(cmp=cmp_func) 
    
    # Add and remove items
    ranked_list.add("100")
    ranked_list.add("200")
    
    assert ranked_list.remove("100") == True
    assert len(ranked_list.get_items()) == 1
    assert ranked_list.get_items()[0] == "200"


def test_ranked_list_update_max_items():
    """Test updating max items."""
    def cmp_func(a, b) -> int:
        return a - b
    
    ranked_list = RankedList(cmp_func)
    
    # Add 5 items
    for i in range(5):
        ranked_list.add(i)
    
    assert len(ranked_list.get_items()) == 5
    
    # Update max_items to 3
    ranked_list.update_max_items(3)
    items = ranked_list.get_items()
    assert len(items) == 3
    assert items == [4, 3, 2]
