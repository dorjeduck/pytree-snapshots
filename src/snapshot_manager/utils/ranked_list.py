from enum import Enum
from functools import cmp_to_key
from typing import Any, TypeVar, Callable, Optional
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class RankedItem:
    """Container for items in the ranked list."""
    order_id: int  # Unique identifier, increased by 1 for each new item
    item: Any  # The item itself

class OrderPolicy(Enum):
    """Policy for ordering items with equal ranking."""
    OLDEST_FIRST = 1  # Among equal items, oldest comes first (reject new equal items)
    NEWEST_FIRST = 2  # Among equal items, newest comes first (replace old equal items)

class RankedListAddResult(Enum):
    """Result of an add operation."""
    SUCCESS = 1  # Item was added
    EXISTS = 2   # Item already exists
    NOT_QUALIFIED = 3  # Item not added (e.g., list full and item not better)

class RankedList:
    """A ranked list that maintains strict ordering even for equal-ranked items."""
    
    def __init__(self, cmp: Callable[[Any, Any], int], max_items: Optional[int] = None, 
                 order_policy: OrderPolicy = OrderPolicy.NEWEST_FIRST):
        """Initialize a RankedList.
        
        Args:
            cmp: Function that takes two items and returns:
                - negative if first ranks higher than second
                - positive if first ranks lower than second
                - zero if equal rank
            max_items: Maximum items to keep (None for unlimited)
            order_policy: How to handle equal-ranked items
        """
        self._cmp = cmp  # Original comparison for items
        self._max_items = max_items
        self._order_policy = order_policy
        self._ranked_items = []  # List of RankedItems 
        self._next_order_id = 0
       
        
    def _internal_cmp(self, a: RankedItem, b: RankedItem) -> int:
        """Compare two RankedItems using the raw comparison function."""
        return self._cmp(a.item, b.item)
        
    def _combined_cmp(self, a, b):
        """Compare items by rank first, then by order based on policy."""
       
        
        # First compare by rank using provided cmp
        rank_cmp = self._internal_cmp(a, b)  
        if rank_cmp != 0:
            return rank_cmp
            
        # For equal ranks, use order based on policy
        if self._order_policy == OrderPolicy.OLDEST_FIRST:
            return b.order_id - a.order_id # Lower order (older) first 
        else:  # NEWEST_FIRST
            return a.order_id - b.order_id  # Higher order (newer) first 
    
    def add(self, item: Any) -> RankedListAddResult:
        """Add an item to the ranked list.
        
        Args:
            item: The item to add. If it's a RankedItem, use its ID directly.
            
        Returns:
            RankedListAddResult indicating success/failure
        """

        # check if item already exists by comparing with each item in the list
       
        if self._max_items == 0: # Empty but has max_items=0
            return RankedListAddResult.NOT_QUALIFIED

        for ranked_item in self._ranked_items:
           if item == ranked_item.item:
               return RankedListAddResult.EXISTS
        
        new_ranked_item = RankedItem(order_id=self._next_order_id, item=item)
        
        # If list is full, check if new item qualifies
        if self._max_items and len(self._ranked_items) >= self._max_items:

            # compare new item with last item in list 
            cmp_result = self._internal_cmp(new_ranked_item, self._ranked_items[-1])

            if cmp_result < 0 or (cmp_result == 0 and self._order_policy == OrderPolicy.OLDEST_FIRST):   
                return RankedListAddResult.NOT_QUALIFIED
            
            # Remove last item to make room
            self._ranked_items.pop()
            
            
        # Add new item
        self._ranked_items.append(new_ranked_item)
        self._next_order_id += 1
        
        # Sort by rank and order
        self._ranked_items.sort(key=cmp_to_key(self._combined_cmp), reverse=True)
        return RankedListAddResult.SUCCESS
        
    def remove(self, item: Any) -> bool:
        """Remove an item from the list.
        
        Args:
            item: The item to remove
                
        Returns:
            bool: True if removed, False if not found
        """

        removed = False
        for ranked_item in self._ranked_items:
            if item == ranked_item.item:
                self._ranked_items.remove(ranked_item)
                removed = True
        return removed
        
    def get_items(self) -> list[Any]:
        """Get all items in ranked order.
        
        Returns:
            list[Any]: Items in ranked order
        """
        return [ranked_item.item for ranked_item in self._ranked_items]
        
    def update_cmp(self, cmp: Callable[[Any, Any], int]):
        """Update comparison function and resort.
        
        Args:
            cmp: New comparison function
        """
        self._cmp = cmp
        self._ranked_items.sort(key=cmp_to_key(self._combined_cmp), reverse=True)
        
    def update_max_items(self, max_items: Optional[int]):
        """Update maximum items and truncate if needed.
        
        Args:
            max_items: New maximum items (None for unlimited)
        """
        self._max_items = max_items
        if max_items and len(self._ranked_items) > max_items:
            self._ranked_items = self._ranked_items[:max_items]
