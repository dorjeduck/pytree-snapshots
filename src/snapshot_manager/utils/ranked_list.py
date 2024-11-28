from functools import cmp_to_key

from enum import Enum


class RankedListAddResult(Enum):
    SUCCESS = 1
    EXISTS = 2
    NOT_QUALIFIED = 3


class RankedList:
    def __init__(self, cmp, max_items=None):
        """
        A list that ranks items based on a custom cmp function.

        Args:

            cmp (callable): A function to compare two items.
                                             Should return:
                                             - Negative value if item1 < item2
                                             - 0 if item1 == item2
                                             - Positive value if item1 > item2
            max_items (int, optional): Maximum number of items to retain. Defaults to None (no limit).
        """
        self.ranked_items = []  # List of items sorted by the cmp
        self.max_items = max_items
        self.cmp = cmp

    def sort_items(self):
        """Sorts the ranked items using the cmp function."""
        if self.cmp:
            self.ranked_items.sort(key=cmp_to_key(self.cmp), reverse=True)

    def add(self, item):
        """
        Add an item to the list and ensure it is ranked correctly.

        Args:
            item: The item to add.

        Returns:
            bool: True if the item was added, False if it was not added
                (e.g., if the list is full and the item is not better).
        """

        # Check if the item is already in the list
        if item in self.ranked_items:
            return RankedListAddResult.EXISTS

        # If the list is full, compare the item with the lowest-ranked one
        if self.max_items is not None and len(self.ranked_items) >= self.max_items:

            if self.cmp(item, self.ranked_items[-1]) <= 0:
                # The new item does not qualify to be added
                return RankedListAddResult.NOT_QUALIFIED
            else:
                # Remove the lowest-ranked item as it will be replaced
                self.ranked_items.pop(-1)

        # Add the item and sort the list
        self.ranked_items.append(item)
        self.sort_items()

        return RankedListAddResult.SUCCESS

    def remove(self, item):
        """
        Remove an item from the list.

        Args:
            item: The item to remove.

        Returns:
            bool: True if the item was removed, False otherwise.
        """
        if item in self.ranked_items:
            self.ranked_items.remove(item)
            return True
        return False

    def update_cmp(self, cmp):
        """
        Update the cmp function and re-sort the list.

        Args:
            cmp (callable): A new cmp function.
        """
        self.cmp = cmp
        self.sort_items()

    def update_max_items(self, max_items):
        """
        Update the maximum number of items allowed and truncate the list if necessary.

        Args:
            max_items (int): The new maximum number of items allowed.
        """
        self.max_items = max_items
        if self.max_items is not None and len(self.ranked_items) > self.max_items:
            # Only truncate if the current list length exceeds the new max_items
            self.ranked_items = self.ranked_items[: self.max_items]

    def get_items(self):
        """
        Retrieve the list of ranked items.

        Returns:
            list: A list of ranked items.
        """
        return self.ranked_items.copy()
