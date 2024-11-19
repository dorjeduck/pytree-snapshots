import time
import pickle
import zlib
import copy


class Snapshot:
    def __init__(self, data, metadata=None, tags=None, deepcopy=True):
        """
        Initialize a Snapshot instance.

        Args:
            data: The data structure to store in the snapshot.
            metadata (dict, optional): User-defined metadata.
            tags (list, optional): Tags associated with the snapshot.
            deepcopy (bool, optional): Whether to deepcopy the data when saving.


        """
        self.timestamp = time.time()
        self.metadata = metadata or {}
        self.tags = list(tags or [])
        self.data = copy.deepcopy(data) if deepcopy else data

    def to_dict(self):
        """Convert the snapshot to a dictionary for serialization."""
        return {
            "data": self.data,
            "metadata": self.metadata,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data):
        """Recreate a snapshot from a dictionary."""
        return cls(
            data=data["data"],
            metadata=data.get("metadata"),
            tags=data.get("tags"),
        )

    def get_data(self, deepcopy=True):
        """
        Retrieve the stored data.

        Args:
            deepcopy (bool, optional): Whether to return a deep copy.

        Returns:
            The stored data.
        """
        return copy.deepcopy(self.data) if deepcopy else self.data

    def add_tags(self, tags):
        """
        Add tags to the snapshot.

        Args:
            tags (list): Tags to add.
        """
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

    def remove_tags(self, tags):
        """
        Remove tags from the snapshot.

        Args:
            tags (list): Tags to remove.
        """
        self.tags = [tag for tag in self.tags if tag not in tags]

    def get_tags(self):
        """
        Retrieve the list of tags associated with the snapshot.

        Returns:
            list: The list of tags.
        """
        return self.tags

    def has_tag(self, tag):
        """
        Check if the snapshot has a specific tag.

        Args:
            tag (str): The tag to check.

        Returns:
            bool: True if the tag exists, False otherwise.
        """
        return tag in self.tags

    def get_metadata(self):
        """
        Retrieve the metadata associated with the snapshot.

        Returns:
            dict: The metadata dictionary.
        """
        return self.metadata

    def set_metadata(self, metadata):
        """
        Set or replace the metadata for the snapshot.

        Args:
            metadata (dict): The new metadata.
        """
        self.metadata = metadata

    def get_timestamp(self):
        """
        Retrieve the timestamp of when the snapshot was created.

        Returns:
            float: The timestamp.
        """
        return self.timestamp

    def set_timestamp(self, timestamp):
        """
        Set or update the timestamp for the snapshot.

        Args:
            timestamp (float): The new timestamp value.
        """
        self.timestamp = timestamp

    def update_metadata(self, new_metadata):
        """
        Update the metadata of the snapshot.

        Args:
            new_metadata (dict): Metadata to update.
        """
        self.metadata.update(new_metadata)
