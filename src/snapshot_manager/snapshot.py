import time
import copy
import uuid


class Snapshot:
    def __init__(self, data, metadata=None, tags=None, deepcopy=True, snapshot_id=None):
        """
        Initialize a Snapshot instance.

        Args:
            id: snapshot id,
            data: The data structure to store in the snapshot.
            metadata (dict, optional): User-defined metadata.
            tags (list, optional): Tags associated with the snapshot.
            deepcopy (bool, optional): If True, creates deep copies of the data, metadata, and tags;
                                        otherwise, stores references.
            snapshot_id (str, optional): A unique identifier for the snapshot. If not provided, a new ID is generated.
        """

        self.id = snapshot_id or str(uuid.uuid4())
        self.timestamp = time.time()
        self.data = copy.deepcopy(data) if deepcopy else data
        self.metadata = (
            copy.deepcopy(metadata) if deepcopy and metadata else (metadata or {})
        )
        self.tags = copy.deepcopy(tags) if deepcopy and tags else list(tags or [])

    def get_id(self):
        """
        Retrieve the snapshot ID.

        Returns:
            str: The snapshot's unique identifier.
        """
        return self.id

    def clone(self, snapshot_id=None):
        """
        Create a deep copy of the entire Snapshot object.

        Args:
            snapshot_id (str, optional): A unique identifier for the cloned snapshot.
                                        If not provided, a new ID is generated.

        Returns:
            Snapshot: A new Snapshot instance with the same data, metadata, tags, and timestamp.
        """
        return Snapshot(
            data=self.data,
            metadata=self.metadata,
            tags=self.tags,
            deepcopy=True,  # Deepcopy will handle all fields internally
            snapshot_id=snapshot_id
            or str(uuid.uuid4()),  # Use provided ID or generate a new one
        )

    def to_dict(self):
        """Convert the snapshot to a dictionary for serialization."""
        return {
            "id": self.id,
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
            snapshot_id=data["id"],
            metadata=data.get("metadata"),
            tags=data.get("tags"),
            deepcopy=False,
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

    def __eq__(self, other):
        """
        Compares snapshots for equality based on their IDs.

        Args:
            other (Snapshot): Another snapshot to compare against.

        Returns:
            bool: True if the snapshots have the same ID, False otherwise.
        """
        if isinstance(other, Snapshot):
            return self.id == other.id
        return False

    def __hash__(self):
        """
        Returns a hash value based on the snapshot ID.

        This ensures that snapshots can be used as keys in dictionaries or added to sets.

        Returns:
            int: The hash value of the snapshot ID.
        """
        return hash(self.id)
    
    
    def __repr__(self):
        """
        Return a string representation of the Snapshot object for debugging and logging.
        """
        return (
            f"Snapshot(id={self.id}, timestamp={self.timestamp}, "
            f"data={self.data}, metadata={self.metadata}, tags={self.tags})"
        )
