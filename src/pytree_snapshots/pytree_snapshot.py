import time
import pickle
import zlib
import copy 

class PytreeSnapshot:
    def __init__(self, pytree, metadata=None, tags=None, compress=False):
        """
        Initialize a PytreeSnapshot instance.

        Args:
            pytree: The data structure to store in the snapshot.
            metadata (dict, optional): User-defined metadata.
            tags (list, optional): Tags associated with the snapshot.
            compress (bool, optional): Whether to compress the snapshot data.
        """
        self.timestamp = time.time()
        self.metadata = metadata or {}
        self.tags = list(tags or [])
        self.compress = compress

        # Store the pytree, compressing it if needed
        self.pytree = self._compress(pytree) if compress else pytree

    def to_dict(self):
        """Convert the snapshot to a dictionary for serialization."""
        return {
            "pytree": self._decompress() if self.compress else self.pytree,
            "metadata": self.metadata,
            "tags": self.tags,
            "compress": self.compress,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data):
        """Recreate a snapshot from a dictionary."""
        return cls(
            pytree=data["pytree"],
            metadata=data.get("metadata"),
            tags=data.get("tags"),
            compress=data.get("compress", False),
        )

    def _compress(self, pytree):
        try:
            return zlib.compress(pickle.dumps(pytree))
        except Exception as e:
            raise RuntimeError("Compression failed") from e

    def _decompress(self):
        try:
            return pickle.loads(zlib.decompress(self.pytree)) if self.compress else self.pytree
        except Exception as e:
            raise RuntimeError("Decompression failed") from e
    
    def get_pytree(self, deepcopy=True):
        """
        Retrieve the pytree, decompressing if needed.

        Args:
            deepcopy (bool, optional): Whether to return a deep copy.

        Returns:
            The pytree.
        """
        pytree = self._decompress() if self.compress else self.pytree
        return copy.deepcopy(pytree) if deepcopy else pytree

    def update_metadata(self, new_metadata):
        """
        Update the metadata of the snapshot.

        Args:
            new_metadata (dict): Metadata to update.
        """
        self.metadata.update(new_metadata)

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