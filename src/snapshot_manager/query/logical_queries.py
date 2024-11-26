from .base_queries import Query


class AndQuery(Query):
    def __init__(self, *queries):
        if not all(hasattr(query, "evaluate") for query in queries):
            raise ValueError("All items in queries must have an 'evaluate' method.")
        self.queries = queries

    def evaluate(self, snapshot):
        return all(query.evaluate(snapshot) for query in self.queries)


class OrQuery(Query):
    def __init__(self, *queries):
        if not all(hasattr(query, "evaluate") for query in queries):
            raise ValueError("All items in queries must have an 'evaluate' method.")
        self.queries = queries

    def evaluate(self, snapshot):
        return any(query.evaluate(snapshot) for query in self.queries)


class NotQuery(Query):
    def __init__(self, query):
        self.query = query

    def evaluate(self, snapshot):
        return not self.query.evaluate(snapshot)
