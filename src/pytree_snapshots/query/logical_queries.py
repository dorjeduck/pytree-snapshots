from .base_queries import Query

class AndQuery(Query):
    def __init__(self, *queries):
        self.queries = queries

    def evaluate(self, snapshot):
        return all(query.evaluate(snapshot) for query in self.queries)


class OrQuery(Query):
    def __init__(self, *queries):
        self.queries = queries

    def evaluate(self, snapshot):
        return any(query.evaluate(snapshot) for query in self.queries)


class NotQuery(Query):
    def __init__(self, query):
        self.query = query

    def evaluate(self, snapshot):
        return not self.query.evaluate(snapshot)