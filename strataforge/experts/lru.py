"""A tiny LRU cache (OrderedDict) for the GPU hot-expert tier."""


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self._d = {}
        self._order = []

    def __contains__(self, key):
        return key in self._d

    def get(self, key):
        if key in self._d:
            self._order.remove(key)
            self._order.append(key)
            return self._d[key]
        return None

    def put(self, key, value):
        if key in self._d:
            self._d[key] = value
            self._order.remove(key)
            self._order.append(key)
            return
        if len(self._d) >= self.capacity:
            evict = self._order.pop(0)
            del self._d[evict]
            self.evicted = evict
        else:
            self.evicted = None
        self._d[key] = value
        self._order.append(key)

    def __len__(self):
        return len(self._d)
