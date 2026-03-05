cache = {}


def get_cached(query):
    return cache.get(query)


def set_cache(query, answer):
    cache[query] = answer