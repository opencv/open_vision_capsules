def cache(method):
    """Caches the result of a method call. Caches are specific to the instance
    of the object that this method is for.
    """

    def on_call(self, *args, **kwargs):
        name = method.__name__
        try:
            return self._cache[name]
        except AttributeError:
            # Create the cache if necessary
            self._cache = {}
        except KeyError:
            # Handled below
            pass

        val = method(self, *args, **kwargs)
        self._cache[name] = val
        return val

    return on_call
