class DotProduct:
    def __init__(self, arg, value, most_contributing_terms):
        self._arg = arg
        self._value = value
        self._most_contributing_terms = most_contributing_terms

    @property
    def arg(self):
        return self._arg

    @property
    def value(self):
        return self._value

    def contributing_terms(self, most_contributing_terms):
        self._most_contributing_terms = most_contributing_terms

    @property
    def contributing_terms(self):
        return self._most_contributing_terms

