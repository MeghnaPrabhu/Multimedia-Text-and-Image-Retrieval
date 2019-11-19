class TermInfo:
    def __init__(self, term, model):
        self._term = term
        self._model = model

    @property
    def model(self):
        return self._model

    @property
    def term(self):
        return self._term
