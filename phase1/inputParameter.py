class InputParameter:

    def __init__(self, primary_param, func):
        self._primary_param = primary_param
        self._func = func

    @property
    def primary_param(self):
        return self._primary_param

    @property
    def func(self):
        return self._func