class GNode:
    def __init__(self, **kwargs):
        self.string = None
        self.type = None
        self.name = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __eq__(self, other):
        if self.name == other.name and self.type == other.type:
            return True
        else:
            return False

    def __repr__(self):
        return self.__dict__.__repr__()

    def __hash__(self):
        return (self.name, self.type).__hash__()

    def setprop(self, key, value):
        setattr(self, key, value)
