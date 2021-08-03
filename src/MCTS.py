


class NODE():

    def __init__(self):
        raise NotImplementedError

    def explore(self, policy):
        raise NotImplementedError

    def next(self, temperature=0.1):
        raise NotImplementedError


        return next_tree, (v, nn_v, p, nn_p)