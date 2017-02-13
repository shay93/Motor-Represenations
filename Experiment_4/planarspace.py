import numpy as np


class PlanarSpace(object):
    """
    Provides a classification state spaces and action spaces,
    so you can write generic code that applies to any Environment.
    E.g. to choose a random action.
    """

    def sample(self, seed=0):
        """
        Uniformly randomly sample a random elemnt of this space
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def flatten(self, x):
        return x.flatten()

    def unflatten(self, x):
        return x.reshape(64,64)

    @property
    def flat_dim(self):
        """
        The dimension of the flattened vector of the tensor representation
        """
        return 64**2

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a Theano tensor variable given the name and extra dimensions prepended
        :param name: name of the variable
        :param extra_dims: extra dimensions in the front
        :return: the created tensor variable
        """
        raise NotImplementedError
