import numpy as np
import pytensor
import pytensor.tensor as pt


class GaussianModel:
    """
    Simple 1D standard Gaussian model.

    p(x) = N(0, 1)

    This class builds a PyTensor computational graph for:
    - log probability
    - gradient of log probability
    """

    def __init__(self):

        # symbolic variable
        x = pt.dscalar("x")

        # log probability of standard normal
        logp = -0.5 * x**2

        # gradient of log probability
        grad_logp = pt.grad(logp, x)

        # compile pytensor functions
        self.logp_fn = pytensor.function([x], logp)
        self.grad_logp_fn = pytensor.function([x], grad_logp)

    def logp(self, x):
        """Evaluate log probability."""
        return self.logp_fn(x)

    def grad_logp(self, x):
        """Evaluate gradient of log probability."""
        return self.grad_logp_fn(x)