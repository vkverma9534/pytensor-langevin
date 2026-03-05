import numpy as np
import pytensor
import pytensor.tensor as pt


class BananaModel:
    """
    2D banana-shaped distribution.

    This distribution is often used as a toy example in MCMC
    because the curvature makes sampling harder than a simple
    Gaussian.
    """

    def __init__(self, b=0.03):

        x = pt.dvector("x")

        x1 = x[0]
        x2 = x[1]

        # banana-shaped log probability
        logp = -0.5 * x1**2 - 0.5 * (x2 + b * x1**2 - 1)**2

        grad_logp = pt.grad(logp, x)

        self.logp_fn = pytensor.function([x], logp)
        self.grad_logp_fn = pytensor.function([x], grad_logp)

    def logp(self, x):
        return self.logp_fn(np.asarray(x))

    def grad_logp(self, x):
        return self.grad_logp_fn(np.asarray(x))