import pytensor
import pytensor.tensor as pt


def build_logp_and_grad(logp_expr, variables):
    """
    Compile PyTensor functions for log-probability and its gradient.

    Parameters
    ----------
    logp_expr : pytensor variable
        Symbolic log probability expression.
    variables : pytensor variable
        Variable with respect to which we compute the gradient.

    Returns
    -------
    logp_fn : callable
        Function that evaluates log probability.
    grad_fn : callable
        Function that evaluates gradient of log probability.
    """

    grad_expr = pt.grad(logp_expr, variables)

    logp_fn = pytensor.function([variables], logp_expr)
    grad_fn = pytensor.function([variables], grad_expr)

    return logp_fn, grad_fn