import numpy as np


class ULASampler:
    """
    Unadjusted Langevin Algorithm (ULA)

    x_{t+1} = x_t + 0.5 * step_size * grad_logp(x_t)
              + sqrt(step_size) * N(0, I)
    """

    def __init__(self, model, step_size=0.1):
        self.model = model
        self.step_size = step_size

    def step(self, x):

        grad = self.model.grad_logp(x)
        noise = np.random.normal(size=np.shape(x))

        x_next = (
            x
            + 0.5 * self.step_size * grad
            + np.sqrt(self.step_size) * noise
        )

        return x_next

    def sample(self, x0, n_samples):

        samples = []
        x = np.asarray(x0)

        for _ in range(n_samples):
            x = self.step(x)
            samples.append(x)

        return np.array(samples)