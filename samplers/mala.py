import numpy as np


class MALASampler:
    """
    Metropolis Adjusted Langevin Algorithm (MALA)

    Uses a Langevin proposal and corrects it with a
    Metropolis-Hastings acceptance step.
    """

    def __init__(self, model, step_size=0.1):
        self.model = model
        self.step_size = step_size

    def _langevin_proposal(self, x):

        grad = self.model.grad_logp(x)
        noise = np.random.normal(size=np.shape(x))

        proposal = (
            x
            + 0.5 * self.step_size * grad
            + np.sqrt(self.step_size) * noise
        )

        return proposal

    def _proposal_log_density(self, x_from, x_to):

        grad = self.model.grad_logp(x_from)

        mean = x_from + 0.5 * self.step_size * grad
        diff = x_to - mean

        return -0.5 * np.sum(diff**2) / self.step_size

    def step(self, x):

        x_prop = self._langevin_proposal(x)

        logp_current = self.model.logp(x)
        logp_prop = self.model.logp(x_prop)

        log_q_forward = self._proposal_log_density(x, x_prop)
        log_q_backward = self._proposal_log_density(x_prop, x)

        log_accept = (
            logp_prop
            + log_q_backward
            - logp_current
            - log_q_forward
        )

        if np.log(np.random.rand()) < log_accept:
            return x_prop, True

        return x, False

    def sample(self, x0, n_samples):

        samples = []
        accepted = 0

        x = np.asarray(x0)

        for _ in range(n_samples):

            x, acc = self.step(x)

            if acc:
                accepted += 1

            samples.append(x)

        acceptance_rate = accepted / n_samples

        return np.array(samples), acceptance_rate