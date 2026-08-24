import numpy as np


class CustomAdamOptimizer:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        Initialize the Adam optimizer

        Parameters:
        - params: List of model parameters (NumPy arrays)
        - lr: Learning rate (default: 0.001)
        - betas: Exponential decay factors for the moving averages of gradients and squared gradients (default: (0.9, 0.999))
        - eps: Small constant added to the denominator for numerical stability (default: 1e-8)
        - weight_decay: Weight-decay (L2 regularization) parameter (default: 0)
        """
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        """
        Perform one parameter-update step

        Parameters:
        - grads: List of gradients (NumPy arrays) corresponding to the model parameters

        Returns:
        - List of updated model parameters (NumPy arrays)
        """
        self.t += 1
        bias_correction1 = 1 - self.betas[0]**self.t
        bias_correction2 = 1 - self.betas[1]**self.t
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if self.weight_decay != 0:
                grad += self.weight_decay * param

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad

            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)

            m_hat = self.m[i] / bias_correction1

            v_hat = self.v[i] / bias_correction2

            step = -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            self.params[i] += step

        return self.params
