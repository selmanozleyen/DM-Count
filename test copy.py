# %%
import torch


def projection_simplex(V, z=1, axis=None):
    """ Projection of x onto the simplex, scaled by z

        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        - axis=None: project V by P(V.ravel(); z)
        - axis=1: project each V[i] by P(V[i]; z[i])
        - axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = torch.sort(V, axis=1,descending=True)[0]
        z = torch.ones(len(V),device='cuda') * z
        cssv = torch.cumsum(U, axis=1) - z[:, None]
        ind = torch.arange(n_features,device='cuda') + 1
        cond = U - cssv / ind > 0
        rho = torch.count_nonzero(cond, axis=1)
        theta = cssv[torch.arange(len(V),device='cuda'), rho - 1] / rho
        return torch.maximum(V - theta[:, None], torch.zeros(1,device='cuda'))

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()

class Regularization(object):
    """Base class for Regularization objects

        Notes
        -----
        This class is not intended for direct use but as aparent for true
        regularizatiojn implementation.
    """

    def __init__(self, gamma=1.0):
        """

        Parameters
        ----------
        gamma: float
            Regularization parameter.
            We recover unregularized OT when gamma -> 0.

        """
        self.gamma = gamma

    def delta_Omega(X):
        """
        Compute delta_Omega(X[:, j]) for each X[:, j].
        delta_Omega(x) = sup_{y >= 0} y^T x - Omega(y).

        Parameters
        ----------
        X: array, shape = len(a) x len(b)
            Itorchut array.

        Returns
        -------
        v: array, len(b)
            Values: v[j] = delta_Omega(X[:, j])
        G: array, len(a) x len(b)
            Gradients: G[:, j] = nabla delta_Omega(X[:, j])
        """
        raise NotImplementedError

    def max_Omega(X, b):
        """
        Compute max_Omega_j(X[:, j]) for each X[:, j].
        max_Omega_j(x) = sup_{y >= 0, sum(y) = 1} y^T x - Omega(b[j] y) / b[j].

        Parameters
        ----------
        X: array, shape = len(a) x len(b)
            Itorchut array.

        Returns
        -------
        v: array, len(b)
            Values: v[j] = max_Omega_j(X[:, j])
        G: array, len(a) x len(b)
            Gradients: G[:, j] = nabla max_Omega_j(X[:, j])
        """
        raise NotImplementedError

    def Omega(T):
        """
        Compute regularization term.

        Parameters
        ----------
        T: array, shape = len(a) x len(b)
            Itorchut array.

        Returns
        -------
        value: float
            Regularization term.
        """
        raise NotImplementedError


class NegEntropy(Regularization):
    """ NegEntropy regularization """

    def delta_Omega(self, X):
        G = torch.exp(X / self.gamma - 1)
        val = self.gamma * torch.sum(G, axis=0)
        return val, G

    def max_Omega(self, X, b):
        max_X = torch.max(X, axis=0) / self.gamma
        exp_X = torch.exp(X / self.gamma - max_X)
        val = self.gamma * (torch.log(torch.sum(exp_X, axis=0)) + max_X)
        val -= self.gamma * torch.log(b)
        G = exp_X / torch.sum(exp_X, axis=0)
        return val, G

    def Omega(self, T):
        return self.gamma * torch.sum(T * torch.log(T))


class SquaredL2(Regularization):
    """ Squared L2 regularization """

    def delta_Omega(self, X):
        max_X = torch.maximum(X, 0)
        val = torch.sum(max_X ** 2, axis=0) / (2 * self.gamma)
        G = max_X / self.gamma
        return val, G

    def max_Omega(self, X, b):
        G = projection_simplex(X / (b * self.gamma), axis=0)
        val = torch.sum(X * G, axis=0)
        val -= 0.5 * self.gamma * b * torch.sum(G * G, axis=0)
        return val, G

    def Omega(self, T):
        return 0.5 * self.gamma * torch.sum(T ** 2)


# %%
# import numpy as np
X = torch.rand((1024,14),device='cuda')
b = torch.rand((14),device='cuda')

# V_ = V.numpy()

projection_simplex(X / b, axis=0)
# n_features = V.shape[1]
# U = torch.sort(V, axis=1,descending=True)[0]
# U_ = np.sort(V_, axis=1)[:, ::-1]
G = projection_simplex(X / (b * 1), axis=0)
val = torch.sum(X * G, axis=0)
val -= 0.5 * 1 * b * torch.sum(G * G, axis=0)
# a = torch.maximum(V -0.5, torch.zeros(1))
# b = np.maximum(V_ -0.5, np.zeros(1))
# # print(U,U.shape)
# z = torch.ones(len(V),device='cuda') * z
# cssv = torch.cumsum(U, axis=1) - z[:, None]
# ind = torch.arange(n_features,device='cuda') + 1
# cond = U - cssv / ind > 0
# rho = torch.count_nonzero(cond, axis=1)
# theta = cssv[torch.arange(len(V),device='cuda'), rho - 1] / rho
# return torch.maximum(V - theta[:, None], torch.zeros(1,device='cuda'))