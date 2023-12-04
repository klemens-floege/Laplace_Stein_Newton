import torch
import torch.nn as nn

from scipy import optimize

class TorchGaussian(nn.Module):
    def __init__(self):
        super(TorchGaussian, self).__init__()

        self.DoF = 2
        self.nData = 1
        self.mu0 = torch.zeros((self.DoF, 1), dtype=torch.float32)
        self.std0 = torch.ones((self.DoF, 1), dtype=torch.float32)
        self.var0 = self.std0 ** 2
        self.stdn = 0.3
        self.varn = self.stdn ** 2

        torch.manual_seed(40)
        self.A = torch.randn((self.DoF, 1), dtype=torch.float32)
        self.thetaTrue = torch.randn(self.DoF, dtype=torch.float32)
        self.data = self.simulate_data()

    def get_forward_model(self, thetas):
        nSamples = thetas.size(0) // self.DoF
        thetas = thetas.view(self.DoF, nSamples)
        tmp = torch.sum(self.A * thetas, 0)
        return tmp if nSamples > 1 else tmp.squeeze()

    def get_jacobian_forward_model(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas = thetas.view(self.DoF, nSamples)
        tmp = self.A.repeat(1, nSamples)
        return tmp if nSamples > 1 else tmp.squeeze()

    def simulate_data(self):
        noise = torch.normal(mean=0, std=self.stdn, size=(1, self.nData))
        return self.get_forward_model(self.thetaTrue) + torch.normal(mean=0, std=self.stdn, size=(1, self.nData))

    def get_minus_log_prior(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas = thetas.view(self.DoF, nSamples)
        shift = thetas - self.mu0
        tmp = 0.5 * torch.sum(shift ** 2 / self.var0, 0)
        return tmp if nSamples > 1 else tmp.squeeze()

    def get_minus_log_likelihood(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        thetas = thetas.view(self.DoF, nSamples)
        F = arg[0] if len(arg) > 0 else self.get_forward_model(thetas)
        shift = F - self.data
        tmp = 0.5 * torch.sum(shift ** 2, 0) / self.varn
        return tmp if nSamples > 1 else tmp.squeeze()

    def get_minus_log_posterior(self, thetas, *arg):
        return self.get_minus_log_prior(thetas) + self.get_minus_log_likelihood(thetas, *arg)

    def get_gradient_minus_log_prior(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas = thetas.view(self.DoF, nSamples)
        tmp = (thetas - self.mu0) / self.var0
        return tmp if nSamples > 1 else tmp.squeeze()

    def get_gradient_minus_log_likelihood(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        thetas = thetas.view(self.DoF, nSamples)
        F = arg[0] if len(arg) > 0 else self.get_forward_model(thetas)
        J = arg[1] if len(arg) > 1 else self.get_jacobian_forward_model(thetas)
        tmp = J * torch.sum(F - self.data, 0) / self.varn
        return tmp if nSamples > 1 else tmp.squeeze()

    def get_gradient_minus_log_posterior(self, thetas, *arg):
        return self.get_gradient_minus_log_prior(thetas) + self.get_gradient_minus_log_likelihood(thetas, *arg)

    def get_gn_hessian_minus_log_posterior(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        thetas = thetas.view(self.DoF, nSamples)
        J = arg[0] if len(arg) > 1 else self.get_jacobian_forward_model(thetas)
        tmp = J.view(self.DoF, 1, nSamples) * J.view(1, self.DoF, nSamples) / self.varn \
              + torch.eye(self.DoF).view(self.DoF, self.DoF, 1)
        return tmp if nSamples > 1 else tmp.squeeze()

    def getMAP(self, *arg):
        x0 = arg[0] if len(arg) > 0 else torch.randn(self.DoF, dtype=torch.float32)

        if x0.size(0) % self.DoF != 0:
            raise ValueError("Number of elements in the initial thetas must be a multiple of self.DoF.")

        nSamples = x0.size(0) // self.DoF
        x0 = x0.view(self.DoF, nSamples)

        # Print the shape of x0
        print("Shape of x0:", x0.shape)

        # Convert x0 to NumPy array
        x0_numpy = x0.detach().cpu().numpy()

        print(x0_numpy.shape)

        res = optimize.minimize(self.get_minus_log_posterior, x0_numpy, method='L-BFGS-B')
        return torch.tensor(res.x)



