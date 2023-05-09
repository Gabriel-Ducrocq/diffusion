import matplotlib.pyplot as plt
import numpy as np
import torch as torch


class BrownianBridge:

    def __init__(self, dimension, a=3, b=4, T0 = 0, TN= 1):
        """
        Brownian Bridge diffusion object, with drift sigma_t.
        :param dimension: integer, dimension of the state space
        :param sigma_t: real valued function taking time as input, drift function
        :param int_sigma_sq_t: real valued function, taking integral bounds as input, integrting the square of sigma
        :param T0: integer, starting time.
        :param TN: integer, ending time
        """

        self.a = a
        self.b = b
        self.dimension = dimension
        self.T0 = T0
        self.TN = TN


    def sigma_t(self, t):
        return self.a*np.exp(-self.b*t)

    def int_sigma_sq_t(self, t0, t):
        return - self.a**2/(2*self.b) * (np.exp(-2*self.b*t) - np.exp(-2*self.b*t0))

    def get_means(self, t, x0, xT,  t0=0, T=1):
        """
        get the conditionnal mean of the Brownian bridge
        :param t: torch tensor(N_batch, 1), times at which we sample
        :param x0: torch tensor (N_batch, self.dimension), starting values of the bridge.
        :param xT: torch tensor (N_batch, self.dimension), ending values of the bridge.
        :param t0: torch tensor(N_batch, 1), times of the starting value
        :param T: torch tensor(N_batch, 1), times of the ending values
        :return: torch tensor (N_batch, self.dimension), conditional means at tme t
        """
        return (self.int_sigma_sq_t(t0, t)/self.int_sigma_sq_t(t0, T))*(xT - x0) + x0

    def get_variances(self, t, t0=0, T=1):
        """
        get the conditionnal variance of the Brownian bridge, where the variance sigma_t is supposed to be proportional
        to the identity matric
        :param t: float, time at which we sample
        :param t0: float, time of the starting value
        :param T: float, time of the ending value
        :return: torch tensor (N_batch, 1), conditional variances at time t
        """
        ### Change from T to t in the first term of the product
        return self.int_sigma_sq_t(t0, t)*(1- self.int_sigma_sq_t(t0, t)/self.int_sigma_sq_t(t0, T))

    def sample(self, t, x0, xT,  t0=0, T=1):
        """
        get the conditionnal mean of the Brownian bridge
        :param t: torch tensor(N_batch, 1), times at which we sample
        :param x0: torch tensor (N_batch, self.dimension), starting values of the bridge.
        :param xT: torch tensor (N_batch, self.dimension), ending values of the bridge.
        :param t0: torch tensor(N_batch, 1), times of the starting value
        :param T: torch tensor(N_batch, 1), times of the ending values
        :return: torch tensor (N_batch, self.dimension), sampled conditional value marginal.
        """
        means = self.get_means(t, x0, xT, t0, T)
        variances = self.get_variances(t, t0, T)
        return torch.randn_like(means)*torch.sqrt(variances) + means


    def compute_drift_maruyama(self, x_t, t, tau, network, t0=0, Unet=False):
        """
        Computing the drift part of the SDE
        :param x_t:
        :param time:
        :param network:
        :Unet: Boolean, if True, set the input in the Unet format.
        :return:
        """
        #input = torch.concat([x_t, t], dim=-1)
        with torch.no_grad():
            if Unet:
                input = torch.reshape(x_t, (x_t.shape[0], 28, 28))
                input = input.to(dtype=torch.float32)
                approximate_expectation = network.forward(input[:, None, :, :], t[:, 0])
            else:
                batch_size = x_t.shape[0]
                t = t.repeat(batch_size, 1)
                input = torch.concat([x_t, t], dim=-1)
                approximate_expectation = network.forward(input)

        #plt.imshow(approximate_expectation[0, 0].detach().numpy(), cmap="gray")
        #plt.show()
        approximate_expectation = torch.reshape(approximate_expectation, (x_t.shape[0], self.dimension))
        drift = (approximate_expectation - x_t)/(self.int_sigma_sq_t(t0, tau) - self.int_sigma_sq_t(t0, t)) * self.sigma_t(t)**2
        return drift

    def euler_maruyama(self, x_0, times, tau, network):
        """

        :param x_0: torch.tensor(1, dim_process), starting point of the Euler-Maruyama scheme
        :param observed_data: torch.tensor(1, dim_data), observed data which defines the posterior distribution
        :param times: torch.tensor(N_times, 1), time discretization of the Eular-Maruyama scheme.
        :param tau: float, time horizon
        :param network: torch network, approximating the expectation
        :return: torch.tensor(1, dim_process), point approximately simulated according to the posterior distribution.
        """
        x_t = x_0
        t = torch.zeros((1,1), dtype=torch.float32)
        trajectories = [0]
        #trajectories.append(x_t.detach().numpy()[0, :])
        batch_size = x_0.shape[0]
        for i, t_new in enumerate(times):
            if i%10 == 0:
                print(i)

            drift = self.compute_drift_maruyama(x_t=x_t, t=t, tau=tau, network=network)
            ##Check transposition here
            x_t_new = x_t + drift * (t_new - t) + np.sqrt((t_new - t)) * torch.randn((batch_size, self.dimension))*self.sigma_t(t)
            #print(drift)
            #print(x_t_new)
            #print("\n")
            x_t = x_t_new
            #print(drift)
            #print(x_t_new)
            #print("\n\n")
            #trajectories.append(x_t.detach().numpy()[0, :])
            t = t_new


        print("done")
        return np.array(trajectories), x_t



if __name__=="__main__":
    bB = BrownianBridge(28*28)
    unet = torch.load("data/unet")
    times = torch.tensor(np.linspace(0, 1, 10000), dtype=torch.float32)[:, None, None]
    print(times.shape)
    traj, test = bB.euler_maruyama(torch.randn(1, 28*28),times[1:, :, :], 1, unet)
    np.save("trajectory.npy", traj)
    test = torch.reshape(test[0], (28, 28))
    test = torch.max(test, torch.zeros_like(test))
    test = torch.min(test, torch.ones_like(test))
    test = test/torch.max(test)
    print(test)
    plt.imshow(test.detach().numpy(), cmap="gray")
    plt.show()


class BrownianBridgeArbitrary:

    def __init__(self, dimension, a=3, b=4, T0=0, TN=1):
        """
        Brownian Bridge diffusion object, with drift sigma_t.
        :param dimension: integer, dimension of the state space
        :param sigma_t: real valued function taking time as input, drift function
        :param int_sigma_sq_t: real valued function, taking integral bounds as input, integrting the square of sigma
        :param T0: integer, starting time.
        :param TN: integer, ending time
        """

        self.a = a
        self.b = b
        self.dimension = dimension
        self.T0 = T0
        self.TN = TN

    def sigma_t(self, t):
        return self.a * np.exp(-self.b * t)

    def int_sigma_sq_t(self, t0, t):
        return - self.a ** 2 / (2 * self.b) * (np.exp(-2 * self.b * t) - np.exp(-2 * self.b * t0))

    def get_means(self, t, x0, xT, t0=0, T=1):
        """
        get the conditionnal mean of the Brownian bridge
        :param t: torch tensor(N_batch, 1), times at which we sample
        :param x0: torch tensor (N_batch, self.dimension), starting values of the bridge.
        :param xT: torch tensor (N_batch, self.dimension), ending values of the bridge.
        :param t0: torch tensor(N_batch, 1), times of the starting value
        :param T: torch tensor(N_batch, 1), times of the ending values
        :return: torch tensor (N_batch, self.dimension), conditional means at tme t
        """
        return (self.int_sigma_sq_t(t0, t) / self.int_sigma_sq_t(t0, T)) * (xT - x0) + x0

    def get_variances(self, t, t0=0, T=1):
        """
        get the conditionnal variance of the Brownian bridge, where the variance sigma_t is supposed to be proportional
        to the identity matric
        :param t: float, time at which we sample
        :param t0: float, time of the starting value
        :param T: float, time of the ending value
        :return: torch tensor (N_batch, 1), conditional variances at time t
        """

        return self.int_sigma_sq_t(t0, T) * (1 - self.int_sigma_sq_t(t0, t) / self.int_sigma_sq_t(t0, T))

    def sample_bridge(self, t, t_min, t_max, x_min, x_max):
        """

        :param t: time to sample
        :param t_min: initial time of the bridge
        :param t_max: final time of the bridge
        :param x_min: initial value of the bridge
        :param x_max: final value of the bridge
        :return: sample from the distribution at time t
        """
        int_t = self.int_sigma_sq_t(0, t)
        int_tmin = self.int_sigma_sq_t(0, t_min)
        int_tmax = self.int_sigma_sq_t(0, t_max)
        denom = int_tmin*int_tmax - int_tmin**2
        first_term = int_tmin*int_tmax - int_t*int_tmin
        second_term = int_tmin*int_t - int_tmin**2

        avg_0 = (self.int_sigma_sq_t(t_min, t)/self.int_sigma_sq_t(t_min, t_max))*(x_max - x_min) + x_min
        var_0 = self.int_sigma_sq_t(t_min, t)*(1- self.int_sigma_sq_t(t_min, t)/self.int_sigma_sq_t(t_min, t_max))

        avg = 1/denom * (first_term*x_min + second_term*x_max)
        var = int_t - 1/denom * (first_term*int_tmin + second_term*int_t)

        avg[t_min == 0] = avg_0[t_min == 0]
        var[t_min == 0] = var_0[t_min == 0]
        return torch.randn_like(avg) * torch.sqrt(var) + avg

    def compute_drift_maruyama(self, x_t, t, tau, network, t0=0, Unet=False):
        """
        Computing the drift part of the SDE
        :param x_t:
        :param time:
        :param network:
        :Unet: Boolean, if True, set the input in the Unet format.
        :return:
        """
        # input = torch.concat([x_t, t], dim=-1)
        with torch.no_grad():
            if Unet:
                input = torch.reshape(x_t, (x_t.shape[0], 28, 28))
                input = input.to(dtype=torch.float32)
                approximate_expectation = network.forward(input[:, None, :, :], t[:, 0])
            else:
                batch_size = x_t.shape[0]
                t = t.repeat(batch_size, 1)
                input = torch.concat([x_t, t], dim=-1)
                print("input SURE")
                print(input)
                approximate_expectation = network.forward(input)

        # plt.imshow(approximate_expectation[0, 0].detach().numpy(), cmap="gray")
        # plt.show()
        approximate_expectation = torch.reshape(approximate_expectation, (x_t.shape[0], self.dimension))
        drift = (approximate_expectation - x_t) / (
                    self.int_sigma_sq_t(t0, tau) - self.int_sigma_sq_t(t0, t)) * self.sigma_t(t) ** 2
        return drift

    def euler_maruyama(self, x_0, times, tau, network, t_0=torch.zeros((1, 1), dtype=torch.float32)):
        """

        :param x_0: torch.tensor(1, dim_process), starting point of the Euler-Maruyama scheme
        :param observed_data: torch.tensor(1, dim_data), observed data which defines the posterior distribution
        :param times: torch.tensor(N_times, 1), time discretization of the Eular-Maruyama scheme.
        :param tau: float, time horizon
        :param network: torch network, approximating the expectation
        :return: torch.tensor(1, dim_process), point approximately simulated according to the posterior distribution.
        """
        x_t = x_0
        t = t_0
        trajectories = [0]
        # trajectories.append(x_t.detach().numpy()[0, :])
        batch_size = x_0.shape[0]
        print("xt")
        print(x_t)
        for i, t_new in enumerate(times):
            if i % 10 == 0:
                print(i)

            print("AAAAA")
            print(i)
            drift = self.compute_drift_maruyama(x_t=x_t, t=t, tau=tau, network=network)
            ##Check transposition here
            x_t_new = x_t + drift * (t_new - t) + np.sqrt((t_new - t)) * torch.randn(
                (batch_size, self.dimension)) * self.sigma_t(t)
            # print(drift)
            # print(x_t_new)
            # print("\n")
            x_t = x_t_new
            print("x_t")
            print(drift)
            #if t_new == 1:
            #    print("OUT !")
            #    return np.array(trajectories), x_t
            # print(drift)
            # print(x_t_new)
            # print("\n\n")
            # trajectories.append(x_t.detach().numpy()[0, :])
            print("\n\n")
            t = t_new

        print("done")
        return np.array(trajectories), x_t
