import abc
import torch
import numpy as np


def pc_sampler(start, model, sde, shape, predictor, corrector, snr, device, inverse_scaler=lambda x: x, continuous=False,
                n_steps=1, denoise=False, eps=1e-3, probability_flow=False):
    """ The PC sampler funciton.
    """
    model.eval()

    def score_fn(x, t):
        # Scale neural network output by standard deviation and flip sign
        if continuous:
            # For VP-trained models, t=0 corresponds to the lowest noise level
            # The maximum value of time embedding is assumed to 999 for
            # continuously-trained models.
            labels = t * 999
            score = model(x, labels)
            std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
            # For VP-trained models, t=0 corresponds to the lowest noise level
            labels = t * (sde.N - 1)
            labels = labels.long()
            score = model(x, labels)
            std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

        score = -score / std[:, None, None]
        return score

    with torch.no_grad():
        window_size = shape[1]
        shape_one_time = (shape[0], 1, shape[2])

        sample = torch.zeros(shape, device=device)
        x_prev = start  # shape: (batch_size, 1, n_features)
        sample[:, 0, :] = start.squeeze(1)
        for time_step_x in range(1, window_size):
            # Initial sample
            x = sde.prior_sampling(shape_one_time).to(device)
            x = torch.cat((x, x_prev), dim=-1)

            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            # Run the predictor-corrector algorithm
            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = predictor(sde, score_fn, probability_flow).update_fn(x, vec_t)
                x, x_mean = corrector(sde, score_fn, snr, n_steps).update_fn(x, vec_t)

            x, _ = inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)
            x = x[:, :, :x.shape[-1] // 2]  # Remove the previous state
            sample[:, time_step_x, :] = x.squeeze(1)
            x_prev = x

    return sample


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
        N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
        z: latent code
        Returns:
        log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
        x: a torch tensor
        t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
        f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
        score_fn: A time-dependent score-based model that takes x and t and returns the score.
        probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):

            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                n_feat = int(x.shape[-1]/2)
                x_curr = x[:, :, :n_feat]
                f, G = discretize_fn(x_curr, t)
                rev_f = f - G[:, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, :, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None] * x - x
        G = sqrt_beta
        return f, G


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

        Args:
        x: A PyTorch tensor representing the current state
        t: A Pytorch tensor representing the current time step.

        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

        Args:
        x: A PyTorch tensor representing the current state
        t: A PyTorch tensor representing the current time step.

        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


# @register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
        return x, x_mean


# @register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        n_feat = int(x.shape[-1]/2)  # *
        x_curr = x[:, :, :n_feat]  # *
        x_prev = x[:, :, n_feat:]  # *
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x_curr)  # *
        x_mean = x_curr - f  # *
        x_curr = x_mean + G[:, None, None] * z

        return torch.cat((x_curr, x_prev), dim=-1), torch.cat((x_mean, x_prev), dim=-1)  # *


# @register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        # if not isinstance(sde, VPSDE):
            # and not isinstance(sde, sde_lib.VESDE) \
            # and not isinstance(sde, sde_lib.subVPSDE):
            # raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        n_feat = int(x.shape[-1]/2)  # *
        x_curr = x[:, :, :n_feat]  # *
        x_prev = x[:, :, n_feat:]  # *
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE):  # or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x_curr)  # *
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x_curr + step_size[:, None, None] * grad  # *
            x_curr = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise  # *

        return torch.cat((x_curr, x_prev), dim=-1), torch.cat((x_mean, x_prev), dim=-1)  # *


# @register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        n_feat = int(x.shape[-1]/2)
        x_curr = x[:, :, :n_feat]

        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        # print(x.shape, t.shape)
        score = self.score_fn(x, t)
        x_mean = (x_curr + beta[:, None, None] * score) / torch.sqrt(1. - beta)[:, None, None]
        noise = torch.randn_like(x_curr)
        x_curr = x_mean + torch.sqrt(beta)[:, None, None] * noise
        x[:, :, :n_feat] = x_curr
        x_mean = torch.cat((x_curr, x[:, :, n_feat:]), dim=-1)
        return x, x_mean

    def update_fn(self, x, t):
        return self.vpsde_update_fn(x, t)


# # @register_predictor(name='none')
# class NonePredictor(Predictor):
#     """An empty predictor that does nothing."""

#     def __init__(self, sde, score_fn, probability_flow=False):
#         pass

#     def update_fn(self, x, t):
#         return x, x


# # @register_corrector(name='ald')
# class AnnealedLangevinDynamics(Corrector):
#     """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

#     We include this corrector only for completeness. It was not directly used in our paper.
#     """

#     def __init__(self, sde, score_fn, snr, n_steps):
#         super().__init__(sde, score_fn, snr, n_steps)
#         if not isinstance(sde, VPSDE):  # \
#             # and not isinstance(sde, sde_lib.VESDE) \
#             # and not isinstance(sde, sde_lib.subVPSDE):
#             raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

#     def update_fn(self, x, t):
#         sde = self.sde
#         score_fn = self.score_fn
#         n_steps = self.n_steps
#         target_snr = self.snr
#         if isinstance(sde, VPSDE):  # or isinstance(sde, sde_lib.subVPSDE):
#             timestep = (t * (sde.N - 1) / sde.T).long()
#             alpha = sde.alphas.to(t.device)[timestep]
#         else:
#             alpha = torch.ones_like(t)

#         std = self.sde.marginal_prob(x, t)[1]

#         for i in range(n_steps):
#             grad = score_fn(x, t)
#             noise = torch.randn_like(x)
#             step_size = (target_snr * std) ** 2 * 2 * alpha
#             x_mean = x + step_size[:, None, None] * grad
#             x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None]

#         return x, x_mean


# @register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x