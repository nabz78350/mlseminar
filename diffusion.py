
import torch
from torch import nn
from torch.nn import functional as F
# from torch.autograd import Variable
from tqdm import tqdm
import os
import numpy as np
from scipy.stats import wasserstein_distance
from utils import VPSDE, ReverseDiffusionPredictor, LangevinCorrector, pc_sampler


class DDPM(nn.Module):
    def __init__(self, model, optimizer, device, timesteps, beta1, beta2, n_epoch, batch_size, lrate, save_dir):
        super(DDPM, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.timesteps = timesteps
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lrate = lrate
        self.save_dir = save_dir

        # construct DDPM noise schedule
        self.b_t = (self.beta2 - self.beta1) * torch.linspace(0, 1, self.timesteps + 1, device=self.device) + self.beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()    
        self.ab_t[0] = 1

    def perturb_input(self, x, t, noise):
        return self.ab_t.sqrt()[t, None, None] * x + (1 - self.ab_t[t, None, None]) * noise

    def denoise_add_noise(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        return mean + noise

    def train(self, train_loader):
        # Initialize lists to store metrics
        losses = []
        maes = []
        wasserstein_distances = []

        for ep in range(self.n_epoch):
            print(f'epoch {ep}')
            loss_epoch = 0
            mae_epoch = 0
            w_dist_epoch = 0
            # linearly decay learning rate
            pbar = tqdm(train_loader, mininterval=2 )    
            for x in pbar:   
                self.model.zero_grad()
                x = x.to(self.device)
                # perturb data
                noise = torch.randn_like(x)
                ### TODO ajouter bruit gaussian covarié
                t = torch.randint(1, self.timesteps, (x.shape[0],1)).to(self.device) 
                x_pert = self.perturb_input(x, t.squeeze(), noise)
                # use network to recover noise
                pred_noise = self.model(x_pert, t)
                # loss is mean squared error between the predicted and true noise
                loss = F.mse_loss(pred_noise, noise)
                loss_epoch += loss.item()/self.batch_size
                loss.backward()
                self.optimizer.step()
                # Calculate MAE and Wasserstein distance
                mae_epoch += F.l1_loss(pred_noise, noise).item()/self.batch_size
                w_dist_epoch += wasserstein_distance(pred_noise.flatten().detach().cpu().numpy(), noise.flatten().cpu().numpy())/self.batch_size
                # Store metrics
            losses.append(loss_epoch)
            maes.append(mae_epoch)
            wasserstein_distances.append(w_dist_epoch)
            # save model periodically
            if ep%4==0 or ep == int(self.n_epoch-1):
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                print(f'Loss: {loss_epoch}, MAE: {mae_epoch}, Wasserstein Distance: {w_dist_epoch}')
        torch.save(self.model.state_dict(), self.save_dir + f"model_final.pth")
        print('saved model at ' + self.save_dir + f"model_final.pth")
        return losses, maes, wasserstein_distances

    @torch.no_grad()
    def sample(self, n_sample, window_size, dim_input, save_rate=20):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, window_size, dim_input).to(self.device) 
        # array to keep track of generated steps for plotting
        intermediate = [] 
        for i in range(self.timesteps-1, -1, -1):
            print(f'sampling timestep {i:3d}', end='\r')
            # reshape time tensor
            t = torch.tensor([i]).to(self.device)
            t = t.repeat(n_sample,1)
            z = torch.randn_like(samples)  if i > 1 else 0
            eps = self.model(samples,t)    # predict noise e_(x_t,t)
            samples = self.denoise_add_noise(samples, i+1, eps, z)

            if i % save_rate ==0 or i==self.timesteps or i<8:
                intermediate.append(samples.detach().cpu().numpy())
        intermediate = np.stack(intermediate)
        return samples, intermediate


class TSGM(nn.Module):
    def __init__(self, score_model, RNN_model, num_noise_steps, beta_0, beta_1,
                 n_epoch_train, n_epoch_pretrain, lrate, device):
        super(TSGM, self).__init__()
        self.device = device
        self.score_model = score_model
        self.RNN = RNN_model.to(self.device)
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.n_epoch_train = n_epoch_train
        self.n_epoch_pretrain = n_epoch_pretrain
        self.num_noise_steps = num_noise_steps
        self.lrate = lrate

        self.sde = VPSDE(beta_0, beta_1, num_noise_steps)

    def pretrain_RNN(self, train_loader):
        optimizer = torch.optim.Adam(self.RNN.parameters(), lr=self.lrate)

        # initializations
        losses = []
        for epoch in range(self.n_epoch_pretrain):
            loss_epoch = 0
            for x in train_loader:
                x = x.to(self.device)
                optimizer.zero_grad()

                x_pred = self.RNN(x)
                loss = F.mse_loss(x_pred, x)

                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
            losses.append(loss_epoch/len(train_loader))
            print(f'Epoch {epoch+1}/{self.n_epoch_pretrain}, Loss: {loss_epoch/len(train_loader)}')
        return losses

    def loss_score(self, x):

        # RNN encoding
        h = self.RNN.encoder(x)[0]  # h/X.shape = (batch_size, window_size, n_features)

        # extract h_prev
        h_prev = h[:, :-1, :]
        h = h[:, 1:, :]

        # forward pass
        s = torch.randint(1, self.num_noise_steps, (h.shape[0], 1)).to(self.device)
        z = torch.randn_like(h)  # sample noise
        # h_s = self.sde.sqrt_alphas_cumprod[s, None] * h + self.sde.sqrt_1m_alphas_cumprod[s, None] * z

        mean, std = self.sde.marginal_prob(h, s)
        h_s = mean + std[:, None] * z

        # concatenate h_s and h_prev
        h_combined = torch.cat((h_s, h_prev), dim=2)

        # compute score
        score = self.score_model(h_combined, s)

        # compute gradient
        # grad = (h_s - h)/(1 - self.sde.alphas_cumprod[s, None])

        # compute loss
        loss = torch.square(score * std[:, None] + z)

        # return F.mse_loss(score, grad)
        return torch.mean(loss)

    def train(self, train_loader, use_alt=False):
        # pre-train the RNN model
        self.pretrain_RNN(train_loader)

        # set up optimizer
        if use_alt:
            self.parameters = list(self.score_model.parameters()) +\
                list(self.RNN.parameters())
        else:
            self.parameters = list(self.score_model.parameters())

        self.optimizer = torch.optim.Adam(self.parameters, lr=self.lrate)

        # initializations
        losses = []
        # maes = []
        # wasserstein_distances = []

        for epoch in range(self.n_epoch_train):
            pbar = tqdm(train_loader, mininterval=2)
            loss_epoch = 0
            # mae_epoch = 0
            # w_dist_epoch = 0
            for x in pbar:
                x = x.to(self.device)
                loss = self.loss_score(x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if use_alt:
                    self.optimizer.zero_grad()
                    x_pred = self.RNN(x)
                    loss_RNN = F.mse_loss(x_pred, x)
                    loss_RNN.backward()
                    self.optimizer.step()

                loss_epoch += loss.item()*x.shape[0]
                # mae_epoch += F.l1_loss(pred_noise, noise).item()/x.shape[0]
                # w_dist_epoch += wasserstein_distance(pred_noise.flatten().detach().cpu().numpy(), noise.flatten().cpu().numpy())/x.shape[0]

            loss_epoch = loss_epoch/len(train_loader)
            losses.append(loss_epoch)
            # maes.append(mae_epoch)
            # wasserstein_distances.append(w_dist_epoch)

            print(f'Epoch {epoch+1}/{self.n_epoch_train}, Loss: {loss_epoch}')

    def sample(self, start, n_sample, window_size, dim_input):
        predictor = ReverseDiffusionPredictor
        corrector = LangevinCorrector
        snr = 0.16
        n_steps = 1
        sde = VPSDE(self.beta_0, self.beta_1, self.num_noise_steps)
        shape = (n_sample, window_size, dim_input)
        h = pc_sampler(start,  # shape: (batch_size, dim_input)
                          self.score_model, sde,
                          shape,
                          predictor, corrector,
                          snr, self.device, n_steps=n_steps)
        x = self.RNN.decoder(h)[0]
        return x
