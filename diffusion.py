
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm import tqdm
import os
import numpy as np
from scipy.stats import wasserstein_distance


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
                # perturb data with gaussian noise
                noise = torch.randn_like(x)
                print(noise.shape)
                
                ### TODO ajouter bruit gaussian covarié
                values = x.numpy()
                batch_size = values.shape[0]
                seq_len = values.shape[1] 
                n_columns = values.shape[2]
                values_reshaped = values.reshape(-1, values.shape[-1])
                cov_matrix = np.cov(values_reshaped, rowvar=False)
                L = np.linalg.cholesky(cov_matrix)
                standard_normal_samples = np.random.normal(loc=0, scale=1, size=(batch_size, seq_len, n_columns))
                transformed_noise = np.einsum('ijk,kl->ijl', standard_normal_samples, L)
                noise = torch.tensor(transformed_noise)
                print(noise.shape)
                t = torch.randint(1, self.timesteps, (x.shape[0],1)).to(self.device) 
                x_pert = self.perturb_input(x, t.squeeze(), noise)
                # use network to recover noise
                pred_noise = self.model(x_pert, t )
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


