import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from qolmat.diffusion_model  import ImputerDiffusion
from qolmat.model  import TabDDPM, TsDDPM
from diffusion import DDPM
from load_data import prepare_data, aggregate_market_data
from models import CustomTransformerTimeSeries
from dataloader import TimeSeriesDataset
from models import AutoEncoder, ResidualBlockTS

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
START_TRAIN = "1999"
END_TRAIN = "2021"
START_TEST = "2022"
SCALER = 'QT'

data = aggregate_market_data()
df_reindexed, df_orig, df = prepare_data(data, from_year = START_TRAIN, start_year_test = START_TEST,scaler = SCALER)
train_df = df_reindexed.loc[:END_TRAIN]
train_df = train_df.interpolate(method='nearest')
X_train = train_df.to_numpy()
X_train = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)

N_FEAT = df_reindexed.shape[1] 

class ModelConfig:
    def __init__(self):
        self.TIMESTEPS = 50
        self.BETA1 = 1e-4
        self.BETA2 = 0.02
        self.SEQ_LEN = 21
        self.HIDDEN_DIM = 64
        self.BATCH_SIZE = 64
        self.N_EPOCH = 10
        self.LRATE = 1e-3
        self.N_FEAT = N_FEAT  # This will be set based on the data
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
        self.NUM_LAYERS_TRANSFORMER = 1
        self.NHEADS_FEATURE = 8
        self.NHEADS_TIME= 8

    def update_config(self, hyperparameter, value):
        if hasattr(self, hyperparameter):
            print(f'Changing {hyperparameter} from {getattr(self, hyperparameter)} to {value}')
            setattr(self, hyperparameter, value)
        else:
            print(f"Unknown hyperparameter: {hyperparameter}")
            
    def to_dict(self):
        # Convert all integer attributes to a dictionary
        config_dict = {attr: getattr(self, attr) for attr in dir(self) if isinstance(getattr(self, attr), int) or isinstance(getattr(self, attr), float) and not attr.startswith('__')  and not callable(getattr(self, attr))}
        return config_dict

def save_config_to_json(config, saved_dir, filename="config.json"):
    config_dict = config
    # Define the full path
    full_path = os.path.join(saved_dir, filename)
    with open(full_path, 'w') as json_file:
        json.dump(config_dict, json_file, indent=4)
    print(f"Configuration saved to {full_path}")



def train_model(model_kind="DDPM",hyperparameter ="SEQ_LEN",parameter_value=21):
    
    model_config = ModelConfig()
    model_config.update_config(hyperparameter,parameter_value)
    model = AutoEncoder(num_noise_steps =  model_config.TIMESTEPS ,
                        dim_input = model_config.N_FEAT,
                        residual_block = ResidualBlockTS(model_config.HIDDEN_DIM,
                                                         model_config.SEQ_LEN,
                                                         model_config.HIDDEN_DIM,
                                                         num_layers_transformer=model_config.NUM_LAYERS_TRANSFORMER,
                                                         nheads_feature=model_config.NHEADS_FEATURE,
                                                         nheads_time=model_config.NHEADS_TIME),
                        dim_embedding = model_config.HIDDEN_DIM)
    train_dataset = TimeSeriesDataset(X_train, seq_len=model_config.SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size = model_config.BATCH_SIZE, shuffle = False)
    optim = torch.optim.Adam(model.parameters(), lr=model_config.LRATE)
    
    save_dir = 'results/'+model_kind + '/'+ hyperparameter+'/'+str(parameter_value)+'/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_config_to_json(model_config.to_dict(),save_dir)
    
    ddpm = DDPM(model = model, 
            optimizer = optim,
            device = model_config.DEVICE, 
            timesteps = model_config.TIMESTEPS, 
            beta1 = model_config.BETA1, 
            beta2 = model_config.BETA2, 
            n_epoch = model_config.N_EPOCH, 
            batch_size = model_config.BATCH_SIZE, 
            lrate = model_config.LRATE, 
            save_dir = save_dir)
    losses, maes, wasserstein_distances =ddpm.train(train_loader=train_loader)
    fit_history = pd.DataFrame([losses,maes,wasserstein_distances])
    fit_history.to_csv(os.path.join(save_dir,'history.csv'))
    gen_samples, _  = ddpm.sample(n_sample = 1500, window_size = model_config.SEQ_LEN, dim_input = model_config.N_FEAT, save_rate=20)
    np.save(os.path.join(save_dir,'samples.npy'),gen_samples.numpy())
    
    

if __name__ == '__main__':
    hyperparameter = "SEQ_LEN"
    values = [5,10,21,63,126,252]
    for val in values :
        train_model(model_kind="DDPM",hyperparameter =hyperparameter,parameter_value=val)
    
    # hyperparameter = "BATCH_SIZE"
    # values = [32,64,128]
    # for val in values :
    #     train_model(model_kind="DDPM",hyperparameter =hyperparameter,parameter_value=val)
        
    # hyperparameter = "LRATE"
    # values = [1e-4,1e-3,1e-2,1e-1]
    # for val in values :
    #     train_model(model_kind="DDPM",hyperparameter =hyperparameter,parameter_value=val)
    
    # hyperparameter = "NUM_LAYERS_TRANSFORMER"
    # values = [1,2,4,8]
    # for val in values :
    #     train_model(model_kind="DDPM",hyperparameter =hyperparameter,parameter_value=val)
    
    