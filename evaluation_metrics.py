import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_size, num_heads, dropout_prob, task='classification'):
        super(TransformerModel, self).__init__()
        self.task = task
        self.fc = nn.Linear(input_dim, hidden_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout_prob)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_dim, 1)
        if self.task == 'classification':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

    def forward(self, x):
        x  = self.fc(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        if self.task == 'classification':
            x = F.log_softmax(x, dim=-1)
        return x

    def train_model(self, train_data, optimizer, num_epochs):
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(train_data):
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def predict(self, test_data):
        predictions = []
        for i, inputs in enumerate(test_data):
            outputs = self.forward(inputs)
            predictions.append(outputs)
        return predictions

# Main function to implement LDS
def calculate_LDS(model, synthetic_data, original_data):
    # create a dataset from the synthetic data labeled as 0 and original data labeled as 1
    data = torch.cat((synthetic_data, original_data))
    labels = torch.cat((torch.zeros(synthetic_data.shape[0]), torch.ones(original_data.shape[0]))).numpy()
    predictions = model.predict(data).numpy()
    y_pred = predictions > 0.5
    return np.mean(y_pred != labels)

"""
# Main function to implement LPS
def calculate_LPS(synthetic_data):
    # Prepare synthetic data
    # Initialize Transformer-based model
    # Train the model using synthetic data
    # Evaluate the model on synthetic data
    # Return LPS score
"""

def row_fidelity(synthetic_df, true_df):
    # Assuming a simple row fidelity calculation: Mean absolute difference per row
    return np.mean(np.abs(synthetic_df - true_df), axis=1)

def column_fidelity(synthetic_df, true_df):
    # Assuming column fidelity as the comparison of mean values per column
    fidelity_scores = {}
    for col in synthetic_df.columns:
        fidelity_scores[col] = np.mean(np.abs(synthetic_df[col] - true_df[col]))
    return fidelity_scores

def kl_divergence_columns(synthetic_df, true_df, epsilon=1e-12):
    """
    Compute the KL divergence between each corresponding column of the synthetic and true dataframes.
    epsilon is used to avoid division by zero.
    """
    kl_scores = pd.DataFrame(index = synthetic_df.columns.tolist(),columns = ['KL divergence col'])
    for col in synthetic_df.columns:
        # Ensuring the data is in the form of a distribution
        synthetic_dist = np.histogram(synthetic_df[col], bins=10, density=True)[0]
        true_dist = np.histogram(true_df[col], bins=10, density=True)[0]
        
        # Adding epsilon to avoid division by zero or log of zero
        synthetic_dist += epsilon
        true_dist += epsilon
        
        kl_div = entropy(true_dist, synthetic_dist)
        kl_scores.loc[col,'KL divergence col'] = kl_div
        
    return kl_scores


def kl_divergence_rows(synthetic_df, true_df, epsilon=1e-12):
    """
    Compute the KL divergence between each corresponding column of the synthetic and true dataframes.
    epsilon is used to avoid division by zero.
    """
    kl_scores = pd.DataFrame(index = synthetic_df.index.tolist(),columns = ['KL divergence rows'])
    for date in synthetic_df.index.tolist():
        # Ensuring the data is in the form of a distribution
        synthetic_dist = np.histogram(synthetic_df.loc[date], bins=10, density=True)[0]
        true_dist = np.histogram(true_df.loc[date], bins=10, density=True)[0]
        
        # Adding epsilon to avoid division by zero or log of zero
        synthetic_dist += epsilon
        true_dist += epsilon
        
        kl_div = entropy(true_dist, synthetic_dist)
        kl_scores.loc[date,'KL divergence rows'] = kl_div
        
    return kl_scores

def wasserstein_distance_columns(synthetic_df, true_df):
    """
    Compute the Wasserstein distance between each corresponding column of the synthetic and true dataframes.
    """
    wasserstein_scores = pd.DataFrame(index = synthetic_df.columns.tolist(),columns = ['Wassertstein distance col'])
    for col in synthetic_df.columns.tolist():
        ws_dist = wasserstein_distance(synthetic_df[col],true_df[col])
        wasserstein_scores.loc[col,'Wassertstein distance col'] = ws_dist
        
    return wasserstein_scores

def wasserstein_distance_rows(synthetic_df, true_df):
    """
    Compute the Wasserstein distance between each corresponding column of the synthetic and true dataframes.
    """
    wasserstein_scores = pd.DataFrame(index = synthetic_df.index.tolist(),columns = ['Wassertstein distance rows'])
    for date in synthetic_df.index.tolist():
        ws_dist = wasserstein_distance(synthetic_df.loc[date],true_df.loc[date])
        wasserstein_scores.loc[date,'Wassertstein distance rows'] = ws_dist        
    return wasserstein_scores

def compute_frobenius_norm(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2, 'fro')

def compute_spectral_norm(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2, 2)

def compute_condition_number(matrix):
    return np.linalg.cond(matrix)

def eigen_decomposition(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort the eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    return eigenvalues, eigenvectors

def compute_principal_components(data):
    # Center the data
    data_meaned = data - np.mean(data, axis=0)
    # Compute covariance matrix
    covariance_matrix = np.cov(data_meaned, rowvar=False)
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    return eigenvalues, eigenvectors

def project_onto_principal_components(data, eigenvectors):
    # Center the data
    data_meaned = data - np.mean(data, axis=0)
    # Project data
    return np.dot(data_meaned, eigenvectors)