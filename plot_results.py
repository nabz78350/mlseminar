import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
"""
def plot_data(df_orig, synth_data, starting_point, columns):
    for col in columns:
        values_orig = df_orig[col]
        values_imp = synth_data[col]
        values_orig = values_orig.cumsum()
        values_imp = values_imp.cumsum() + starting_point[col]
        fig, ax = plt.subplots(figsize=(10, 3))
        plt.plot(values_orig, color="black", label="original")
        plt.plot(values_imp, color="blue", label="Sampled")
        plt.ylabel(col, fontsize=10)
        plt.legend(loc=[1.01, 0], fontsize=10)
        plt.title(col)
        plt.show()
"""
import plotly.graph_objects as go

def plot_data(df_orig, synth_data, starting_point, columns):
    for col in columns:
        values_orig = df_orig[col]
        values_imp = synth_data[col]
        values_orig = values_orig.cumsum()
        values_imp = values_imp.cumsum() + starting_point[col]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_orig.index, y=values_orig, mode='lines', name='original'))
        fig.add_trace(go.Scatter(x=synth_data.index, y=values_imp, mode='lines', name='Sampled'))
        fig.update_layout(title=col, xaxis_title='Date', yaxis_title=col)
        fig.show()

        
def plot_covariance_matrices(cov_matrix_true, cov_matrix_synthetic, titles=['True Data', 'Synthetic Data']):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    sns.heatmap(cov_matrix_true, ax=axs[0], cmap='viridis', annot=True)
    axs[0].set_title(titles[0])

    sns.heatmap(cov_matrix_synthetic, ax=axs[1], cmap='viridis', annot=True)
    axs[1].set_title(titles[1])

    plt.show()


def plot_eigenvalues(eigenvalues_true, eigenvalues_synthetic):
    plt.figure(figsize=(10, 6))
    plt.plot(eigenvalues_true, label='True Data', marker='o')
    plt.plot(eigenvalues_synthetic, label='Synthetic Data', marker='x')
    plt.ylabel('Eigenvalue')
    plt.xlabel('Component')
    plt.title('Comparison of Eigenvalues')
    plt.legend()
    plt.grid(True)
    plt.show()

def pca_visualization(data_true, data_synthetic, n_components=2):
    from sklearn.decomposition import PCA
    pca_true = PCA(n_components=n_components).fit_transform(data_true)
    pca_synthetic = PCA(n_components=n_components).fit_transform(data_synthetic)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(pca_true[:, 0], pca_true[:, 1], alpha=0.5, label='True Data')
    plt.title('PCA of True Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(pca_synthetic[:, 0], pca_synthetic[:, 1], alpha=0.5, label='Synthetic Data', color='r')
    plt.title('PCA of Synthetic Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    plt.tight_layout()
    plt.show()



def plot_projection_on_principal_components(projection_true, projection_synthetic):
    plt.figure(figsize=(12, 6))
    
    # Assuming we're interested in the first two principal components for visualization
    plt.scatter(projection_true[:, 0], projection_true[:, 1], alpha=0.5, label='True Data', color='blue')
    plt.scatter(projection_synthetic[:, 0], projection_synthetic[:, 1], alpha=0.5, label='Synthetic Data', color='red')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('Projection of True and Synthetic Data on Principal Components')
    plt.show()

def plot_cumulative_variance(eigenvalues_true, eigenvalues_synthetic, title='Cumulative Variance Explained'):
    # Compute cumulative variance explained
    cumulative_variance_true = np.cumsum(eigenvalues_true) / np.sum(eigenvalues_true)
    cumulative_variance_synthetic = np.cumsum(eigenvalues_synthetic) / np.sum(eigenvalues_synthetic)

    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance_true, label='True Data', marker='o')
    plt.plot(cumulative_variance_synthetic, label='Synthetic Data', marker='x')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def biplot(score, loadings, labels=None, title="Biplot", points_label=None, axis_labels=('PC1', 'PC2')):
    plt.figure(figsize=(10, 7))
    xs = score[:,0]
    ys = score[:,1]
    n = loadings.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c='gray') # Plot points
    if points_label is not None:
        for i, txt in enumerate(points_label):
            plt.annotate(txt, (xs[i] * scalex, ys[i] * scaley), textcoords="offset points", xytext=(0,10), ha='center')
    for i in range(n):
        # Plot arrows
        plt.arrow(0, 0, loadings[i,0], loadings[i,1], color='r', alpha=0.5)
        if labels is not None:
            plt.text(loadings[i,0]* 1.15, loadings[i,1] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.grid(True)
    plt.show()