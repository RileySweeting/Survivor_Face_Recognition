# face_recognition.py
# Authors: Sierra Doyle & Riley Sweeting
# Description: A facial recognition system implemented using Scikit-learn principal component analysis. Classifies and compares faces to those of 839 contestants from the TV show 'Survivor' using k-nearest-neighbors and k-means-clustering algorithms.

# Import packages
import argparse
import os
import pdb
import numpy as np
import pandas as pd # Importing rankings
from sklearn.decomposition import PCA # PCA method
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread # Method to read image data
from skimage.util import montage # Condense images
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score # Evaluate effectiveness of k means clusters 

# Data directories
ROOT = os.path.dirname(os.path.abspath(__file__)) # path to source directory of this file
SURVIVOR_DIR = os.path.join(ROOT, "data", "survivor") # path to source directory of survivor images
PROF_DIR = os.path.join(ROOT, "data", "professors") # path to source directory of professor images

# Global variables
HEIGHT = 70
WIDTH = 70

# Command line parser
parser = argparse.ArgumentParser(description="Apply unsupervised learning methods to the problem of face recognition")
parser.add_argument('--debug', help='use pdb to look into code before exiting program', action='store_true')
parser.add_argument('--variance', default=0.90, type=float, help='Specified variance to keep')
parser.add_argument('--verbose', action='store_true', help='Select whether to display plots')

# Main method
def main(args):
    # Header
    print(f'============================')
    print(f'PRINCIPAL COMPONENT ANALYSIS')
    print(f'============================')
    print('Loading data...')
    
    # Load survivor and professor image data
    survivors = load(SURVIVOR_DIR, True)
    professors = load(PROF_DIR, False)
    
    # Load survivor rankings
    rankings = pd.read_csv(os.path.join(ROOT, 'data', 'rankings.csv'), header=None).to_numpy()
    rankings = rankings[np.argsort(rankings[:, 0])]
    
    # Useful variables
    data = survivors[0] # Shape (HEIGHT * WIDTH * 3, num_survivors)
    prof_data = professors[0] # Shape (HEIGHT * WIDTH * 3, num_professors)
    survivor_names = survivors[1]
    professor_names = professors[1]
    
    # Compute mean image    
    mean_image = data.mean(axis=1)
    
    # Plot mean image
    if args.verbose:
        plt.figure(num='Figure 1')
        plt.imshow(mean_image.reshape(HEIGHT, WIDTH, 3))
        plt.title("Mean Face")
        plt.axis('off')
        plt.show(block=False)
        
    # Display progress
    print('\nCalculating principal components...')
    
    # Initialize PCA model with specified variance 
    pca = PCA(n_components=args.variance)
    
    # Display progress
    print('\nTransforming images into low dimensional face space...')
        
    # Extract survivor weights and components/eigenvectors (PC's) from trained PCA model    
    s_weights = pca.fit_transform(data.T)
    components = pca.components_
    m = len(components) # 'm' number of components kept for specified variance 
    
    # Calculate professor weights by transforming professors into 'face space'
    p_weights = pca.transform(prof_data.T)
    
    # Reconstruct original images
    reconstructed = reconstruct(p_weights, pca)
    
    # Plot reconstructed images
    if args.verbose:
        visualize(reconstructed, professor_names, m)
    
    # Determine face of maximum euclidean distance between given original and reconstructed images   
    idx_max, dist_max = max_euclidean_dist(prof_data.T, reconstructed, professor_names)
    
    # Print results
    print(f'\n============================================')
    print(f'Which professor looks the least like a face?')
    print(f'============================================')
    print(f'Professor {professor_names[idx_max]} looks the least like a face with a distance of {dist_max:.2f}')
    
    # Using 'Nearest Neighbors' algorithm, determine which given face is most similar to the host Jeff
    idx_min, dist_min = nearest_neighbor(s_weights[0], p_weights, 1)
    
    # Print results
    print(f'\n=====================================================')
    print(f'Which professor looks the most like host Jeff Probst?')
    print(f'=====================================================')
    print(f'Professor {professor_names[idx_min]} looks the most like Survivor Host Jeff Probst with a minimum distance of {dist_min:.2f}')
    
    # Using 'Nearest Neighbors' algorithm, determine which given face is most similar to Ben Ruby
    idx_b, dist_b = nearest_neighbor(p_weights[0], s_weights, 1)
    
    # Print results
    print(f'\n============================================')
    print(f'Which survivor looks the most like Ben Ruby?')
    print(f'============================================')
    print(f'Survivor {survivor_names[idx_b]} looks the most like Ben Ruby with a minimum distance of {dist_b:.2f}')
    
    # Using 'K Means Clustering', given sample faces, determine which season they belong in by averaging seasons of idividuals per cluster    
    seasons = likely_season(s_weights, p_weights, survivor_names, pca, args)
    
    # Print results
    print(f'\n===========================================')
    print(f'Which season does each professor belong on?')
    print(f'===========================================')
    for i in range(len(seasons)):
        print(f'Professor {professor_names[i]} is predicted to be in season {seasons[i]}')
    
    # Using K Nearest Neighbors, average the rankings of the sample images nearest neighbors
    idx, rank = winner(s_weights[1:], p_weights, professor_names, rankings[:, 1], 9, pca, args)
    
    # Display winner information
    print(f'\n=============================================')
    print(f'Which professor is predicted to win survivor?')
    print(f'=============================================')
    print(f'Professor {professor_names[idx]} is the predicted winner with an average ranking of {rank:.2f}')
    
    # Set PDB trace if debug argument is selected
    if args.debug:
        pdb.set_trace()
        
def reconstruct(weights, pca):
    '''
    Reconstructs original images given eigenface weights of reduced images
    
    Parameters:
        weights: Eigenface weights of images transformed into face space
        pca: Trained pca model
    
    Returns:
        Reconstructed images of original image dimensions
    '''
    # Reconstruct original images and reshape to image shape (num_samples, HEIGHT, WIDTH, 3)
    reconstructed_images = pca.inverse_transform(weights).reshape(len(weights), HEIGHT, WIDTH, 3)
    
    # Clip RGB values to [0, 1] to eliminate unwanted messages
    reconstructed_images = np.clip(reconstructed_images, 0, 1)
    
    return reconstructed_images
        
def visualize(images, names, m):
    '''
    Plots reconstructed images with image titles using Matplotlib
    
    Parameters:
        images: Reconstructed images of original image dimensions
        names: Array of image titles/names
    
    Returns:
        None
    '''  
    # Set up the plot grid (2 rows, 3 columns in this case)
    fig = plt.figure(num='Figure 2', figsize=(8,8))
    axes = fig.subplots(2, 3)

    # Flatten axes to make indexing easier & hide axes
    axes = axes.flatten()
    for axis in axes:
        axis.axis('off')

    # Loop through images and titles and add to the plot
    for i in range(len(images)):
        axis = axes[i]
        axis.imshow(images[i])
        axis.set_title(names[i])
        
    # Show plot
    plt.suptitle(f'Reconstructed Professor Faces using {m} Principal Components')
    plt.show(block=False)

def max_euclidean_dist(orig, rec, names):
    '''
    Calculates the minimum euclidean distance between original and reconstructed images
    
    Parameters:
        orig: Original images
        rec: Reconstructed images
        names: Array of image titles/names
    
    Returns:
        idx: Index of image with maximum euclidean distance
        dist: Maximum distance
    '''  
    # Flatten the images
    v1 = rec.reshape(len(rec), HEIGHT * WIDTH * 3)
    v2 = orig.reshape(len(orig), HEIGHT * WIDTH * 3)
    
    # Compute the euclidean distances
    dists = np.sqrt(np.sum(np.power(v2 - v1, 2), axis=1))
    
    # Maximum distance
    dist = np.max(dists)
    
    # Index of maximum distance
    idx = np.where(dists == dist)[0][0]
    
    # Return index of professor with max distance, and max distance
    return idx, dist

def nearest_neighbor(reference, data, num_neighbors):
    '''
    Given a reference image weights and dataset, determines the closest neighboring image(s)
    
    Parameters:
        reference: Reference image from which the neighbors are computed
        data: Eigenface weights of the sample image data
        num_neighbors: Number of neighbors to compute
        names: Array of image titles/names
    
    Returns:
        idx: Index of image with minimum distance from reference image
        dist: Distance of closest image
    ''' 
    # Fit the nearest neighbors model to the data with specified number of neighbors
    model = NearestNeighbors(n_neighbors=num_neighbors).fit(data)
    
    # Compute nearest neighbor(s)
    dist, idx = model.kneighbors(reference.reshape(1, len(reference)))
    
    # Return results
    return idx[0][0], dist[0][0]

def likely_season(data, professors, survivor_names, pca, args):
    '''
    Performs k means clustering on the survivor image dataset. Assigns a given professor to a cluster
    
    Parameters:
        data: Eigenface weights of the sample image data
        professors: Eigenface weights of professor image data
        survivor_names: Array of sample images titles/names
        pca: Trained pca model
        args: Arguments from argparser
    
    Returns:
        seasons: Array of predicted seasons for each professor
    ''' 
    # Initialize seed for repeatability
    np.random.seed(100)
    
    # Fit the k means clustering model to the survivor data with specified number of clusters
    model = KMeans(n_clusters=20, init='k-means++', n_init="auto").fit(data)
    
    # Predict cluster of each professor
    labels = model.predict(professors)
    
    # Seasons corresponding to professors
    seasons = []
    
    # For each professor
    for label in labels:
    
        # Get indices for all survivor samples with corresponding label/cluster
        indices = np.where(model.labels_ == label)[0]
        
        # Retrieve seasons (as integers)
        sns = np.array([int(s[0]) for s in survivor_names])
        
        # Gather relevent survivors seasons
        survivors = sns[indices]
        
        # Determine index of mode season
        mode_idx = np.argmax(np.bincount(survivors)) + 1 # Account for 0 indexing
        
        # Append to results array
        seasons.append(mode_idx)
    
    # Visualize contestants per cluster as montage subplots
    if args.verbose:
        # Set up the plot grid (4 rows, 5 columns in this case)
        fig = plt.figure(num='Figure 3', figsize=(8,8))
        axes = fig.subplots(4, 5)

        # Flatten axes to make indexing easier & hide axes
        axes = axes.flatten()
        for axis in axes:
            axis.axis('off')
            
        # Montage images per cluster
        for idx in range(model.n_clusters):
            # Select survivor images of current cluster
            cluster_data = data[np.where(model.labels_ == idx)[0]]
            
            # Reconstruct survivor images
            cluster = reconstruct(cluster_data[:9], pca)
    
            # Clip RGB values to [0, 1] to eliminate unwanted messages
            cluster = np.clip(cluster, 0, 1)
            
            # Montage images
            montaged = montage(cluster, channel_axis=3)
            
            # Clip RGB values to [0, 1] to eliminate unwanted messages
            montaged = np.clip(montaged, 0, 1)
            
            # Plot montage in subplot
            axis = axes[idx]
            axis.imshow(montaged)
            axis.set_title(f'Cluster {idx + 1}')
            
        # Show plot
        plt.suptitle(f'9 Survivors from {model.n_clusters} K-Means Clusters')
        plt.show(block=False)
        
    # Return results
    return seasons
        
def winner(data, samples, names, rankings, k, pca, args):
    '''
    Using K-Nearest Neighbors, calculates k nearest neighbors per given sample image and averages rankings of neighbors
    
    Parameters:
        data: Eigenface weights of survivor image data to train knn model
        samples: Eigenface weights of professor image data to calculate neighbors of
        names: Array of professor images titles/names
        rankings: Array of average neighbor rankings for each professor
        k: Number of neighbors to compute
        pca: Traine pca model
        args: Arguments from argparser
    
    Returns:
        idx: Index of professor with minimum average ranking
        min_avg: Minimum average
    ''' 
    # Fit the nearest neighbors model to the data with specified number of neighbors
    model = NearestNeighbors(n_neighbors=k).fit(data)
    
    # Indices of k nearest neighbors per sample
    dist, indices = model.kneighbors(samples)
    
    # Average rankings
    averages = []
    
    # Reconstruct sample images
    sample_imgs = reconstruct(samples, pca)
    
    # Set up the plot grid (5 rows, 2 columns in this case)
    fig = plt.figure(num='Figure 4', figsize=(9,9))
    axes = fig.subplots(len(samples), 2)
    
    # Compute nearest neighbors per sample
    for idx in range(len(samples)):       
        # Determine rankings of neighbors
        ranks = rankings[indices[idx]]

        # Average the rankings
        averages.append(np.mean(ranks))
        
        # Gather neighbor weights
        weights = data[indices[idx]]
        
        # Reconstruct images
        images = reconstruct(weights, pca)
        
        # Montage images
        montaged = montage(images, channel_axis=3)
        
        # Plot montage in subplot column 1
        axis = axes[idx][1]
        axis.imshow(montaged)
        axis.set_title(f'Nearest {k} Neighbors')
        axis.axis('off')
        
        # Plot professors in subplot column 0
        axis = axes[idx][0]
        axis.imshow(sample_imgs[idx])
        axis.set_title(f'Professor {names[idx]} | Average Score: {averages[idx]:.2f}')
        axis.axis('off')
        
    # Index of sample with highest average ranking
    idx_min = np.argmin(np.array(averages))
    
    # Visualize contestants per cluster as montage subplots
    if args.verbose:
        # Show plot
        plt.suptitle(f'Professors\' {k} Nearest Neighbors & Average Rankings')
        plt.show()
        
    # Minimum average
    min_avg = averages[idx_min]
        
    # Return results
    return idx_min, min_avg
       
# Load and format image data files
def load(directory, is_survivor):
    '''Load data (and labels) from directory.'''
    files = os.listdir(directory)  # extract filenames
    n = len(files)  # number of files

    # Load all images into 3D array
    pixels = HEIGHT * WIDTH # 70 x 70
    data = np.zeros((pixels * 3, n), dtype='float32') # 3 is for RGB channels
    for i in range(n):
        # print(f'Loading image: {files[i]}')
        img = imread(os.path.join(directory, files[i])) # Store images as 3D numpy array
        
        # NOTE: I added this because some images had a 4th layer (alpha channel perhaps?)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        
        # Check sizes of each image
        if img.shape[0] != HEIGHT or img.shape[1] != WIDTH or img.shape[2] != 3:
            print(f"WARNING!!! Image ({img.shape[0]}, {img.shape[1]}, {img.shape[2]}) does not have expected dimensions ({HEIGHT, WIDTH, 3})")
            return -1
        
        # Flatten 3D images into 1D vector of pixels and rgb channels
        data[:, i] = img.flatten()
        
    # Initialize labels
    labels = []
        
    # Extract labels (names and season #) from filenames
    if is_survivor:
        for file in files:
            season = int(file.split('_')[0][1:])
            name = file.split('_')[1:]
            name[-1] = name[-1].split('.')[0]
            labels.append([season, " ".join(name)])
    else:
        for file in files:
            name = file.split('_')
            name[-1] = name[-1].split('.')[0]
            labels.append(" ".join(name))
            
    return data, labels
  
    
# Check if script is being run directly (not imported as a module)
if __name__ == "__main__":
	main(parser.parse_args())