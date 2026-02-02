import numpy as np

"""
CSCI 3360 Data Science I - Lab 3
Topic: Numpy, Linear Algebra, and Statistics

Instructions:
1.  Edit the functions below to complete the assignment.
2.  DO NOT change the function names or arguments.
3.  Run 'python -m unittest test_lab3.py' to verify your work.
"""

# ==========================================
# Problem 1: Solving Systems of Linear Equations
# ==========================================

def solve_linear_system(A, b):
    """
    Solves the system Ax = b for x.
    
    Args:
        A (np.ndarray): Coefficient matrix of shape (N, N).
        b (np.ndarray): Constant vector of shape (N,).
        
    Returns:
        x (np.ndarray): Solution vector of shape (N,).
    """
    # TODO: Solve the system using numpy's linear algebra solver
    x = None
    return x

# ==========================================
# Problem 2: Distance Matrix Calculation
# ==========================================

def calculate_distance_matrix(X):
    """
    Computes the pairwise Euclidean distance matrix using broadcasting.
    
    Args:
        X (np.ndarray): Input data of shape (N, D).
        
    Returns:
        dists (np.ndarray): Distance matrix of shape (N, N) where dists[i, j]
                            is the distance between point i and point j.
    """
    # TODO: Use broadcasting to calculate squared differences
    # Constraint: Do NOT use a for loop.
    # Hint: Reshape X or use np.newaxis to align (N, 1, D) and (1, N, D)
    
    dists = None
    return dists

# ==========================================
# Problem 3: Image Compression via SVD
# ==========================================

def compress_matrix(matrix, k):
    """
    Performs SVD and reconstructs the matrix using top k singular values.
    
    Args:
        matrix (np.ndarray): The original matrix.
        k (int): Number of singular values to keep.
        
    Returns:
        reconstructed (np.ndarray): The compressed matrix.
        mse (float): The Mean Squared Error between original and reconstructed.
    """
    # TODO: Compute SVD using np.linalg.svd
    # U, S, Vt = ...
    
    # TODO: Reconstruct using top k singular values
    # Hint: You must construct the diagonal matrix Sigma of the correct size.
    # Be careful with dimensions when multiplying back.
    
    reconstructed = None
    
    # TODO: Calculate Mean Squared Error (MSE)
    # MSE = mean of (original - reconstructed)^2
    mse = None
    
    return reconstructed, mse

# ==========================================
# Problem 4: Verifying the Central Limit Theorem
# ==========================================

def get_sample_means(population, sample_size, num_samples):
    """
    Draws multiple samples from the population and calculates the mean of each.
    
    Args:
        population (np.ndarray): The full population data.
        sample_size (int): The number of items in a single sample (N).
        num_samples (int): How many samples to draw.
        
    Returns:
        sample_means (np.ndarray): Array of shape (num_samples,) containing means.
    """
    sample_means = []
    
    # TODO: Loop num_samples times:
    # 1. Pick a random sample of size `sample_size` from `population`
    # 2. Compute the mean of that sample
    # 3. Append to list
    
    return np.array(sample_means)

# ==========================================
# Problem 5: Data Standardization
# ==========================================

def standardize_data(X):
    """
    Standardizes the dataset so each feature has mean 0 and std 1.
    Formula: Z = (X - mean) / std
    
    Args:
        X (np.ndarray): Input data (N samples, D features).
        
    Returns:
        X_std (np.ndarray): Standardized data.
    """
    # TODO: Calculate mean and std for each feature (column-wise)
    # Use axis=0
    
    # TODO: Apply Z-score formula
    X_std = None
    
    return X_std

# ==========================================
# Bonus: Cosine Similarity Search
# ==========================================

def find_most_similar(q, D):
    """
    Finds the index of the vector in D that is most similar to q.
    
    Args:
        q (np.ndarray): Query vector (D,).
        D (np.ndarray): Database matrix (N, D).
        
    Returns:
        best_index (int): Index of the row in D with highest cosine similarity.
        max_sim (float): The similarity score.
    """
    # TODO: Compute Cosine Similarity for all rows in D against q
    # Formula: (A . B) / (||A|| * ||B||)
    
    best_index = None
    max_sim = -1.0
    
    return best_index, max_sim
