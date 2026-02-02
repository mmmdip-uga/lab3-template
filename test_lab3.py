import unittest
import numpy as np
import lab3  # Updated import

class TestLab3(unittest.TestCase):

    def setUp(self):
        # Set a seed for reproducibility
        np.random.seed(42)

    # ==========================================
    # Test Problem 1: Linear Systems
    # ==========================================
    def test_solve_linear_system(self):
        # System: 2x + y = 8, 3x + 4y = 18 -> Solution x=2.8, y=2.4
        A = np.array([[2, 1], [3, 4]])
        b = np.array([8, 18])
        
        expected = np.linalg.solve(A, b)
        result = lab3.solve_linear_system(A, b)
        
        self.assertIsNotNone(result, "Problem 1: Function returned None")
        self.assertTrue(np.allclose(result, expected), f"Problem 1: Incorrect solution. Expected {expected}, got {result}")

    # ==========================================
    # Test Problem 2: Distance Matrix
    # ==========================================
    def test_distance_matrix(self):
        # 3 points on a line: 0, 10, 20. Distances should be 0, 10, 20 etc.
        X = np.array([[0], [10], [20]])
        
        result = lab3.calculate_distance_matrix(X)
        expected = np.array([
            [0., 10., 20.],
            [10., 0., 10.],
            [20., 10., 0.]
        ])
        
        self.assertIsNotNone(result, "Problem 2: Function returned None")
        self.assertEqual(result.shape, (3, 3), "Problem 2: Incorrect shape")
        self.assertTrue(np.allclose(result, expected), "Problem 2: Distance calculation incorrect")

        # Test with 2D points (Pythagorean triples)
        # (0,0) to (3,4) distance is 5
        X2 = np.array([[0, 0], [3, 4]])
        res2 = lab3.calculate_distance_matrix(X2)
        self.assertAlmostEqual(res2[0, 1], 5.0, msg="Problem 2: 2D distance incorrect")

    # ==========================================
    # Test Problem 3: SVD Compression
    # ==========================================
    def test_svd_compression(self):
        # Create a random 10x10 matrix
        mat = np.random.rand(10, 10)
        k = 5
        
        rec, mse = lab3.compress_matrix(mat, k)
        
        self.assertIsNotNone(rec, "Problem 3: Reconstructed matrix is None")
        self.assertIsNotNone(mse, "Problem 3: MSE is None")
        self.assertEqual(rec.shape, mat.shape, "Problem 3: Reconstructed shape mismatch")
        
        # Verify correctness manually
        U, S, Vt = np.linalg.svd(mat)
        
        # Keep top k
        Sigma_k = np.zeros((k, k))
        np.fill_diagonal(Sigma_k, S[:k])
        
        # Reconstruct: U_k * S_k * Vt_k
        expected_rec = U[:, :k] @ Sigma_k @ Vt[:k, :]
        expected_mse = np.mean((mat - expected_rec)**2)
        
        self.assertTrue(np.allclose(rec, expected_rec), "Problem 3: SVD reconstruction incorrect")
        self.assertAlmostEqual(mse, expected_mse, places=5, msg="Problem 3: MSE calculation incorrect")

    # ==========================================
    # Test Problem 4: Central Limit Theorem
    # ==========================================
    def test_clt(self):
        pop = np.arange(0, 100) # Uniform distribution 0-99
        pop_mean = np.mean(pop)
        
        # Draw 500 samples of size 30
        means = lab3.get_sample_means(pop, sample_size=30, num_samples=500)
        
        self.assertEqual(len(means), 500, "Problem 4: Incorrect number of sample means returned")
        
        # Law of Large Numbers check: The mean of sample means should be close to population mean
        mean_of_means = np.mean(means)
        
        # Allow small margin of error due to randomness
        self.assertTrue(np.abs(mean_of_means - pop_mean) < 2.0, 
                        f"Problem 4: Mean of samples ({mean_of_means}) too far from pop mean ({pop_mean})")

    # ==========================================
    # Test Problem 5: Standardization
    # ==========================================
    def test_standardization(self):
        # Create data with mean 10 and std 2
        X = np.random.normal(loc=10, scale=2, size=(100, 3))
        
        X_std = lab3.standardize_data(X)
        
        # Check means are approx 0
        means = np.mean(X_std, axis=0)
        self.assertTrue(np.allclose(means, 0, atol=1e-7), "Problem 5: Means are not 0")
        
        # Check stds are approx 1
        stds = np.std(X_std, axis=0)
        self.assertTrue(np.allclose(stds, 1, atol=1e-7), "Problem 5: Standard deviations are not 1")

    # ==========================================
    # Test Bonus: Cosine Similarity
    # ==========================================
    def test_bonus_cosine(self):
        q = np.array([1, 0]) # Horizontal vector
        
        # Database: [Vertical, Horizontal(same), Opposite]
        D = np.array([
            [0, 1],  # 90 degrees, sim 0
            [2, 0],  # 0 degrees, sim 1 (Best)
            [-1, 0]  # 180 degrees, sim -1
        ])
        
        idx, sim = lab3.find_most_similar(q, D)
        
        self.assertEqual(idx, 1, "Bonus: Wrong index returned. Should be index 1 (parallel vector)")
        self.assertAlmostEqual(sim, 1.0, msg="Bonus: Similarity score incorrect")

if __name__ == '__main__':
    unittest.main()
