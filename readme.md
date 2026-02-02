# CSCI 3360: Lab Assignment (Numpy & Linear Algebra)

![Points](../../blob/badges/.github/badges/points.svg)

**Points:** 100 (+10 Bonus)  

## 📋 Instructions
1.  Open the file `lab3.py`.
2.  Locate the functions marked with `TODO`.
3.  Write your Python code inside these functions to solve the problems described in the docstrings.
    * **Problem 1:** Solve a generic system of linear equations ($Ax=b$).
    * **Problem 2:** Calculate a distance matrix using Numpy broadcasting.
    * **Problem 3:** Compress a matrix using SVD and calculate the Mean Squared Error (MSE).
    * **Problem 4:** Verify the Central Limit Theorem by sampling from a population.
    * **Problem 5:** Standardize a dataset (Z-score normalization).
    * **Bonus:** Implement Cosine Similarity search.

## How to Check Your Grade
**Method 1: The Badge**
The "Points" badge at the top of this file shows your current score (e.g., 100/100).
* **Note:** It takes about 1-2 minutes to update after you push your code. Refresh the page to see the change.

**Method 2: Detailed Feedback**
If you don't get 100%, check which specific tests failed:
1.  Click the **Actions** tab at the top of this repository.
2.  Click the latest workflow run (usually named "GitHub Classroom Workflow").
3.  Click the **Autograding** job to see the logs.
4.  Scroll down to see exactly which math outputs were incorrect.

You can run the tests on your own computer before pushing to GitHub. Open your terminal in this folder and run:

```bash
python3 -m unittest test_lab3.py
