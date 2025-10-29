
import numpy as np
EPS = 1e-12

def radial_activation(d):
    d_arr = np.array(d, dtype=float)
    d_safe = np.where(d_arr <= 0, EPS, d_arr)
    return (d_safe ** 2) * np.log(d_safe)

def initialize_centers_random(X, n_centers, seed=None):
    rng = np.random.default_rng(seed)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    centers = rng.uniform(mins, maxs, size=(n_centers, X.shape[1]))
    return centers

def compute_phi(X, centers):
    n_samples = X.shape[0]
    n_centers = centers.shape[0]
    Phi = np.zeros((n_samples, n_centers))
    for j in range(n_centers):
        diff = X - centers[j]
        D = np.linalg.norm(diff, axis=1)
        Phi[:, j] = radial_activation(D)
    return Phi

def build_A(Phi):
    ones = np.ones((Phi.shape[0], 1))
    return np.hstack([ones, Phi])

def solve_weights(A, Y):
    W, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return W

def predict(A, W):
    return A.dot(W)
