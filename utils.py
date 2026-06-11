"""
Script containing utility functions 

made by: Pablo Bakker and Jochem
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt


# Steering Vector => a(theta): M x 1
def a(theta, M, delta):
  """
  Function to compute the steering vector for a uniform linear array of M sensors, 
  a spacing of delta, and a signal direction of theta.
  Returns the steering vector a(theta) of shape (M, 1).
  """
  # Get parameters
  theta0 = theta * np.pi/180 # convert to radians
  delay = delta * np.sin(theta0)
  microphones = np.arange(M)

  # Steering vector
  a = np.exp(1j * 2*np.pi * microphones[:,np.newaxis] * delay)

  return a

# Data Generation
def gendata(M, N, delta, theta, f, SNR):
    """ 
    Function to generate sinusoidal data with complex AWGN noise received by a uniform linear array of M sensors, 
    with N samples, array spacing of delta, signal directions theta, frequencies f, and signal-to-noise ratios SNR. 
    Returns the received signal X of shape (M, N), the steering matrix A of shape (M, d), and the signal matrix S of shape (d, N).
    """
    d = len(theta) # Number of signals

    A = np.zeros((M, d), dtype=complex)
    for i in range(d):
        A[:,i] = a(theta[i], M, delta).flatten() # Steering vector for each signal direction
    
    S = np.zeros((d, N), dtype=complex)
    for i in range(d):
        S[i, :] = np.exp(1j * 2 * np.pi * f[i] * np.arange(N)) # Sinusoidal signal for each frequency

    signal = A @ S # Received signal at the array

    SNR_linear = 10**(SNR/10) # Convert SNR from dB to linear scale
    noise = 1/np.sqrt(2*SNR_linear) * (np.random.randn(M, N) + 1j*np.random.randn(M, N)) # Complex AWGN noise
    
    return signal + noise, A, S


# ESPRIT Algorithm
def esprit(X, d):
    """
    ESPRIT for direction-of-arrival estimation on a ULA.
    With X an M x N data matrix, and d the number of sources.
    returns: theta (length-d array of angles in degrees)
    """

    # Get the signal subspace from the SVD of X
    U, _, _ = np.linalg.svd(X)
    Uz = U[:, :d]  # (M x d) spans the column space of A

    # Shift-invariance split 
    Ux = Uz[:-1, :]  # rows 1 ... M-1
    Uy = Uz[1:, :] # rows 2 ... M

    # Ux^+ Uy = T^{-1} Theta T  
    Uxy = np.linalg.pinv(Ux) @ Uy
    phi = np.linalg.eigvals(Uxy)

    # Invert phi = exp(j 2 pi delta sin(theta)) for theta
    sin_theta = np.angle(phi) / (np.pi)
    theta = np.degrees(np.arcsin(sin_theta))

    return theta


def espritfreq(X, d):
    """
    ESPRIT for frequency estimation, exploiting the shift-
    invariant structure of S in the model X = A S.
    With X an M x N data matrix and d the number of sources
    returns: f (length-d array of normalized frequencies in [0, 1))
    """
    # Transposed exactly same problem as with DOA
    Xt = X.T                       

    # Get the signal subspace from the SVD of Xt
    U, _, _ = np.linalg.svd(Xt)
    Uz = U[:, :d] # (N x d) spans the column space of S^T

    # Shift-invariance split 
    Ux = Uz[:-1, :]  # samples k = 0 ... N-2
    Uy = Uz[1:, :]  # samples k = 1 ... N-1

    # Ux^+ Uy = T^{-1} Theta T
    Uxy = np.linalg.pinv(Ux) @ Uy
    psi = np.linalg.eigvals(Uxy)

    # Invert psi = exp(j 2 pi f) for f, mapped to [0, 1)
    f = np.angle(psi) / (2 * np.pi)
    f = np.mod(f, 1.0)

    return f


# Joint Diagonalization from matlab made by Claude
def joint_diag(A, jthresh=1.0e-8):
    """
    Joint approximate diagonalization of n complex matrices.

    Python code translated by an LLM from MATLAB of J.-F. Cardoso's joint_diag.m (Jacobi-angles method,
    Cardoso & Souloumiac, SIAM J. Mat. Anal. Appl. 17(1), 1996).

    Parameters
    ----------
    A : ndarray, shape (m, m*n)
        Horizontal concatenation of n matrices each m x m:
        A = [A1 A2 ... An].
    jthresh : float
        Stopping threshold on the Givens rotation sines.

    Returns
    -------
    V : ndarray, shape (m, m)
        Unitary matrix. Columns are the common (approximate) eigenvectors.
    D : ndarray, shape (m, m*n)
        Stack of V^H A_k V, each approximately diagonal.
    """
    A = np.array(A, dtype=complex)   # work on a copy; the algorithm mutates A
    m, nm = A.shape

    # B maps the 3-vector of Givens statistics into the real symmetric
    # 3x3 problem whose top eigenvector gives the rotation angles.
    B = np.array([[1, 0, 0],
                  [0, 1, 1],
                  [0, -1j, 1j]], dtype=complex)
    Bt = B.conj().T

    V = np.eye(m, dtype=complex)

    encore = True
    while encore:
        encore = False
        for p in range(m - 1):
            Ip = np.arange(p, nm, m)          # columns p, p+m, p+2m, ...
            for q in range(p + 1, m):
                Iq = np.arange(q, nm, m)      # columns q, q+m, q+2m, ...

                # Givens statistics g : shape (3, n)
                g = np.array([A[p, Ip] - A[q, Iq],
                              A[p, Iq],
                              A[q, Ip]])

                # Real symmetric 3x3 problem; take eigenvector of largest eigval
                M = np.real(B @ (g @ g.conj().T) @ Bt)
                w, vcp = np.linalg.eigh(M)     # eigh returns ascending eigenvalues
                angles = vcp[:, np.argmax(w)]  # eigenvector for the largest
                if angles[0] < 0:
                    angles = -angles

                c = np.sqrt(0.5 + angles[0] / 2.0)
                s = 0.5 * (angles[1] - 1j * angles[2]) / c

                if abs(s) > jthresh:
                    encore = True
                    G = np.array([[c, -np.conj(s)],
                                  [s,  c]], dtype=complex)
                    pair = [p, q]

                    # Update V (columns) and A (rows then columns)
                    V[:, pair] = V[:, pair] @ G
                    A[pair, :] = G.conj().T @ A[pair, :]
                    A[:, np.concatenate([Ip, Iq])] = np.concatenate(
                        [c * A[:, Ip] + s * A[:, Iq],
                         -np.conj(s) * A[:, Ip] + c * A[:, Iq]], axis=1)

    D = A
    return V, D


# Joint Estimation of Directions and Frequencies
def joint(X, d, m):
    """
    Joint estimation of directions and frequencies via temporal smoothing and joint diagonalization. 
    with X an M x N data matrix, d the number of sources and m the temporal smoothing factor.

    Returns:
    theta : (d,) directions in degrees
    f : (d,) normalized frequencies in [0, 1)
    """
    M, N = X.shape
    delta = 0.5
    Ns = N - m + 1 # columns surviving the smoothing

    # Build the temporally-smoothed data matrix
    Xs = np.zeros((m * M, Ns), dtype=complex)
    for l in range(m):
        Xs[l*M:(l+1)*M, :] = X[:, l:l + Ns]

    # Signal subspace 
    U, _, _ = np.linalg.svd(Xs)
    Uz = U[:, :d]  # (m*M x d)

    # Selecting within each m-block antennas to shift in space
    rows_x, rows_y = [], []
    for l in range(m):
        base = l * M
        rows_x.extend(range(base, base + M - 1))
        rows_y.extend(range(base + 1, base + M))
    Ux_s, Uy_s = Uz[rows_x, :], Uz[rows_y, :]
    Uxy_s = np.linalg.pinv(Ux_s) @ Uy_s  # T^{-1} Phi T

    # Selecting within samples to shift in time
    Ux_t = Uz[0:(m-1)*M, :]
    Uy_t = Uz[M:m*M, :]
    Uxy_t = np.linalg.pinv(Ux_t) @ Uy_t  #  T^{-1} Psi T

    # Joint diagonalization of the two matrices 
    Mstack = np.hstack([Uxy_s, Uxy_t])
    V, D = joint_diag(Mstack)   # joint_diag expects [M1 M2] concatenated horizontally, returns V, D

    # Eigenvalues are the diagonals of the approximately diagonalized blocks.
    phi = np.diag(D[:, 0:d])
    psi = np.diag(D[:, d:2*d])

    # Map eigenvalues back to physical parameters 
    sin_theta = np.angle(phi) / (2 * np.pi * delta)
    theta = np.degrees(np.arcsin(sin_theta))

    f = np.mod(np.angle(psi) / (2 * np.pi), 1.0)

    return theta, f


# Functions for matching estimates to truth since order is arbitrary
def match(estimates, truth):
    """Permute estimates to best match truth (minimum total |error|, d=2)."""
    # two possible orderings for d=2
    keep = np.array([0, 1])
    swap = np.array([1, 0])
    err_keep = np.sum(np.abs(estimates[keep] - truth))
    err_swap = np.sum(np.abs(estimates[swap] - truth))
    return estimates[keep] if err_keep <= err_swap else estimates[swap]

def match_joint(th, fr, th_true, fr_true):
    """Pair (theta, f) jointly: choose the ordering minimizing combined error.
       Normalize each parameter's error by its scale so neither dominates."""
    th_scale = max(np.ptp(th_true), 1.0)   # spread of true angles
    fr_scale = max(np.ptp(fr_true), 1e-3)  # spread of true freqs
    keep = np.array([0, 1])
    swap = np.array([1, 0])
    err_keep = np.sum(np.abs(th[keep]-th_true))/th_scale + np.sum(np.abs(fr[keep]-fr_true))/fr_scale
    err_swap = np.sum(np.abs(th[swap]-th_true))/th_scale + np.sum(np.abs(fr[swap]-fr_true))/fr_scale
    perm = keep if err_keep <= err_swap else swap
    return th[perm], fr[perm]


# Statistics over runs
def stats(arr):  # arr: (nSNR, nruns, d)
    return arr.mean(axis=1), arr.std(axis=1)   # each (nSNR, d)


# Function to match rows of an estimated source matrix S_est to the true source matrix
def match_rows(S_true, S_est):
    """Permute + phase/scale-align S_est rows to S_true (resolve ambiguities)."""
    d = S_true.shape[0]
    C = np.abs(S_true.conj() @ S_est.conj().T)      # correlation magnitudes
    perm = np.argmax(C, axis=1)
    S_aligned = S_est[perm].copy()
    for i in range(d):
        scale = (S_true[i] @ S_aligned[i].conj()) / (S_aligned[i] @ S_aligned[i].conj())
        S_aligned[i] = S_aligned[i] * scale
    return S_aligned

# Compute which angle a weight vector w_row responds most strongly to from a list of angles
def target_index(w_row, angles, M, delta):
    """Return index of the angle this weight vector responds most strongly to."""
    resp = [abs(w_row.conj() @ a(ang, M, delta).flatten()) for ang in angles]
    return int(np.argmax(resp))


# Compute spatial response of a weight vector w_row to a range of angles
def spatial_response(w_row, M, delta, angles):
    return np.array([abs(w_row.conj() @ a(ang, M, delta).flatten()) for ang in angles])