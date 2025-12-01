import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import svds
import cvxpy as cp

from .graphSVD import graphSVD
from .utils import _euclidean_proj_simplex

class GpLSI(object):
    def __init__(
        self,
        lambd=None,
        lamb_start=0.0001,
        step_size=1.2,
        grid_len=29,
        maxiter=50,
        eps=1e-05,
        method="two-step",
        use_mpi=False,
        return_anchor_docs=True,
        verbose=0,
        precondition=False,
        initialize=True
    ):
        """
        Graph-regularized probabilistic latent semantic indexing (GpLSI).

        Parameters
        ----------
        lambd : float or None
            If provided, fixed regularization parameter used by graphSVD.
            If None, we do a grid search over lamb_start * step_size**j, j=0..grid_len-1.
        lamb_start : float
            Starting value for the lambda grid.
        step_size : float
            Multiplicative step between successive lambdas on the grid.
        grid_len : int
            Number of lambda values in the grid.
        maxiter : int
            Maximum number of iterations for graphSVD solver.
        eps : float
            Convergence tolerance for graphSVD solver.
        method : {"two-step", "pLSI"}
            - "two-step": graph-aligned pLSI (GpLSI).
            - "pLSI": vanilla pLSI via truncated SVD on X (no graph).
        use_mpi : bool
            Placeholder flag if you later want distributed graphSVD. Currently unused.
        return_anchor_docs : bool
            If True, store the indices of selected anchor documents in `anchor_indices`.
        verbose : int
            Verbosity level; passed down to graphSVD.
        precondition : bool
            Whether to use Klopp-style preconditioning in SPA.
        initialize : bool
            Whether to run an initialization pass in graphSVD.

        Attributes (after calling fit)
        --------------------------------
        U, V, L : np.ndarray
            SVD factors of the graph-regularized representation (or vanilla SVD if method="pLSI").
        U_init, V_init, L_init : np.ndarray or None
            Initialization SVD factors returned by graphSVD (if initialize=True).
        W_hat : np.ndarray, shape (n_samples, K)
            Estimated topic proportion matrix (rows are documents on the simplex).
        A_hat : np.ndarray, shape (K, n_features)
            Estimated topic loading matrix (rows are topics).
        lambd : float
            Selected regularization parameter (if method != "pLSI").
        lambd_errs : list[float]
            Grid of CV errors corresponding to each lambda.
        used_iters : list[int]
            Number of iterations used at each lambda.
        anchor_indices : list[int]
            Indices of anchor documents selected by SPA (if return_anchor_docs=True).
        """
        self.lambd = lambd
        self.lamb_start = lamb_start
        self.step_size = step_size
        self.grid_len = grid_len
        self.maxiter = maxiter
        self.eps = eps
        self.method = method
        self.return_anchor_docs = return_anchor_docs
        self.verbose = verbose
        self.use_mpi = use_mpi
        self.precondition = precondition
        self.initialize = initialize

    def fit(self, X, N, K, edge_df, weights):
        """
        Fit GpLSI/pLSI to matrix X with an optional graph regularization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Row-normalized document-term matrix.
            Typically X[i, :] sums to 1 and each row corresponds to a "document":
            - in spatial datasets: a spot/cell
            - in WhatsCooking: a (cuisine, ingredient) profile
            - in CRC: a cell or a small region

        N : float
            Average document length (mean row sum of the *unnormalized* count matrix D).
            Used by graphSVD to scale the data; in your pipelines this is usually:
            N = row_sums.mean() where row_sums = D.sum(axis=1).

        K : int
            Number of topics/components.

        edge_df : pandas.DataFrame
            Graph edge list over the n_samples nodes, with **integer indices**:

            Required columns
            ----------------
            - "src": int
                Source node index in [0, n_samples-1].
            - "tgt": int
                Target node index in [0, n_samples-1].
            - "weight": float
                Edge weight (e.g. exp(-distance / phi), or 1 for unweighted).

            Notes
            -----
            * The graph is assumed to be undirected. If you only store (i, j),
              you do not need to store (j, i); the Laplacian is built from this.
            * All node indices used in "src" and "tgt" must be valid row indices
              for X. In other words, there should be no edges to nodes outside
              [0, n_samples-1].

        weights : scipy.sparse.spmatrix, shape (n_samples, n_samples)
            Sparse adjacency/weight matrix whose nonzero entries correspond to
            edge_df["weight"], e.g.:

            weights = csr_matrix(
                (edge_df["weight"].values, (edge_df["src"].values, edge_df["tgt"].values)),
                shape=(n_samples, n_samples),
            )

            In your real-data helpers this is typically a CSR matrix.
            `graphSVD` uses this to construct the graph Laplacian.

        Returns
        -------
        self : GpLSI
            Fitted estimator.

        Notes
        -----
        - If `method == "pLSI"`, the graph is *ignored*: we simply run a truncated
          SVD on X (via `svds`) and skip graphSVD.
        - Otherwise, graphSVD is called:

              U, V, L, U_init, V_init, L_init, lambd, lambd_errs, used_iters = graphSVD(...)

          and then SPA is used to extract anchor documents and latent topics.
        """
        if self.method == "pLSI":
            print("Running pLSI...")
            self.U, self.L, self.V = svds(X, k=K)
            self.L = np.diag(self.L)
            self.V = self.V.T
            self.U_init = None
        else:
            print("Running graph aligned pLSI...")
            (
                self.U,
                self.V,
                self.L,
                self.U_init,
                self.V_init,
                self.L_init,
                self.lambd,
                self.lambd_errs,
                self.used_iters
            ) = graphSVD(
                X,
                N,
                K,
                edge_df,
                weights,
                self.lamb_start,
                self.step_size,
                self.grid_len,
                self.maxiter,
                self.eps,
                self.verbose,
                self.initialize
            )
        
        print("Running SPOC...")
        J, H_hat = self.preconditioned_spa(self.U, K, self.precondition)

        self.W_hat = self.get_W_hat(self.U, H_hat)
        self.A_hat = self.get_A_hat(self.W_hat,X)
        if self.return_anchor_docs:
            self.anchor_indices = J
        
        if self.U_init is not None:
            J_init, H_hat_init = self.preconditioned_spa(self.U_init, K, self.precondition)
            self.W_hat_init = self.get_W_hat(self.U_init, H_hat_init)
            self.A_hat_init = self.get_A_hat(self.W_hat_init,X)

        return self

    @staticmethod
    def preprocess_U(U, K):
        for k in range(K):
            if U[0, k] < 0:
                U[:, k] = -1 * U[:, k]
        return U
    
    @staticmethod
    def precondition_M(M, K):
        Q = cp.Variable((K, K), symmetric=True)
        objective = cp.Maximize(cp.log_det(Q))
        constraints = [cp.norm(Q @ M, axis=0) <= 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        Q_value = Q.value
        return Q_value
    
    def preconditioned_spa(self, U, K, precondition=True):
        J = []
        M = self.preprocess_U(U, K).T
        if precondition:
            L = self.precondition_M(M, K)
            S = L @ M
        else:
            S = M
        
        for t in range(K):
                maxind = np.argmax(norm(S, axis=0))
                s = np.reshape(S[:, maxind], (K, 1))
                S1 = (np.eye(K) - np.dot(s, s.T) / norm(s) ** 2).dot(S)
                S = S1
                J.append(maxind)
        H_hat = U[J, :]
        return J, H_hat

    def get_W_hat(self, U, H):
        projector = H.T.dot(np.linalg.inv(H.dot(H.T)))
        theta = U.dot(projector)
        theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj

    def get_A_hat(self, W_hat, M):
        projector = (np.linalg.inv(W_hat.T.dot(W_hat))).dot(W_hat.T)
        theta = projector.dot(M)
        theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj
