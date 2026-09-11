import numpy as np
from numpy.linalg import norm
import cvxpy as cp

from .graphSVD import _svds, graphSVD
from .preprocessing import (
    PreprocessingError,
    preprocess_features,
    weighted_debiased_correction,
)
from .recovery import (
    recover_W,
    refit_A_full_l2,
    refit_A_full_poisson,
    spectral_A_unweighted,
)
from .utils import _euclidean_proj_simplex
from .vertex_hunting import VertexHuntingError, vertex_hunt

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
        initialize=True,
        threshold_method="none",
        alpha=0.005,
        weight_method="none",
        tau=0.0,
        weight_cap=None,
        weight_common_scale="none",
        vertex_hunter="spa_current",
        vertex_hunter_parameters=None,
        A_recovery="current",
        initialization="current",
        embedding_source="U_hat",
        random_state=None,
        condition_threshold=1e12,
        graph_nfolds=5,
        graph_cv_fold_mode="legacy_first_three",
        graph_n_jobs=3,
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
        random_state : int or None
            Seed the spectral initialization and graph cross-validation folds.
            None preserves the historical global NumPy random state.
        graph_nfolds : int
            Number of graph folds to construct (default 5).
        graph_cv_fold_mode : {"legacy_first_three", "all"}
            Select lambda using the first three folds for historical parity,
            or all nonempty folds. This applies with every preprocessing mode.
        graph_n_jobs : int
            Number of graph cross-validation workers (default 3); use 1 for
            serial execution.

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
        self.threshold_method = threshold_method
        self.alpha = alpha
        self.weight_method = weight_method
        self.tau = tau
        self.weight_cap = weight_cap
        self.weight_common_scale = weight_common_scale
        self.vertex_hunter = vertex_hunter
        self.vertex_hunter_parameters = dict(vertex_hunter_parameters or {})
        self.A_recovery = A_recovery
        self.initialization = initialization
        self.embedding_source = embedding_source
        self.random_state = random_state
        self.condition_threshold = condition_threshold
        self.graph_nfolds = graph_nfolds
        self.graph_cv_fold_mode = graph_cv_fold_mode
        self.graph_n_jobs = graph_n_jobs

    def fit(
        self,
        X,
        N,
        K,
        edge_df,
        weights,
        *,
        true_A=None,
        population_M=None,
        counts=None,
        fail_on_rank_loss=True,
    ):
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
        X_original = np.asarray(X, dtype=float)
        baseline_noop = (
            self.threshold_method == "none"
            and self.weight_method == "none"
            and self.vertex_hunter == "spa_current"
            and self.A_recovery == "current"
            and self.initialization == "current"
            and self.embedding_source == "U_hat"
        )
        if baseline_noop:
            # Keep the historical path free of new preprocessing calls so the
            # regression fixture tests the actual pre-refactor computation.
            X_spectral = X_original
            self.preprocessing_result = None
            retained_indices = np.arange(X_original.shape[1], dtype=int)
            feature_weights = np.ones(X_original.shape[1])
        else:
            self.preprocessing_result = preprocess_features(
                X_original,
                N,
                threshold_method=self.threshold_method,
                alpha=self.alpha,
                weight_method=self.weight_method,
                tau=self.tau,
                weight_cap=self.weight_cap,
                weight_common_scale=self.weight_common_scale,
                K=K,
                true_A=true_A,
                population_M=population_M,
                fail_on_rank_loss=fail_on_rank_loss,
            )
            X_spectral = self.preprocessing_result.X_transformed
            retained_indices = self.preprocessing_result.threshold.retained_indices
            feature_weights = self.preprocessing_result.weighting.weights
        if min(X_spectral.shape) <= K:
            raise PreprocessingError(
                f"transformed matrix shape {X_spectral.shape} cannot support rank K={K}"
            )
        if np.linalg.matrix_rank(X_spectral) < K:
            raise PreprocessingError(
                f"transformed matrix has rank below requested K={K}"
            )

        self.X_original = X_original
        self.X_spectral = X_spectral
        self.retained_feature_indices = retained_indices
        self.discarded_feature_indices = np.setdiff1d(
            np.arange(X_original.shape[1]), retained_indices, assume_unique=True
        )
        self.feature_weights = feature_weights

        if self.method == "pLSI":
            print("Running pLSI...")
            rng = (
                None
                if self.random_state is None
                else np.random.default_rng(self.random_state)
            )
            self.U, self.L, self.V = _svds(X_spectral, K, rng)
            self.L = np.diag(self.L)
            self.V = self.V.T
            self.U_init = None
            self.V_init = None
            self.L_init = None
            self.graph_metadata = {
                "U_bar": self.U,
                "U_bar_history": [self.U],
                "score_history": [],
                "lambd_history": [],
                "cv_history": [],
                "initialization": "direct_svd",
            }
        else:
            print("Running graph aligned pLSI...")
            if baseline_noop:
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
                    X_spectral,
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
                    self.initialize,
                    random_state=self.random_state,
                    nfolds=self.graph_nfolds,
                    cv_fold_mode=self.graph_cv_fold_mode,
                    n_jobs=self.graph_n_jobs,
                )
                self.graph_metadata = None
            else:
                N_values = np.asarray(N, dtype=float)
                N_mean = float(N_values.mean())
                unequal_N = N_values.ndim > 0 and N_values.size > 1 and not np.allclose(
                    N_values.reshape(-1), N_values.reshape(-1)[0]
                )
                correction = None
                if self.initialization in {
                    "weighted_debiased",
                    "weighted_debiased_mean_N_approx",
                }:
                    if unequal_N and self.initialization == "weighted_debiased":
                        raise PreprocessingError(
                            "unequal document lengths require the explicitly named "
                            "weighted_debiased_mean_N_approx initialization"
                        )
                    eta_retained = X_original[:, retained_indices].mean(axis=0)
                    correction = weighted_debiased_correction(
                        eta_retained,
                        feature_weights,
                        n=X_original.shape[0],
                        N=N_mean,
                    )
                (
                    self.U,
                    self.V,
                    self.L,
                    self.U_init,
                    self.V_init,
                    self.L_init,
                    self.lambd,
                    self.lambd_errs,
                    self.used_iters,
                    self.graph_metadata,
                ) = graphSVD(
                    X_spectral,
                    N_mean,
                    K,
                    edge_df,
                    weights,
                    self.lamb_start,
                    self.step_size,
                    self.grid_len,
                    self.maxiter,
                    self.eps,
                    self.verbose,
                    self.initialize,
                    initialization=self.initialization,
                    debias_correction=correction,
                    return_metadata=True,
                    random_state=self.random_state,
                    nfolds=self.graph_nfolds,
                    cv_fold_mode=self.graph_cv_fold_mode,
                    n_jobs=self.graph_n_jobs,
                )
        
        print("Running SPOC...")
        if baseline_noop:
            J, H_hat = self.preconditioned_spa(self.U, K, self.precondition)

            self.W_hat = self.get_W_hat(self.U, H_hat)
            self.A_hat = self.get_A_hat(self.W_hat, X_original)
            self.vertex_result = None
            self.W_recovery_result = None
            self.A_recovery_result = None
            if self.return_anchor_docs:
                self.anchor_indices = J

            if self.U_init is not None:
                J_init, H_hat_init = self.preconditioned_spa(self.U_init, K, self.precondition)
                self.W_hat_init = self.get_W_hat(self.U_init, H_hat_init)
                self.A_hat_init = self.get_A_hat(self.W_hat_init, X_original)
        else:
            if self.embedding_source == "U_hat":
                embedding = self.U
            elif self.embedding_source == "U_bar":
                embedding = self.graph_metadata["U_bar"]
            else:
                raise ValueError("embedding_source must be 'U_hat' or 'U_bar'")
            parameters = dict(self.vertex_hunter_parameters)
            parameters.setdefault("precondition", self.precondition)
            if self.vertex_hunter != "spa_current":
                parameters.pop("precondition", None)
            parameters["condition_threshold"] = self.condition_threshold
            parameters["raise_on_failure"] = True
            self.vertex_result = vertex_hunt(
                embedding,
                K,
                self.vertex_hunter,
                random_state=self.random_state,
                **parameters,
            )
            H_hat = self.vertex_result.vertices
            recovery_embedding = self.vertex_result.embedding_used
            if recovery_embedding is None:
                recovery_embedding = embedding
            self.W_recovery_result = recover_W(
                recovery_embedding, H_hat, condition_threshold=self.condition_threshold
            )
            self.W_hat = self.W_recovery_result.simplex_projected
            if self.return_anchor_docs:
                selected = self.vertex_result.selected_observation_indices
                self.anchor_indices = [] if selected is None else selected.tolist()

            if self.A_recovery == "current":
                self.A_hat = self.get_A_hat(self.W_hat, X_original)
                self.A_recovery_result = None
            elif self.A_recovery == "A_spectral_unweighted":
                if self.embedding_source != "U_hat":
                    raise ValueError(
                        "A_spectral_unweighted requires U_hat; U_bar does not share "
                        "the returned orthogonal SVD coordinates"
                    )
                right_vectors = self.V
                if self.vertex_hunter == "spa_current":
                    signs = np.asarray(
                        self.vertex_result.parameters["coordinate_signs"], dtype=float
                    )
                    # spa_current preserves the historical in-place sign convention
                    # U' = U S.  Use V' = V S so U' L V'^T remains on the same scale.
                    right_vectors = self.V * signs[None, :]
                self.A_recovery_result = spectral_A_unweighted(
                    H_hat,
                    self.L,
                    right_vectors,
                    feature_weights,
                    retained_indices,
                    X_original.shape[1],
                )
                self.A_hat = self.A_recovery_result.A_hat
            elif self.A_recovery in {"A_full_L2", "full_L2"}:
                self.A_recovery_result = refit_A_full_l2(self.W_hat, X_original)
                self.A_hat = self.A_recovery_result.A_hat
            elif self.A_recovery in {"A_full_Pois", "full_Pois"}:
                N_values = np.asarray(N, dtype=float)
                if N_values.ndim == 0:
                    N_values = np.full(X_original.shape[0], float(N_values))
                if counts is None:
                    counts = X_original * N_values.reshape(-1, 1)
                self.A_recovery_result = refit_A_full_poisson(
                    self.W_hat, counts, N_values
                )
                self.A_hat = self.A_recovery_result.A_hat
            else:
                raise ValueError(f"unknown A_recovery={self.A_recovery!r}")

        self.vertex_matrix = H_hat
        self.transformed_reconstruction_error = float(
            np.linalg.norm(X_spectral - self.U @ self.L @ self.V.T)
        )
        self.original_reconstruction_error = float(
            np.linalg.norm(X_original - self.W_hat @ self.A_hat)
        )

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
