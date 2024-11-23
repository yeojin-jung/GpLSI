import numpy as np
from numpy.linalg import norm, svd, solve, qr
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment
import networkx as nx

import cvxpy as cp
from cvxpy import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem

from GpLSI import generate_topic_model as gen_model
from GpLSI.utils import *
from GpLSI.graphSVD import *

class GpLSI_(object):
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
        initialize2=False,
    ):
        """
        Parameters
        -----------

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
        self.initialize2 = initialize2

    def fit(self, X, K, edge_df, weights):
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
                self.used_iters,
                self.time_init,
                self.lambd_init
            ) = graphSVD(
                X,
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
                self.initialize2
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

    @staticmethod
    def get_W_hat_cvx(U, H, n, K):
        Theta = Variable((n, K))
        constraints = [cp.sum(Theta[i, :]) == 1 for i in range(n)]
        constraints += [Theta[i, j] >= 0 for i in range(n) for j in range(K)]
        obj = Minimize(cp.norm(U - Theta @ H, "fro"))
        prob = Problem(obj, constraints)
        prob.solve()
        return np.array(Theta.value)

    def get_W_hat(self, U, H):
        projector = H.T.dot(np.linalg.inv(H.dot(H.T)))
        theta = U.dot(projector)
        theta_simplex_proj = np.array([self._euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj

    def get_A_hat(self, W_hat, M):
        projector = (np.linalg.inv(W_hat.T.dot(W_hat))).dot(W_hat.T)
        theta = projector.dot(M)
        theta_simplex_proj = np.array([self._euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj
    
    def get_A_hat_klopp(self, L, V, H):
        theta = (H @ L) @ V.T
        theta_simplex_proj = np.array([self._euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj
    
    @staticmethod
    def _euclidean_proj_simplex(v, s=1):
        (n,) = v.shape
        # check if we are already on the simplex
        if v.sum() == s and np.alltrue(v >= 0):
            return v
        
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
       
        theta = (cssv[rho] - s) / (rho + 1.0)
        w = (v - theta).clip(min=0)
        return w
    
    @staticmethod
    def get_A_hat_cvx(W, U, L, V, p, K):
        Theta = Variable((K, p))
        constraints = [cp.sum(Theta[i, :]) == 1 for i in range(K)]
        constraints += [Theta[i, j] >= 0 for i in range(K) for j in range(p)]
        L = np.diag(np.diag(L))
        M = (U @ L) @ V.T
        obj = Minimize(cp.norm(M - W @ Theta , "fro"))
        prob = Problem(obj, constraints)
        prob.solve()
        return np.array(Theta.value)

    @staticmethod
    def _euclidean_proj_simplex(v, s=1):
        (n,) = v.shape
        # check if we are already on the simplex
        if v.sum() == s and np.alltrue(v >= 0):
            return v
        
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
       
        theta = (cssv[rho] - s) / (rho + 1.0)
        w = (v - theta).clip(min=0)
        return w

