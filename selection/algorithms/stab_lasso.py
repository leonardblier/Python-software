import numpy as np
from sklearn.linear_model import Lasso
from ..constraints.affine import (constraints, selection_interval,
                                 interval_constraints,
                                 sample_from_constraints,
                                 one_parameter_MLE,
                                 gibbs_test,
                                 stack)
from ..distributions.discrete_family import discrete_family

from lasso import lasso, instance

from sklearn.cluster import AgglomerativeClustering

from scipy.stats import norm as ndist, t as tdist

import statsmodels.api as sm

try:
    import cvxpy as cvx
except ImportError:
    warnings.warn('cvx not available')
    pass

DEBUG = False

import pdb

    
class stab_lasso(object):

    alpha = 0.05
    UMAU = False

    
    def __init__(self, y, X, lam, n_split = 1, size_split = None, k=None, sigma=1, connectivity=None):
        r"""

        Create a new post-selection dor the LASSO problem

        Parameters
        ----------

        y : np.float(y)
            The target, in the model $y = X\beta$

        X : np.float((n, p))
            The data, in the model $y = X\beta$

        lam : np.float
            Coefficient of the L-1 penalty in
            $\text{minimize}_{\beta} \frac{1}{2} \|y-X\beta\|^2_2 + 
                \lambda\|\beta\|_1$

        sigma : np.float
            Standard deviation of the gaussian distribution :
            The covariance matrix is
            `sigma**2 * np.identity(X.shape[0])`.
            Defauts to 1.
        """
        self.y = y
        self.X = X
        self.sigma = sigma
        n, p = X.shape
        self.lagrange = lam / n
        self.n_split = n_split
        if size_split == None:
            size_split = n
        self.size_split = size_split
        if k == None:
            k = p
        self._n_clusters = k
        self.connectivity = connectivity

        self._covariance = self.sigma**2 * np.identity(X.shape[0])

    @staticmethod
    def projection(X, k, connectivity):
        """
        Take the data, and returns a matrix, to reduce the dimension
        Returns, P, invP, (P.(X.T)).T
        """
        n, p = X.shape
        
        ward = AgglomerativeClustering(n_clusters = k, connectivity=connectivity)
        ward.fit(X.T)

        labels = ward.labels_
        P = np.zeros((k, p))
        for i in range(p):
            P[labels[i] ,i] = 1.
        P_inv = np.copy(P)
        P_inv = P_inv.T
        
        s_array = P.sum(axis = 0)
        P = P/s_array
            
        # P_inv = np.linalg.pinv(P)
        
        X_proj = np.dot(P, X.T).T

        
        return P, P_inv, X_proj, labels

    
    def fit(self, sklearn_alpha=None, **lasso_args):
        X = self.X
        y = self.y
        n, p = X.shape
        sigma = self.sigma
        #lam = self.lagrange * n
        n_split = self.n_split
        size_split = self.size_split
        n_clusters = self._n_clusters
        connectivity = self.connectivity

        # pdb.set_trace()
        cons_list = []
        beta_array = np.zeros((p, n_split))

        split_array = np.zeros((n_split, size_split), dtype=int)
        clust_array = np.zeros((n_split, p), dtype=int)
        for i in range(n_split):
            split = np.random.choice(n, size_split, replace = False)
            split.sort()
            split_array[i, :] = split
            
            X_splitted = X[split,:]
            y_splitted = y[split]
            

            P, P_inv, X_proj, labels = self.projection(X_splitted, \
                                                       n_clusters, \
                                                       connectivity)
            clust_array[i, :] = labels

            theta = 0.5
            lam = theta * np.max(np.abs(np.dot(X_proj.T, y_splitted)))
            lasso_splitted = lasso(y_splitted, X_proj, lam, sigma)

            lasso_splitted.fit(sklearn_alpha, **lasso_args)

            beta_proj = lasso_splitted.soln
            beta = np.dot(P_inv, beta_proj)
            
            beta_array[:,i] = beta
            # constraint_splitted = lasso_splitted.constraints

            # linear_part_splitted = constraint_splitted.linear_part
            # offset = constraint_splitted.offset

            # n_inequality, _ = linear_part_splitted.shape
            # linear_part = np.zeros((n_inequality, n))
            # linear_part[:, split] = linear_part_splitted

            # constraint = constraints(linear_part, offset)
            # constraint = constraint_splitted
            # cons_list.append(constraint)

        beta = beta_array.mean(axis=1)
        # self._constraints = stack(*cons_list)
        self._soln = beta
        self._beta_array = beta_array
        self._split_array = split_array
        self._clust_array = clust_array

    @property
    def soln(self):
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln


    @property
    def constraints(self):
        return self._constraints

    @property
    def intervals(self):
        if hasattr(self, "_intervals"):
            return self._intervals

        X = self.X
        y = self.y
        n, p = X.shape
        n_split = self.n_split

        beta_array = self._beta_array

        C = self._constraints

        eta_array = np.zeros((n_split, n, p))
        
        for k in range(n_split):
            beta = beta_array[:, k]
            active = (beta != 0)
            X_E = X[:, active]
            try:
                XEinv = np.linalg.pinv(X_E)
            except:
                XEinv = None
                pass
            
            if XEinv is not None:
                eta_array[k, :, active] = XEinv

        eta_array = (1./n_split)*eta_array.sum(0)

        self._intervals = []
        for i in range(p):
            eta = eta_array[:, i]

            if eta.any():
                _interval = C.interval(eta, self.y, alpha=self.alpha,
                                       UMAU=self.UMAU)
                self._intervals.append((i, eta, (eta*self.y).sum(), _interval))
        return self._intervals


    @property
    def active_pvalues(self, doc="Tests for active variables adjusted " + \
                       " for selection."):
        if hasattr(self, "_pvals"):
            return self._pvals

        X = self.X
        y = self.y
        n, p = X.shape
        n_split = self.n_split
        

        C = self._constraints

        beta_array = self._beta_array
        eta_array = np.zeros((n_split, n, p))
        
        for k in range(n_split):
            beta = beta_array[:, k]

            if any(beta):
                active = (beta != 0)
                X_E = X[:, active]
                XEinv = np.linalg.pinv(X_E)
            else:
                XEinv = None
            if XEinv is not None:
                eta_array[k, :, active] = XEinv

        eta_array = (1./n_split)*eta_array.sum(0)

        self._pvals = []
        for i in range(p):
            eta = eta_array[:, i]

            if eta.any():
                _pval = C.pivot(eta, self.y)
                _pval = 2 * min(_pval, 1 - _pval)
                                    
                self._pvals.append((i,_pval))
        return self._pvals

    def split_pval(self):
        X = self.X
        y = self.y
        n, p = X.shape
        sigma = self.sigma
        lam = self.lagrange * n
        n_split = self.n_split
        size_split = self.size_split
        n_clusters = self._n_clusters


        beta_array = self._beta_array
        split_array = self._split_array
        clust_array = self._clust_array
        
        pvalues = np.ones((n_split, p))
        
        for i in range(n_split):
            split = np.zeros(n, dtype='bool')
            split[split_array[i,:]] = True
            y_test = y[~split]
            X_test = X[~split,:]
            
            clust = clust_array[i,:]
            P = np.zeros((n_clusters, p))
            for j in range(p):
                P[clust[j] ,j] = 1.
            P_inv = np.copy(P)
            P_inv = P_inv.T
            s_array = P.sum(axis = 0)
            P = P/s_array
            
            beta = beta_array[:, i]
            beta_proj = np.dot(P, beta)
            model_proj = (beta_proj != 0)
            model_proj_size = model_proj.sum()
            X_test_proj = np.dot(P, X_test.T).T

            X_model = X_test_proj[:, model_proj]
            beta_model = beta_proj[model_proj]

            res = sm.OLS(y_test, X_model).fit()
            #pdb.set_trace()
            pvalues_proj = np.ones(n_clusters)
            pvalues_proj[model_proj] = np.clip(model_proj_size * res.pvalues, 0., 1.)

            pvalues[i, :] = np.dot(P_inv, pvalues_proj)

        pvalues_aggr = pval_aggr(pvalues)
        self._pvalues = pvalues
        self._pval_aggr = pvalues_aggr
        return pvalues_aggr

    
    def select_model_fwer(self, alpha):
        pvalues = self._pvalues
        p, = pvalues.shape
        model = [(i, pvalues[i]) for i in range(p) if pvalues[i] < alpha]
        return model

    def select_model_fdr(self, q):
        pvalues = self._pvalues
        p, = palues.shape
        pvalues_sorted = np.sort(pvalues) / np.arange(1, n+1)

        newq = q * (1./2 + np.log(2))
        h = max(i for i in range(p) if pvalues_sorted[i] <= newq)
        model = [(i, pvalues[i]) for i in range(p) if pvalues[i] < pvalues_sorted[h]]
        return model

    
        
        
        


def pval_aggr(pvalues, gamma_min = 0.05):
    n_split, p = pvalues.shape
    
    kmin = max(1, int(gamma_min * n_split))
    pvalues_sorted = np.sort(pvalues, axis = 0)[kmin:, :]
    gamma_array = 1./n_split * (np.arange(kmin+1, n_split+1))
    pvalues_sorted = pvalues_sorted  / gamma_array[:, np.newaxis]
    q = pvalues_sorted.min(axis = 0)
    q *= 1 - np.log(gamma_min)
    q = q.clip(0., 1.)
    return q



def test(lam, n_split, size_split):
    A = instance(n = 25, p = 50, s = 5, sigma = 5.)
    X, y, beta, active, sigma = A
    B = stab_lasso(y, X, lam, n_split=n_split, size_split=size_split)
    B.fit()
    # print ("Model fitted")
    I = B.intervals
    P = B.active_pvalues
    return B, P, I


    
def multiple_test(N):
    for i in range(N):
        print i
        test()
    return "bwah"
