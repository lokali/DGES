import os, json, codecs, time, hashlib, warnings, random
import numpy as np
import scipy
from math import log, sqrt
from collections.abc import Iterable
from scipy.stats import chi2, norm

from csl.utils.KCI.KCI import KCI_CInd, KCI_UInd

CONST_BINCOUNT_UNIQUE_THRESHOLD = 1e5
NO_SPECIFIED_PARAMETERS_MSG = "NO SPECIFIED PARAMETERS"
fisherz = "fisherz"
zerodel_fisherz = "zerodel_fisherz" # testwise deletion of zero values
kci = "kci"
zerodel_kci = "zerodel_kci"
chisq = "chisq"
gsq = "gsq"
d_separation = "d_separation"


def CIT(data, method='fisherz', **kwargs):
    '''
    Parameters
    ----------
    data: numpy.ndarray of shape (n_samples, n_features)
    method: str, in ["fisherz", "mv_fisherz", "mc_fisherz", "kci", "chisq", "gsq"]
    kwargs: placeholder for future arguments, or for KCI specific arguments now
        TODO: utimately kwargs should be replaced by explicit named parameters.
              check https://github.com/cmu-phil/causal-learn/pull/62#discussion_r927239028
    '''
    if method == fisherz:
        return FisherZ(data, **kwargs)
    elif method == kci:
        return KCI(data, **kwargs)
    elif method == zerodel_kci:
        return ZeroDel_KCI(data, **kwargs)
    elif method in [chisq, gsq]:
        return Chisq_or_Gsq(data, method_name=method, **kwargs)
    elif method == zerodel_fisherz:
        return ZeroDel_FisherZ(data, **kwargs)
    elif method == d_separation:
        return D_Separation(data, **kwargs)
    else:
        raise ValueError("Unknown method: {}".format(method))

class CIT_Base(object):
    # Base class for CIT, contains basic operations for input check and caching, etc.
    def __init__(self, data, cache_path=None, **kwargs):
        '''
        Parameters
        ----------
        data: data matrix, np.ndarray, in shape (n_samples, n_features)
        cache_path: str, path to save cache .json file. default as None (no io to local file).
        kwargs: for future extension.
        '''
        assert isinstance(data, np.ndarray), "Input data must be a numpy array."
        self.data = data
        self.data_hash = hashlib.md5(str(data).encode('utf-8')).hexdigest()
        self.sample_size, self.num_features = data.shape
        self.cache_path = cache_path
        self.SAVE_CACHE_CYCLE_SECONDS = 30
        self.last_time_cache_saved = time.time()
        self.pvalue_cache = {'data_hash': self.data_hash}
        if cache_path is not None:
            assert cache_path.endswith('.json'), "Cache must be stored as .json file."
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as fin: self.pvalue_cache = json.load(fin)
                except: os.remove(cache_path) # e.g., file broken. remove this file
                assert self.pvalue_cache['data_hash'] == self.data_hash, "Data hash mismatch."
            else: os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    def check_cache_method_consistent(self, method_name, parameters_hash):
        self.method = method_name
        if method_name not in self.pvalue_cache:
            self.pvalue_cache['method_name'] = method_name # a newly created cache
            self.pvalue_cache['parameters_hash'] = parameters_hash
        else:
            assert self.pvalue_cache['method_name'] == method_name, "CI test method name mismatch." # a loaded cache
            assert self.pvalue_cache['parameters_hash'] == parameters_hash, "CI test method parameters mismatch."

    def assert_input_data_is_valid(self, allow_nan=False, allow_inf=False):
        assert allow_nan or not np.isnan(self.data).any(), "Input data contains NaN. Please check."
        assert allow_inf or not np.isinf(self.data).any(), "Input data contains Inf. Please check."

    def save_to_local_cache(self):
        if not self.cache_path is None and time.time() - self.last_time_cache_saved > self.SAVE_CACHE_CYCLE_SECONDS:
            with codecs.open(self.cache_path, 'w') as fout: fout.write(json.dumps(self.pvalue_cache, indent=2))
            self.last_time_cache_saved = time.time()

    def get_formatted_XYZ_and_cachekey(self, X, Y, condition_set):
        '''
        reformat the input X, Y and condition_set to
            1. convert to built-in types for json serialization
            2. handle multi-dim unconditional variables (for kernel-based)
            3. basic check for valid input (X, Y no overlap with condition_set)
            4. generate unique and hashable cache key

        Parameters
        ----------
        X: int, or np.*int*, or Iterable<int | np.*int*>
        Y: int, or np.*int*, or Iterable<int | np.*int*>
        condition_set: Iterable<int | np.*int*>

        Returns
        -------
        Xs: List<int>, sorted. may swapped with Ys for cache key uniqueness.
        Ys: List<int>, sorted.
        condition_set: List<int>
        cache_key: string. Unique for <X,Y|S> in any input type or order.
        '''
        def _stringize(ulist1, ulist2, clist):
            # ulist1, ulist2, clist: list of ints, sorted.
            _strlst  = lambda lst: '.'.join(map(str, lst))
            return f'{_strlst(ulist1)};{_strlst(ulist2)}|{_strlst(clist)}' if len(clist) > 0 else \
                   f'{_strlst(ulist1)};{_strlst(ulist2)}'

        # every time when cit is called, auto save to local cache.
        self.save_to_local_cache()

        METHODS_SUPPORTING_MULTIDIM_DATA = ["kci"]
        if condition_set is None: condition_set = []
        # 'int' to convert np.*int* to built-in int; 'set' to remove duplicates; sorted for hashing
        condition_set = sorted(set(map(int, condition_set)))

        # usually, X and Y are 1-dimensional index (in constraint-based methods)
        if self.method not in METHODS_SUPPORTING_MULTIDIM_DATA:
            X, Y = (int(X), int(Y)) if (X < Y) else (int(Y), int(X))
            assert X not in condition_set and Y not in condition_set, "X, Y cannot be in condition_set."
            return [X], [Y], condition_set, _stringize([X], [Y], condition_set)

        # also to support multi-dimensional unconditional X, Y (usually in kernel-based tests)
        Xs = sorted(set(map(int, X))) if isinstance(X, Iterable) else [int(X)]  # sorted for comparison
        Ys = sorted(set(map(int, Y))) if isinstance(Y, Iterable) else [int(Y)]
        Xs, Ys = (Xs, Ys) if (Xs < Ys) else (Ys, Xs)
        assert len(set(Xs).intersection(condition_set)) == 0 and \
               len(set(Ys).intersection(condition_set)) == 0, "X, Y cannot be in condition_set."
        return Xs, Ys, condition_set, _stringize(Xs, Ys, condition_set)

class FisherZ(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('fisherz', NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid()
        self.correlation_matrix = np.corrcoef(data.T)

    def __call__(self, X, Y, condition_set=None, return_rho=False, return_sample_size=False):
        '''
        Perform an independence test using Fisher-Z's test.

        Parameters
        ----------
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        '''
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            res = self.pvalue_cache[cache_key]
            if len(res) == 2: res = (res[0], res[1], self.sample_size)  # to be adabpted to old cache format
            if return_rho: return res if return_sample_size else res[:2]
            else: return (res[0], res[2]) if return_sample_size else res[0]
        var = Xs + Ys + condition_set
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        inv = np.linalg.pinv(sub_corr_matrix)
        r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
        if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(r) # may happen when samplesize is very small or relation is deterministic
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        res = (p, r, self.sample_size)
        self.pvalue_cache[cache_key] = res[:2]
        if return_rho: return res if return_sample_size else res[:2]
        else: return (res[0], res[2]) if return_sample_size else res[0]

class ZeroDel_FisherZ(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent(zerodel_fisherz, NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid()
        self.nonzero_boolean_mat = data != 0

    def _get_index_no_mv_rows(self, var):
        return np.where(np.all(self.nonzero_boolean_mat[:, var], axis=1))[0]

    def __call__(self, X, Y, condition_set=None, return_rho=False, return_sample_size=False):
        '''
        Perform an independence test using Fisher-Z's test for data with missing values.

        Parameters
        ----------
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        '''
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        var = Xs + Ys + condition_set
        if cache_key in self.pvalue_cache:
            res = self.pvalue_cache[cache_key]
            if return_rho: return res if return_sample_size else res[:2]
            else: return (res[0], res[2]) if return_sample_size else res[0]

        test_wise_deletion_XYcond_rows_index = self._get_index_no_mv_rows(condition_set)
        N_test_wise_deleted = len(test_wise_deletion_XYcond_rows_index)
        if N_test_wise_deleted <= len(condition_set) + 3:
            warnings.warn('Too few samples after testwise deletion to run fisherz test. Please check your data. Set pval=1.')
            self.pvalue_cache[cache_key] = res = (1 - np.finfo(float).eps, 0, N_test_wise_deleted)
            if return_rho: return res if return_sample_size else res[:2]
            else: return (res[0], res[2]) if return_sample_size else res[0]

        test_wise_deleted_data_var = self.data[test_wise_deletion_XYcond_rows_index][:, var]
        # print(test_wise_deleted_data_var)
        sub_corr_matrix = np.corrcoef(test_wise_deleted_data_var.T)
        if np.any(np.isnan(sub_corr_matrix)):
            sub_corr_matrix = np.nan_to_num(sub_corr_matrix)
        # print(sub_corr_matrix)
        inv = scipy.linalg.pinv(sub_corr_matrix)   # use pinv to avoid singular matrix error; any other way to calculate this partial correlation?
        # print(inv)

        if inv[0, 0] * inv[1, 1] <=0:
            warnings.warn('Error in inv. Please check your data. Set pval=1.')
            self.pvalue_cache[cache_key] = res = (1 - np.finfo(float).eps, 0, N_test_wise_deleted)
            if return_rho: return res if return_sample_size else res[:2]
            else: return (res[0], res[2]) if return_sample_size else res[0]

        r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])

        if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(r)  # may happen when samplesize is very small or relation is deterministic
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(N_test_wise_deleted - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        self.pvalue_cache[cache_key] = res = (p, r, N_test_wise_deleted)
        if return_rho: return res if return_sample_size else res[:2]
        else: return (res[0], res[2]) if return_sample_size else res[0]

class KCI(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        kci_ui_kwargs = {k: v for k, v in kwargs.items() if k in
                         ['kernelX', 'kernelY', 'null_ss', 'approx', 'est_width', 'polyd', 'kwidthx', 'kwidthy']}
        kci_ci_kwargs = {k: v for k, v in kwargs.items() if k in
                         ['kernelX', 'kernelY', 'kernelZ', 'null_ss', 'approx', 'use_gp', 'est_width', 'polyd',
                          'kwidthx', 'kwidthy', 'kwidthz']}
        self.check_cache_method_consistent(
            'kci', hashlib.md5(json.dumps(kci_ci_kwargs, sort_keys=True).encode('utf-8')).hexdigest())
        self.assert_input_data_is_valid()
        self.kci_ui = KCI_UInd(**kci_ui_kwargs)
        self.kci_ci = KCI_CInd(**kci_ci_kwargs)
        self.max_sample_size = kwargs.get('max_sample_size', int(1e7))

    def __call__(self, X, Y, condition_set=None):
        # Kernel-based conditional independence test.
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        subdata = self.data[random.sample(range(self.sample_size), min(self.sample_size, self.max_sample_size)), :]
        p = self.kci_ui.compute_pvalue(subdata[:, Xs], subdata[:, Ys])[0] if len(condition_set) == 0 else \
            self.kci_ci.compute_pvalue(subdata[:, Xs], subdata[:, Ys], subdata[:, condition_set])[0]
        self.pvalue_cache[cache_key] = p
        return p

class ZeroDel_KCI(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        kci_ui_kwargs = {k: v for k, v in kwargs.items() if k in
                         ['kernelX', 'kernelY', 'null_ss', 'approx', 'est_width', 'polyd', 'kwidthx', 'kwidthy']}
        kci_ci_kwargs = {k: v for k, v in kwargs.items() if k in
                         ['kernelX', 'kernelY', 'kernelZ', 'null_ss', 'approx', 'use_gp', 'est_width', 'polyd',
                          'kwidthx', 'kwidthy', 'kwidthz']}
        self.check_cache_method_consistent(
            zerodel_kci, hashlib.md5(json.dumps(kci_ci_kwargs, sort_keys=True).encode('utf-8')).hexdigest())
        self.assert_input_data_is_valid()
        self.kci_ui = KCI_UInd(**kci_ui_kwargs)
        self.kci_ci = KCI_CInd(**kci_ci_kwargs)
        self.max_sample_size = kwargs.get('max_sample_size', int(1e7))
        self.nonzero_boolean_mat = data != 0
        print('Hey, i am zero del KCI.')

    def _get_index_no_mv_rows(self, var):
        return np.where(np.all(self.nonzero_boolean_mat[:, var], axis=1))[0]

    def __call__(self, X, Y, condition_set=None, return_sample_size_for_debug=False):
        # Kernel-based conditional independence test.
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        var = Xs + Ys + condition_set
        if cache_key in self.pvalue_cache:
            return (self.pvalue_cache[cache_key], min(len(self._get_index_no_mv_rows(var)), self.max_sample_size)) \
                if return_sample_size_for_debug else self.pvalue_cache[cache_key]

        test_wise_deletion_XYcond_rows_index = self._get_index_no_mv_rows(var)
        # print(test_wise_deletion_XYcond_rows_index.shape)
        # if len(test_wise_deletion_XYcond_rows_index) > self.max_sample_size:
            # test_wise_deletion_XYcond_rows_index = random.sample(list(test_wise_deletion_XYcond_rows_index), self.max_sample_size)

        N_test_wise_deleted = len(test_wise_deletion_XYcond_rows_index)
        if N_test_wise_deleted <= len(condition_set) + 3:   # here d+3 is just a heuristic number; not strictly following fisherz
            warnings.warn('Too few samples after testwise deletion to run fci test. Please check your data. Set pval=1.')
            return (1., N_test_wise_deleted) if return_sample_size_for_debug else 1.
        subdata = self.data[test_wise_deletion_XYcond_rows_index][:, var]
        p = self.kci_ui.compute_pvalue(subdata[:, [0]], subdata[:, [1]])[0] if len(condition_set) == 0 else \
            self.kci_ci.compute_pvalue(subdata[:, [0]], subdata[:, [1]], subdata[:, 2:])[0]
        self.pvalue_cache[cache_key] = p
        return (p, N_test_wise_deleted) if return_sample_size_for_debug else p


class Chisq_or_Gsq(CIT_Base):
    def __init__(self, data, method_name, **kwargs):
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]
        assert method_name in ['chisq', 'gsq']
        super().__init__(np.apply_along_axis(_unique, 0, data).astype(np.int64), **kwargs)
        self.check_cache_method_consistent(method_name, NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid()
        self.cardinalities = np.max(self.data, axis=0) + 1

    def chisq_or_gsq_test(self, dataSXY, cardSXY, G_sq=False):
        """by Haoyue@12/18/2021
        Parameters
        ----------
        dataSXY: numpy.ndarray, in shape (|S|+2, n), where |S| is size of conditioning set (can be 0), n is sample size
                 dataSXY.dtype = np.int64, and each row has values [0, 1, 2, ..., card_of_this_row-1]
        cardSXY: cardinalities of each row (each variable)
        G_sq: True if use G-sq, otherwise (False by default), use Chi_sq
        """
        def _Fill2DCountTable(dataXY, cardXY):
            """
            e.g. dataXY: the observed dataset contains 5 samples, on variable x and y they're
                x: 0 1 2 3 0
                y: 1 0 1 2 1
            cardXY: [4, 3]
            fill in the counts by index, we have the joint count table in 4 * 3:
                xy| 0 1 2
                --|-------
                0 | 0 2 0
                1 | 1 0 0
                2 | 0 1 0
                3 | 0 0 1
            note: if sample size is large enough, in theory:
                    min(dataXY[i]) == 0 && max(dataXY[i]) == cardXY[i] - 1
                however some values may be missed.
                also in joint count, not every value in [0, cardX * cardY - 1] occurs.
                that's why we pass cardinalities in, and use `minlength=...` in bincount
            """
            cardX, cardY = cardXY
            xyIndexed = dataXY[0] * cardY + dataXY[1]
            xyJointCounts = np.bincount(xyIndexed, minlength=cardX * cardY).reshape(cardXY)
            xMarginalCounts = np.sum(xyJointCounts, axis=1)
            yMarginalCounts = np.sum(xyJointCounts, axis=0)
            return xyJointCounts, xMarginalCounts, yMarginalCounts

        def _Fill3DCountTableByBincount(dataSXY, cardSXY):
            cardX, cardY = cardSXY[-2:]
            cardS = np.prod(cardSXY[:-2])
            cardCumProd = np.ones_like(cardSXY)
            cardCumProd[:-1] = np.cumprod(cardSXY[1:][::-1])[::-1]
            SxyIndexed = np.dot(cardCumProd[None], dataSXY)[0]

            SxyJointCounts = np.bincount(SxyIndexed, minlength=cardS * cardX * cardY).reshape((cardS, cardX, cardY))
            SMarginalCounts = np.sum(SxyJointCounts, axis=(1, 2))
            SMarginalCountsNonZero = SMarginalCounts != 0
            SMarginalCounts = SMarginalCounts[SMarginalCountsNonZero]
            SxyJointCounts = SxyJointCounts[SMarginalCountsNonZero]

            SxJointCounts = np.sum(SxyJointCounts, axis=2)
            SyJointCounts = np.sum(SxyJointCounts, axis=1)
            return SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts

        def _Fill3DCountTableByUnique(dataSXY, cardSXY):
            # Sometimes when the conditioning set contains many variables and each variable's cardinality is large
            # e.g. consider an extreme case where
            # S contains 7 variables and each's cardinality=20, then cardS = np.prod(cardSXY[:-2]) would be 1280000000
            # i.e., there are 1280000000 different possible combinations of S,
            #    so the SxyJointCounts array would be of size 1280000000 * cardX * cardY * np.int64,
            #    i.e., ~3.73TB memory! (suppose cardX, cardX are also 20)
            # However, samplesize is usually in 1k-100k scale, far less than cardS,
            # i.e., not all (and actually only a very small portion of combinations of S appeared in data)
            #    i.e., SMarginalCountsNonZero in _Fill3DCountTable_by_bincount is a very sparse array
            # So when cardSXY is large, we first re-index S (skip the absent combinations) and then count XY table for each.
            # See https://github.com/cmu-phil/causal-learn/pull/37.
            cardX, cardY = cardSXY[-2:]
            cardSs = cardSXY[:-2]

            cardSsCumProd = np.ones_like(cardSs)
            cardSsCumProd[:-1] = np.cumprod(cardSs[1:][::-1])[::-1]
            SIndexed = np.dot(cardSsCumProd[None], dataSXY[:-2])[0]

            uniqSIndices, inverseSIndices, SMarginalCounts = np.unique(SIndexed, return_counts=True,
                                                                       return_inverse=True)
            cardS_reduced = len(uniqSIndices)
            SxyIndexed = inverseSIndices * cardX * cardY + dataSXY[-2] * cardY + dataSXY[-1]
            SxyJointCounts = np.bincount(SxyIndexed, minlength=cardS_reduced * cardX * cardY).reshape(
                (cardS_reduced, cardX, cardY))

            SxJointCounts = np.sum(SxyJointCounts, axis=2)
            SyJointCounts = np.sum(SxyJointCounts, axis=1)
            return SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts

        def _Fill3DCountTable(dataSXY, cardSXY):
            # about the threshold 1e5, see a rough performance example at:
            # https://gist.github.com/MarkDana/e7d9663a26091585eb6882170108485e#file-count-unique-in-array-performance-md
            if np.prod(cardSXY) < CONST_BINCOUNT_UNIQUE_THRESHOLD: return _Fill3DCountTableByBincount(dataSXY, cardSXY)
            return _Fill3DCountTableByUnique(dataSXY, cardSXY)

        def _CalculatePValue(cTables, eTables):
            """
            calculate the rareness (pValue) of an observation from a given distribution with certain sample size.

            Let k, m, n be respectively the cardinality of S, x, y. if S=empty, k==1.
            Parameters
            ----------
            cTables: tensor, (k, m, n) the [c]ounted tables (reflect joint P_XY)
            eTables: tensor, (k, m, n) the [e]xpected tables (reflect product of marginal P_X*P_Y)
              if there are zero entires in eTables, zero must occur in whole rows or columns.
              e.g. w.l.o.g., row eTables[w, i, :] == 0, iff np.sum(cTables[w], axis=1)[i] == 0, i.e. cTables[w, i, :] == 0,
                   i.e. in configuration of conditioning set == w, no X can be in value i.

            Returns: pValue (float in range (0, 1)), the larger pValue is (>alpha), the more independent.
            -------
            """
            eTables_zero_inds = eTables == 0
            eTables_zero_to_one = np.copy(eTables)
            eTables_zero_to_one[eTables_zero_inds] = 1  # for legal division

            if G_sq == False:
                sum_of_chi_square = np.sum(((cTables - eTables) ** 2) / eTables_zero_to_one)
            else:
                div = np.divide(cTables, eTables_zero_to_one)
                div[div == 0] = 1  # It guarantees that taking natural log in the next step won't cause any error
                sum_of_chi_square = 2 * np.sum(cTables * np.log(div))

            # array in shape (k,), zero_counts_rows[w]=c (0<=c<m) means layer w has c all-zero rows
            zero_counts_rows = eTables_zero_inds.all(axis=2).sum(axis=1)
            zero_counts_cols = eTables_zero_inds.all(axis=1).sum(axis=1)
            sum_of_df = np.sum((cTables.shape[1] - 1 - zero_counts_rows) * (cTables.shape[2] - 1 - zero_counts_cols))
            return 1 if sum_of_df == 0 else chi2.sf(sum_of_chi_square, sum_of_df)

        if len(cardSXY) == 2:  # S is empty
            xyJointCounts, xMarginalCounts, yMarginalCounts = _Fill2DCountTable(dataSXY, cardSXY)
            xyExpectedCounts = np.outer(xMarginalCounts, yMarginalCounts) / dataSXY.shape[1]  # divide by sample size
            return _CalculatePValue(xyJointCounts[None], xyExpectedCounts[None])

        # else, S is not empty: conditioning
        SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts = _Fill3DCountTable(dataSXY, cardSXY)
        SxyExpectedCounts = SxJointCounts[:, :, None] * SyJointCounts[:, None, :] / SMarginalCounts[:, None, None]
        return _CalculatePValue(SxyJointCounts, SxyExpectedCounts)

    def __call__(self, X, Y, condition_set=None):
        # Chi-square (or G-square) independence test.
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        indexs = condition_set + Xs + Ys
        p = self.chisq_or_gsq_test(self.data[:, indexs].T, self.cardinalities[indexs], G_sq=self.method == 'gsq')
        self.pvalue_cache[cache_key] = p
        return p


class D_Separation(CIT_Base):
    def __init__(self, data, true_dag=None, **kwargs):
        '''
        Use d-separation as CI test, to ensure the correctness of constraint-based methods. (only used for tests)
        Parameters
        ----------
        data:   numpy.ndarray, just a placeholder, not used in D_Separation
        true_dag:   nx.DiGraph object, the true DAG
        '''
        super().__init__(data, **kwargs)  # data is just a placeholder, not used in D_Separation
        self.check_cache_method_consistent('d_separation', NO_SPECIFIED_PARAMETERS_MSG)
        self.true_dag = true_dag
        import networkx as nx; global nx
        # import networkx here violates PEP8; but we want to prevent unnecessary import at the top (it's only used here)

    def __call__(self, X, Y, condition_set=None):
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        p = float(nx.d_separated(self.true_dag, {Xs[0]}, {Ys[0]}, set(condition_set)))
        # pvalue is bool here: 1 if is_d_separated and 0 otherwise. So heuristic comparison-based uc_rules will not work.

        # here we use networkx's d_separation implementation.
        # an alternative is to use causal-learn's own d_separation implementation in graph class:
        #   self.true_dag.is_dseparated_from(
        #       self.true_dag.nodes[Xs[0]], self.true_dag.nodes[Ys[0]], [self.true_dag.nodes[_] for _ in condition_set])
        #   where self.true_dag is an instance of GeneralGrpah class.
        # I have checked the two implementations: they are equivalent (when the graph is DAG),
        # and generally causal-learn's implementation is faster.
        # but just for now, I still use networkx's, for two reasons:
        # 1. causal-learn's implementation sometimes stops working during run (haven't check detailed reasons)
        # 2. GeneralGraph class will be hugely refactored in the near future.
        self.pvalue_cache[cache_key] = p
        return p
