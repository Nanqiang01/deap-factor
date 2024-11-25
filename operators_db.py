from functools import lru_cache

import numba as nb
import numpy as np
import pandas as pd


# 输入检查
def _check_input(X):
    """检查输入类型"""
    if isinstance(X, pd.DataFrame):
        pass
    else:
        raise ValueError("输入类型错误！")


# 二维输入检查
def _check_shape(X, Y):
    """检查矩阵大小是否一致"""
    if isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
        if np.any(X.index != Y.index) or np.any(X.columns != Y.columns):
            raise ValueError("X和Y的index或columns不一致！")
    else:
        raise ValueError("X和Y的类型不一致！")


########## 元素算子 ##########


def add(X, Y):
    """加法"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _add_nb(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


@nb.njit(nogil=True, cache=True)
def _add_nb(X, Y):
    """加法"""
    return np.add(X, Y)


def sub(X, Y):
    """减法"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _sub_nb(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


@nb.njit(nogil=True, cache=True)
def _sub_nb(X, Y):
    """减法"""
    return np.subtract(X, Y)


def mul(X, Y):
    """乘法"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _mul_nb(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


@nb.njit(nogil=True, cache=True)
def _mul_nb(X, Y):
    """乘法"""
    return np.multiply(X, Y)


def div(X, Y):
    """除法"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _div_nb(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


@nb.njit(nogil=True, cache=True)
def _div_nb(X, Y):
    """除法"""
    return np.divide(X, Y)


def log(X):
    """对数"""
    _check_input(X)
    return pd.DataFrame(_log_nb(X.to_numpy()), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _log_nb(X):
    """对数"""
    return np.log(X)


def exp(X):
    """指数"""
    _check_input(X)
    return pd.DataFrame(_exp_nb(X.to_numpy()), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _exp_nb(X):
    """指数"""
    return np.exp(X)


def sqrt(X):
    """开方"""
    _check_input(X)
    return pd.DataFrame(_sqrt_nb(X.to_numpy()), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _sqrt_nb(X):
    """开方"""
    return np.sqrt(X)


def pow(X, a):
    """幂"""
    _check_input(X)
    return pd.DataFrame(_pow_nb(X.to_numpy(), a), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _pow_nb(X, a):
    """幂"""
    return np.power(X, a)


def abs(X):
    """绝对值"""
    _check_input(X)
    return pd.DataFrame(_abs_nb(X.to_numpy()), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _abs_nb(X):
    """绝对值"""
    return np.abs(X)


def sign(X):
    """指示函数"""
    _check_input(X)
    return pd.DataFrame(_sign_nb(X.to_numpy()), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _sign_nb(X):
    """指示函数"""
    return np.sign(X)


def neg(X):
    """取反"""
    _check_input(X)
    return pd.DataFrame(_neg_nb(X.to_numpy()), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _neg_nb(X):
    """取反"""
    return np.negative(X)


def inv(X):
    """取倒数"""
    _check_input(X)
    return pd.DataFrame(_inv_nb(X.to_numpy()), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _inv_nb(X):
    """取倒数"""
    return np.reciprocal(X)


def sigmoid(X):
    """Sigmoid函数"""
    _check_input(X)
    return pd.DataFrame(_sigmoid_nb(X.to_numpy()), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _sigmoid_nb(X):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-X))


########## 时序算子 ##########


def ts_delay(X, p):
    """滞后p期"""
    _check_input(X)
    return pd.DataFrame(_ts_delay_nb(X.to_numpy(), p), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _ts_delay_nb(X, p):
    """滞后p期"""
    T, N = X.shape
    res = np.full_like(X, np.nan)
    res[p:] = X[: T - p]
    return res


def ts_delta(X, p):
    """一阶差分"""
    _check_input(X)
    return pd.DataFrame(_ts_delta_nb(X.to_numpy(), p), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _ts_delta_nb(X, p):
    """一阶差分"""
    T, N = X.shape
    res = np.full_like(X, np.nan)
    res[p:] = X[p:] - X[: T - p]
    return res


def ts_delta_pct(X, p):
    """一阶差分（百分比）"""
    _check_input(X)
    return pd.DataFrame(
        _ts_delta_pct_nb(X.to_numpy(), p), index=X.index, columns=X.columns
    )


@nb.njit(nogil=True, cache=True)
def _ts_delta_pct_nb(X, p):
    """一阶差分（百分比）"""
    T, N = X.shape
    res = np.full_like(X, np.nan)
    res[p:] = (X[p:] - X[: T - p]) / X[: T - p]
    return res


def ts_sum(X, p):
    """过去p期求和"""
    _check_input(X)
    return X.rolling(p).sum(engine="numba")


def ts_mean(X, p):
    """过去p期均值"""
    _check_input(X)
    return X.rolling(p).mean(engine="numba")


def ts_std(X, p):
    """过去p期标准差"""
    _check_input(X)
    return X.rolling(p).std(engine="numba")


@lru_cache
def ts_cov(X, Y, p):
    """过去p期协方差"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return X.rolling(p).cov(Y)


@lru_cache
def ts_corr(X, Y, p):
    """过去p期相关系数"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return X.rolling(p).corr(Y)


@lru_cache
def ts_rank(X, p):
    """过去p期排序"""
    _check_input(X)
    return X.rolling(p).rank()


def ts_max(X, p):
    """过去p期最大值"""
    _check_input(X)
    return X.rolling(p).max(engine="numba")


def ts_min(X, p):
    """过去p期最小值"""
    _check_input(X)
    return X.rolling(p).min(engine="numba")


def ts_argmax(X, p):
    """过去p期最大值位置"""
    _check_input(X)
    return pd.DataFrame(
        _ts_argmax_nb(X.to_numpy(), p), index=X.index, columns=X.columns
    )


@nb.njit(nogil=True, cache=True)
def _ts_argmax_nb(X, p):
    """过去p期最大值位置"""
    T, N = X.shape
    res = np.full_like(X, np.nan)
    for col in range(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(np.isnan(x_array)) < p / 5:
                max_idx = np.argmax(x_array)
                res[t - 1, col] = t - p + max_idx
    return res


def ts_argmin(X, p):
    """过去p期最小值位置"""
    _check_input(X)
    return pd.DataFrame(
        _ts_argmin_nb(X.to_numpy(), p), index=X.index, columns=X.columns
    )


@nb.njit(nogil=True, cache=True)
def _ts_argmin_nb(X, p):
    """过去p期最小值位置"""
    T, N = X.shape
    res = np.full_like(X, np.nan)
    for col in range(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(np.isnan(x_array)) < p / 5:
                min_idx = np.argmin(x_array)
                res[t - 1, col] = t - p + min_idx
    return res


@lru_cache
def ts_skew(X, p):
    """过去p期偏度"""
    _check_input(X)
    return X.rolling(p).skew()


@lru_cache
def ts_kurt(X, p):
    """过去p期峰度"""
    _check_input(X)
    return X.rolling(p).kurt()


def ts_median(X, p):
    """过去p期中位数"""
    _check_input(X)
    return X.rolling(p).median(engine="numba")


def ts_prod(X, p):
    """过去p期乘积"""
    _check_input(X)
    return pd.DataFrame(_ts_prod_nb(X.to_numpy(), p), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _ts_prod_nb(X, p):
    """过去p期乘积"""
    T, N = X.shape
    res = np.full_like(X, np.nan)
    for col in range(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(np.isnan(x_array)) < p / 5:
                res[t - 1, col] = np.nanprod(x_array)
    return res


def ts_mad(X, p):
    """过去p期绝对中位差"""
    _check_input(X)
    return pd.DataFrame(_ts_mad_nb(X.to_numpy(), p), index=X.index, columns=X.columns)


@nb.njit(nogil=True, cache=True)
def _ts_mad_nb(X, p):
    """过去p期绝对中位差"""
    T, N = X.shape
    res = np.full_like(X, np.nan)
    for col in range(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(np.isnan(x_array)) < p / 5:
                res[t - 1, col] = np.nanmedian(np.abs(x_array - np.nanmedian(x_array)))
    return res


def ts_scale(X, p, constant=0):
    """过去p期标准化"""
    _check_input(X)
    return pd.DataFrame(
        _ts_scale_nb(X.to_numpy(), p, constant), index=X.index, columns=X.columns
    )


@nb.njit(nogil=True, cache=True)
def _ts_scale_nb(X, p, constant):
    """过去p期标准化"""
    T, N = X.shape
    res = np.full_like(X, np.nan)
    for col in range(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(~np.isnan(x_array)) > p / 5:
                x_min = np.nanmin(x_array)
                x_max = np.nanmax(x_array)
                res[t - 1, col] = ((x_array[-1] - x_min) / (x_max - x_min)) + constant
    return res


def ts_decay_linear(X, p):
    """过去p期线性加权"""
    _check_input(X)
    return pd.DataFrame(
        _ts_decay_linear_nb(X.to_numpy(), p), index=X.index, columns=X.columns
    )


@nb.njit(nogil=True, cache=True)
def _ts_decay_linear_nb(X, p):
    """过去p期线性加权"""
    T, N = X.shape
    res = np.full_like(X, np.nan)
    weights = np.arange(1, p + 1)
    weights = weights / np.sum(weights)

    for i in range(N):
        for t in range(p, T + 1):
            arr = X[t - p : t, i]
            res[t - 1, i] = np.dot(arr, weights[::-1])

    return res


def ts_decay_exp_window(X, p):
    """span为p的指数加权"""
    _check_input(X)
    return pd.DataFrame(
        _ts_decay_exp_window_nb(X.to_numpy(), p), index=X.index, columns=X.columns
    )


@nb.njit(nogil=True, cache=True)
def _ts_decay_exp_window_nb(X, p):
    """span为p的指数加权"""
    T, N = X.shape
    res = np.full_like(X, np.nan)
    alpha = 2 / (p + 1)
    for col in range(N):
        valid_count = 0
        last_avg = np.nan
        denominator = 1

        for t in range(T):
            value = X[t, col]
            if not np.isnan(value):
                valid_count += 1

                if valid_count == 1:
                    last_avg = value
                    denominator = 1
                else:
                    denominator = denominator * (1 - alpha) + 1
                    last_avg = (1 - alpha) * last_avg + value

                res[t, col] = last_avg / denominator
            else:
                res[t, col] = np.nan

    return res


########## 截面算子 ##########


@lru_cache
def cs_rank(X):
    """截面排序"""
    _check_input(X)
    return X.rank(axis=1, pct=True, method="average")


def cs_winsorize(X, n):
    """截面去极值"""
    _check_input(X)
    return pd.DataFrame(
        _cs_winsorize_nb(X.to_numpy(), n), index=X.index, columns=X.columns
    )


@nb.njit(nogil=True, cache=True)
def _cs_winsorize_nb(X, n=3):
    """截面去极值"""
    T, N = X.shape
    res = np.full_like(X, np.nan)
    for t in range(T):
        valid_idx = ~np.isnan(X[t])
        x_array = X[t, valid_idx]
        # 使用3MAD法去极值
        median = np.nanmedian(x_array)
        mad = np.nanmedian(np.abs(x_array - median))
        upper = median + n * 1.4826 * mad
        lower = median - n * 1.4826 * mad
        res[t, valid_idx] = np.clip(x_array, lower, upper)
    return res


def cs_regression_neut(X, Y):
    """截面回归残差"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _cs_regression_neut_nb(X.to_numpy(), Y.to_numpy()),
        index=X.index,
        columns=X.columns,
    )


@nb.njit(nogil=True, cache=True)
def _cs_regression_neut_nb(X, Y):
    """截面回归残差"""
    T, N = Y.shape
    res = np.full_like(X, np.nan)
    for t in range(T):
        y_row = Y[t, :]
        x_row = X[t, :]
        valid_y = ~np.isnan(y_row)
        valid_x = ~np.isnan(x_row)
        if np.count_nonzero(valid_y) > 1 and np.count_nonzero(valid_x) > 1:
            y = y_row[valid_y]
            x = x_row[valid_x]
            y_demean = y - np.nanmean(y)
            x_demean = x - np.nanmean(x)
            beta = np.nansum(y_demean * x_demean) / np.nansum(x_demean**2)
            res[t, valid_y] = y_demean - beta * x_demean

    return res


def cs_regression_proj(X, Y):
    """截面回归残差"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _cs_regression_proj_nb(X.to_numpy(), Y.to_numpy()),
        index=X.index,
        columns=X.columns,
    )


@nb.njit(nogil=True, cache=True)
def _cs_regression_proj_nb(X, Y):
    """截面回归残差"""
    T, N = Y.shape
    res = np.full_like(X, np.nan)
    for t in range(T):
        y_row = Y[t, :]
        x_row = X[t, :]
        valid_y = ~np.isnan(y_row)
        valid_x = ~np.isnan(x_row)
        if np.count_nonzero(valid_y) > 2 and np.count_nonzero(valid_x) > 2:
            y = y_row[valid_y]
            x = x_row[valid_x]
            y_demean = y - np.nanmean(y)
            x_demean = x - np.nanmean(x)
            beta = np.nansum(y_demean * x_demean) / np.nansum(x_demean**2)
            res[t, valid_y] = np.nanmean(y) + beta * x_demean

    return res
