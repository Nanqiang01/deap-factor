from itertools import repeat

import numpy as np
from deap import gp
from pandas import DataFrame

import operators_nb as op


def set_pset(field_list):
    # 创建一个 primitive set: primitiveset（名称，变量个数）
    pset = gp.PrimitiveSetTyped(
        name="main", in_types=repeat(DataFrame, len(field_list)), ret_type=DataFrame
    )

    # 定义function set：addPrimitive(函数名,参数个数)  不同的函数需要的参数个数不一样
    # 元素算子
    pset.addPrimitive(
        primitive=op.add, in_types=[DataFrame, DataFrame], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.sub, in_types=[DataFrame, DataFrame], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.mul, in_types=[DataFrame, DataFrame], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.div, in_types=[DataFrame, DataFrame], ret_type=DataFrame
    )
    pset.addPrimitive(primitive=op.log, in_types=[DataFrame], ret_type=DataFrame)
    pset.addPrimitive(primitive=op.exp, in_types=[DataFrame], ret_type=DataFrame)
    pset.addPrimitive(primitive=op.sqrt, in_types=[DataFrame], ret_type=DataFrame)
    pset.addPrimitive(primitive=op.pow, in_types=[DataFrame, int], ret_type=DataFrame)
    pset.addPrimitive(primitive=op.abs, in_types=[DataFrame], ret_type=DataFrame)
    pset.addPrimitive(primitive=op.sign, in_types=[DataFrame], ret_type=DataFrame)
    pset.addPrimitive(primitive=op.neg, in_types=[DataFrame], ret_type=DataFrame)
    pset.addPrimitive(primitive=op.inv, in_types=[DataFrame], ret_type=DataFrame)
    pset.addPrimitive(primitive=op.sigmoid, in_types=[DataFrame], ret_type=DataFrame)
    # 时序算子
    pset.addPrimitive(
        primitive=op.ts_delay, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_delta, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_delta_pct, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_sum, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_mean, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_std, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_cov, in_types=[DataFrame, DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_corr, in_types=[DataFrame, DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_rank, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_max, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_min, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_argmax, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_argmin, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_skew, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_kurt, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_median, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_prod, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_mad, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_scale, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_decay_linear, in_types=[DataFrame, int], ret_type=DataFrame
    )
    pset.addPrimitive(
        primitive=op.ts_decay_exp_window, in_types=[DataFrame, int], ret_type=DataFrame
    )
    # 截面算子
    pset.addPrimitive(primitive=op.cs_rank, in_types=[DataFrame], ret_type=DataFrame)
    pset.addPrimitive(
        primitive=op.cs_winsorize, in_types=[DataFrame], ret_type=DataFrame
    )
    # pset.addPrimitive(
    #     primitive=op.cs_regression_neut,
    #     in_types=[DataFrame, DataFrame],
    #     ret_type=DataFrame,
    # )
    # pset.addPrimitive(
    #     primitive=op.cs_regression_proj,
    #     in_types=[DataFrame, DataFrame],
    #     ret_type=DataFrame,
    # )
    # 常数算子
    pset.addPrimitive(op.get1, in_types=[], ret_type=int)
    pset.addPrimitive(op.get5, in_types=[], ret_type=int)
    pset.addPrimitive(op.get10, in_types=[], ret_type=int)
    pset.addPrimitive(op.get20, in_types=[], ret_type=int)
    pset.addPrimitive(op.get60, in_types=[], ret_type=int)
    pset.addPrimitive(op.get122, in_types=[], ret_type=int)
    pset.addPrimitive(op.get244, in_types=[], ret_type=int)

    # 定义terminal set
    periods = [
        1,
        5,
        10,
        20,
        60,
        122,
        244,
    ]
    for period in periods:
        pset.addTerminal(terminal=period, ret_type=int)  # 常数
    # pset.addEphemeralConstant("randint", lambda: np.random.sample(periods, 1), int)

    for i, field in enumerate(field_list):
        pset.renameArguments(**{f"ARG{i}": field})
    return pset
