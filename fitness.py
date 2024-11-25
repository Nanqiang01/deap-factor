import numpy as np
from scipy.stats import pearsonr, spearmanr


def fitness_ic(toolbox, individual):
    """
    计算IC适应度
    :param individual: 个体
    :return: 适应度
    """
    func = toolbox.compile(expr=individual)
    predictions = func(toolbox.X_train)
    ic = pearsonr(predictions, toolbox.y_train)[0]
    return np.mean(ic)


def fitness_rankic(toolbox, individual):
    """
    计算RankIC适应度
    :param individual: 个体
    :return: 适应度
    """
    func = toolbox.compile(expr=individual)
    predictions = func(toolbox.X_train)
    rankic = spearmanr(predictions, toolbox.y_train)[0]
    return np.mean(rankic)


def fitness_icir(toolbox, individual):
    """
    计算ICIR适应度
    :param individual: 个体
    :return: 适应度
    """
    func = toolbox.compile(expr=individual)
    predictions = func(toolbox.X_train)
    ic = pearsonr(predictions, toolbox.y_train)[0]
    return np.mean(ic) / np.std(ic)


def fitness_rankicir(toolbox, individual):
    """
    计算RankICIR适应度
    :param individual: 个体
    :return: 适应度
    """
    func = toolbox.compile(expr=individual)
    predictions = func(toolbox.X_train)
    rankic = spearmanr(predictions, toolbox.y_train)[0]
    return np.mean(rankic) / np.std(rankic)
