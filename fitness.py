import numpy as np
from scipy.stats import pearsonr, spearmanr


def fitness_ic(individual, toolbox, X_train, y_train):
    """
    计算IC适应度
    :param individual: 个体
    :return: 适应度
    """
    func = toolbox.compile(expr=individual)
    ic = pearsonr(func(**X_train), y_train, axis=1)[0]
    return (np.mean(ic),)


def fitness_rankic(individual, toolbox, X_train, y_train):
    """
    计算RankIC适应度
    :param individual: 个体
    :return: 适应度
    """
    func = toolbox.compile(expr=individual)
    rankic = spearmanr(func(**X_train), y_train, axis=1)[0]
    return (np.mean(rankic),)


def fitness_icir(individual, toolbox, X_train, y_train):
    """
    计算ICIR适应度
    :param individual: 个体
    :return: 适应度
    """
    func = toolbox.compile(expr=individual)
    ic = pearsonr(func(**X_train), y_train, axis=1)[0]
    return (np.mean(ic) / np.std(ic),)


def fitness_rankicir(individual, toolbox, X_train, y_train):
    """
    计算RankICIR适应度
    :param individual: 个体
    :return: 适应度
    """
    func = toolbox.compile(expr=individual)
    rankic = spearmanr(func(**X_train), y_train, axis=1)[0]
    return (np.mean(rankic) / np.std(rankic),)
