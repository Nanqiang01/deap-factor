import gc
import warnings

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from deap_factor import DeapFactor
from fitness import fitness_rankic
from primitive_set import set_pset

# 忽略警告
warnings.filterwarnings("ignore")

# 读取数据
data_dir = "D:/southwall/Desktop/SAIF Study/SMQF/data/raw/ddb/"
field_list = [
    "adjopen",
    "adjhigh",
    "adjlow",
    "adjclose",
    "preclose",
    "volume",
    "pct_change",
    "vwap",
]
label = ["pct_change"]
X = DataLoader(data_dir, field_list)
y = DataLoader(data_dir, label)
logger.info("数据加载完成")

# 切分数据集
X_train = {}
X_test = {}
split_date = "2019-01-01"
# 切分X
for field in field_list:
    df = getattr(X, field)
    X_train[field], X_test[field] = df.loc[:split_date], df.loc[split_date:]
# 切分y
df = getattr(y, label[0])
y_train, y_test = df.loc[:split_date], df.loc[split_date:]
# 删除无用变量
del X, y, df
gc.collect()
logger.info("数据切分完成")

# 设置算子
pset = set_pset(field_list)
logger.info("算子设置完成")

# 初始化deap因子引擎
engine = DeapFactor(pset, fitness_rankic)
engine.set_input(X_train, "X_train")
engine.set_input(y_train, "y_train")
engine.run(pop_size=100, ngen=30)

# 打印最优个体
for i in range(len(engine.hof)):
    logger.info(f"因子表达式：{engine.hof.items[i]}，适应度：{engine.hof.keys[i]}")
logger.complete()
