import gc

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from deap_factor import DeapFactor
from fitness import fitness_rankic
from primitive_set import set_pset

# 读取数据
data_dir = "data/"
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
label = ["vwap"]
X = DataLoader(data_dir, field_list)
y = DataLoader(data_dir, label)
logger.info("数据加载完成")

# 切分数据集
X_train = {}
X_test = {}
# 切分X
for field in field_list:
    df = getattr(X, field)
    X_train[field], X_test[field] = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=False
    )
# 切分y
df = getattr(y, label[0]).shift(1)  # 为了预测未来的vwap，所以将vwap shift一期
y_train, y_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
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
engine.run()
logger.complete([engine.hof[i] for i in range(5)])
