import pickle

import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, data_dir, field_list):
        """
        加载数据，返回 DataFrame
        :param file_paths: 文件路径列表
        :return: pd.DataFrame
        """
        for field in field_list:
            with open(data_dir + field + ".dt", "rb") as f:
                length = np.frombuffer(f.read(8), dtype=np.int64)[0]
                meta_dict: dict = pickle.loads(f.read(length))
                shape = (len(meta_dict["index"]), len(meta_dict["columns"]))
                f.seek(length + 8)
                values = np.frombuffer(f.read(), dtype=meta_dict["dtype"]).reshape(
                    shape
                )
                data = pd.DataFrame(
                    values, index=meta_dict["index"], columns=meta_dict["columns"]
                )
                setattr(self, field, data)
