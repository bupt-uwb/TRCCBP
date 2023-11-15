from torch.utils.data import Dataset
import scipy.io as sio
import torch
import numpy as np


class LoadDataset(Dataset):
    def __init__(self, dataset):
        super(LoadDataset, self).__init__()

        data = dataset['signals']
        label = dataset['label']

        self.x_data = data.reshape(data.shape[0], 1, data.shape[1]).astype(np.float32)
        # print(self.x_data.dtype)
        self.label = label

        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        feature = self.x_data[index] / 5.5
        label = self.label[index]
        # 坑：Cell数组读取标签时提取数据需要做的索引处理
        id = label[0][0][0]
        # print(id)
        status = label[1][0]
        gt = np.array([label[2][0][0], label[3][0][0]])
        # print(gt.dtype)
        dict_data = {
            "feature": feature,
            "gt": gt,
            "status": status,
            "id": id
        }
        # print(dict_data)
        return dict_data

    def __len__(self):
        return self.len
