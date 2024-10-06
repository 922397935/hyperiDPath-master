import torch
import numpy as np
from torch.utils.data import Dataset


"""
这个pathDataset继承自torch的Dataset数据集，构建深度学习数据集
"""


class PathDataset(Dataset):
    def __init__(self, drug_indication_array, total_path_dict, type_dict,
                       max_path_length=10, max_path_num=10, rng=None):
        """
        Args:
            drug_indication_array: 药物适应症关联作用
            total_path_dict:
            type_dict:
            max_path_length: 最大路径长度, 8
            max_path_num: 最大路径数目, 256
            rng: torch.dataset的随机数
        """
        self.drug_indication_array = drug_indication_array
        self.total_path_dict = total_path_dict
        self.type_dict = type_dict
        self.max_path_length = max_path_length
        self.max_path_num = max_path_num
        self.rng = rng

    def __len__(self):
        return len(self.drug_indication_array)

    def __getitem__(self, index):
        drug, indication, label = self.drug_indication_array[index]
        path_list = self.total_path_dict[tuple([drug, indication, label])]
        path_array_list = []
        type_array_list = []
        lengths_list = []
        mask_list = []
        for path in path_list:
            path = path[:self.max_path_length]
            pad_num = max(0, self.max_path_length - len(path))
            path_array_list.append(path + [0]*pad_num)
            type_array_list.append([self.type_dict[n] for n in path]+[0]*pad_num)
            lengths_list.append(len(path))
            mask_list.append([1]*len(path)+[0]*pad_num)
        replace = len(path_array_list) < self.max_path_num
        select_idx_list = [idx for idx in self.rng.choice(len(path_array_list), size=self.max_path_num, replace=replace)]
        path_array = np.array([path_array_list[idx] for idx in select_idx_list])
        type_array = np.array([type_array_list[idx] for idx in select_idx_list])
        lengths_array = np.array([lengths_list[idx] for idx in select_idx_list])
        mask_array = np.array([mask_list[idx] for idx in select_idx_list])

        path_feature = torch.from_numpy(path_array).type(torch.LongTensor)
        type_feature = torch.from_numpy(type_array).type(torch.LongTensor)
        label = torch.from_numpy(np.array([label])).type(torch.FloatTensor)
        lengths = torch.from_numpy(lengths_array).type(torch.LongTensor)
        mask = torch.from_numpy(mask_array).type(torch.ByteTensor)

        return drug, indication, path_feature, type_feature, lengths, mask, label

