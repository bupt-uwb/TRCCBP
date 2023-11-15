import pandas as pd
import numpy as np
import scipy.io as scio


annotations_file = './label.csv'
labels = pd.read_csv(annotations_file)
label = [labels.iloc[:,1], labels.iloc[:, 2], labels.iloc[:, 3], labels.iloc[:, 4]]  # label为一个四元素列表，分别为目标的id，状态(rest, apnea, sport)，SBP，DBP
scio.savemat('./data/bps.mat', {'bps': np.array(label).transpose()})
# print(np.array(label).transpose())