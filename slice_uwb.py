import scipy.io as scio
import pca_filter
import numpy as np
import os


fileDir = '../20220426_rawdata/'
UWBName = []
for i in os.listdir(fileDir):
    if os.path.splitext(i)[1] == '.mat':
        UWBName.append(i)

for ii in range(len(UWBName)):
    dataFile = fileDir + UWBName[ii]
    data = scio.loadmat(dataFile)
    data = data['data']
    RawData = data[:, 0:436].copy()
    PureData = pca_filter.p_f(RawData, 30, 36)
    PureData = PureData[:,0:120]
    for j in range(2):
        Puredata = PureData[(j*300):(j+1)*300,:]
        scio.savemat('../20220509_slices/'+UWBName[ii][:-4]+'_'+str(j+1)+'.mat', {'Puredata': Puredata})