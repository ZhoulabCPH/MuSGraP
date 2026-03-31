# Base / Native
import math
import os
import pickle
import re
import lifelines.exceptions
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from lifelines.utils import concordance_index
# Numerical / Array
# from lifelines.utils import concordance_index
# from lifelines.statistics import logrank_test
import matplotlib as mpl
import numpy as np
mpl.rcParams['axes.linewidth'] = 3 #set the value globally

# Torch
import torch
import torch.nn as nn
from torch.nn import init, Parameter
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate
# import torch_geometric
# from torch_geometric.data import Batch


def CoxLoss(survtime, censor, hazard_pred, device='cuda'):
    # 生成一个掩码，标记哪些样本是 NaN
    nan_mask = torch.isnan(survtime) | torch.isnan(censor) | torch.isnan(hazard_pred)

    # 只计算非 NaN 样本
    survtime = survtime[~nan_mask]  # 取出非 NaN 的样本
    censor = censor[~nan_mask]
    hazard_pred = hazard_pred[~nan_mask]

    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)


def CIndex(hazards, labels, survtime_all):
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazards[j] < hazards[i]: concord += 1
                    elif hazards[j] < hazards[i]: concord += 0.5

    return(concord/total)


# def CIndex_lifeline(hazards, labels, survtime_all):
#     return(concordance_index(survtime_all, -hazards, labels))
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
import pandas as pd
def coxph_log_rank(survtime_all, labels, covariates):
    """
    Calculate the log-rank p-value and 95% CI for the hazard ratio (HR) using Cox Proportional Hazards model.

    Parameters:
    survtime_all (array-like): The survival times for each sample.
    labels (array-like): The event status (1 if the event occurred, 0 if censored).
    covariates (pd.DataFrame): A dataframe of covariates/features to be used for the Cox model.

    Returns:
    p_value (float): The log-rank p-value from the comparison of groups.
    hr (float): The hazard ratio from the Cox model.
    hr_lower (float): The lower bound of the 95% confidence interval of the hazard ratio.
    hr_upper (float): The upper bound of the 95% confidence interval of the hazard ratio.
    """
    data=pd.DataFrame({'covariates':covariates.astype(np.int64),
                  'survtime':survtime_all,
                  'event':labels.astype(np.int64)})
    # Fit Cox Proportional Hazards model
    try:
        cph = CoxPHFitter()
        cph.fit(data, duration_col='survtime', event_col='event')

        # Extract hazard ratio and 95% CI
        hr = cph.hazard_ratios_[0]  # HR for the first covariate (adjust as needed for multiple covariates)
        cov_name = cph.hazard_ratios_.index.tolist()
        # 提取置信区间
        coef_lower = cph.confidence_intervals_.loc[cov_name[0], '95% lower-bound']
        coef_upper = cph.confidence_intervals_.loc[cov_name[0], '95% upper-bound']

        # 计算 HR 的置信区间上下限
        hr_lower = np.exp(coef_lower)  # exp(coef lower 95%)
        hr_upper = np.exp(coef_upper)  # exp(coef upper 95%)

        summary = cph.summary
        # print(summary[['coef', 'exp(coef)', 'p']])  # 打印系数、指数系数和 p 值
        p_value=summary[['p']].values[0][0]

        # 计算 C-index
        c_index = cph.concordance_index_
    except lifelines.exceptions.ConvergenceError:
        c_index, p_value, hr, hr_lower, hr_upper=0,0,0,0,0



    # from lifelines import KaplanMeierFitter
    # import matplotlib.pyplot as plt
    # kmf = KaplanMeierFitter()
    # # 创建两个组：covariates 为 0 和 covariates 为 1
    # group1 = data[data['covariates'] == 0]
    # group2 = data[data['covariates'] == 1]
    # # 绘制第一组（covariates == 0）的 Kaplan-Meier 曲线
    # kmf.fit(group1['survtime'], event_observed=group1['event'], label='Group 0')
    # ax = kmf.plot()  # 这次返回一个坐标轴 ax
    # # 绘制第二组（covariates == 1）的 Kaplan-Meier 曲线
    # kmf.fit(group2['survtime'], event_observed=group2['event'], label='Group 1')
    # kmf.plot(ax=ax)  # 传递相同的 ax 给第二次绘制
    # # 添加图表标题和标签
    # plt.title('Kaplan-Meier Survival Curves by Covariate Group')
    # plt.xlabel('Time')
    # plt.ylabel('Survival Probability')
    # # 显示图形
    # plt.show()

    return c_index,p_value, hr, hr_lower, hr_upper




