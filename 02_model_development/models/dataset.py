
import os
import torchvision
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import h5py
from torch import nn, optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageFilter
is_amp = True
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from augmentation import *
import random
from torch.utils.data import random_split

random.seed(42)


def adjust_matrix(matrix, target_rows=3000):
    # 获取当前矩阵的行数
    current_rows = matrix.shape[0]

    if current_rows < target_rows:
        # 行数不足，生成随机数据并补充
        additional_rows = target_rows - current_rows
        # 生成 additional_rows 个随机的 1x1024 向量
        random_data = torch.randint(0, 2, (additional_rows, matrix.shape[1])).float()
        # 将原矩阵与生成的随机矩阵拼接
        matrix = torch.cat((matrix, random_data), dim=0)

    elif current_rows > target_rows:
        # 行数超过，随机抽取 target_rows 行
        indices = torch.randint(0, current_rows, (target_rows,))
        matrix = matrix[indices]

    # 返回处理后的矩阵
    return matrix
# 读取 NIfTI 文件
def load_ct_image(file_path: str):
    # 使用 SimpleITK 读取 NIfTI 格式的图像
    image = sitk.ReadImage(file_path)
    # 将 SimpleITK 图像转化为 NumPy 数组
    image_array = sitk.GetArrayFromImage(image)  # 结果形状为 (D, H, W)
    # 转换为 [C, D, H, W] 形式，并加入一个维度表示通道数
    # 假设是单通道 CT 图像，如果是多通道（比如 CT + ROI），你需要调整
    image_array = image_array # 添加一个通道维度 [C, D, H, W]
    # 将数据类型转换为 float32（常用于深度学习）
    image_array = image_array.astype(np.float32)
    # 将 NumPy 数组转换为 PyTorch Tensor
    image_tensor = torch.tensor(image_array)
    return image_tensor
import SimpleITK as sitk
import os

def make_big_model_feature_Fundation(arg):

    Discovery=torch.load(arg.fundation_path_feature_CHCAMS)
    #按照训练和验证的划分，将图像也划分
    Patients=[name.split('_')[0] for name in Discovery['Patch_name']]

    #读取训练/验证集合
    CHCAMS_Train_Clincial=pd.read_csv(arg.Train_cohort)
    train_patients=np.array(CHCAMS_Train_Clincial['PatientID'].values,dtype=np.str_)
    CHCAMS_Val_Clincial = pd.read_csv(arg.Val_cohort)
    val_patients = np.array(CHCAMS_Val_Clincial['PatientID'].values,dtype=np.str_)
    Train_indices = {x: [i for i, val in enumerate(Patients) if val == x] for x in train_patients}
    Val_indices = {x: [i for i, val in enumerate(Patients) if val == x] for x in val_patients}



    ################TMUGH队列################
    TMUGH_Foundations = torch.load(arg.fundation_path_feature_TMUGH)
    TMUGH_Patients=[name.split('_')[0] for name in TMUGH_Foundations['Patch_name']]

    TMUGH_Clincial = pd.read_csv(arg.TMUGH_cohort)
    TMUGH_patients = np.array(TMUGH_Clincial['PatientID'].values,dtype=np.str_)
    TMUGH_indices = {x: [i for i, val in enumerate(TMUGH_Patients) if val == x] for x in TMUGH_patients}


    ################HMUCH队列################
    HMUCH_Foundations = torch.load(arg.fundation_path_feature_HMUCH)
    HMUCH_Patients=[name.split('_')[0] for name in HMUCH_Foundations['Patch_name']]
    HMUCH_Clincial = pd.read_csv(arg.HMUCH_cohort)
    HMUCH_patients = np.array(HMUCH_Clincial['PatientID'].values,dtype=np.str_)
    HMUCH_indices = {x: [i for i, val in enumerate(HMUCH_Patients) if val == x] for x in HMUCH_patients}



    Cohorts={'train_indices':Train_indices,
             'val_indices':Val_indices,
             'TMUGH_indices': TMUGH_indices,
             'HMUCH_indices': HMUCH_indices,

             'CHCAMS_feature': torch.tensor(Discovery['feature']),
             'TMUGH_feature': torch.tensor(TMUGH_Foundations['feature']),
             'HMUCH_feature': torch.tensor(HMUCH_Foundations['feature']),

             'CHCAMS_patch_name':Discovery['Patch_name'],
             'TMUGH_patch_name': TMUGH_Foundations['Patch_name'],
             'HMUCH_patch_name': HMUCH_Foundations['Patch_name'],

             'train_clincial':CHCAMS_Train_Clincial,
             'val_clincial': CHCAMS_Val_Clincial,
             'TMUGH_clincial': TMUGH_Clincial,
             'HMUCH_clincial': HMUCH_Clincial,
             'dim':Discovery['feature'][0].shape[0],
             }
    return Cohorts





class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomRotation(
                degrees=90,
                resample=False,
                expand=False,
                center=None,
                fill=255,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.4),
            Solarization(p=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            # transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomRotation(
                degrees=90,
                resample=False,
                expand=False,
                center=None,
                fill=255,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

class Transform_:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        return y1,y1

def get_percent_subset(dataset,ratio):
    DF_Length = len(list(dataset.items()))
    Sizes = int(DF_Length * ratio)
    random_numbers = random.sample(range(0, DF_Length), Sizes)
    selected_elements = list([list(dataset.items())[i] for i in random_numbers])
    return selected_elements

class SCLCDataset(Dataset):
    def __init__(self,df,WSI_feature,name,clincials,args,ratio=1):
        self.df=get_percent_subset(df,ratio)
        self.args=args
        self.name=np.array(name)
        self.clincial=clincials
        self.Data_WSI=WSI_feature
        self.length = len(self.df)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        #1.随机选取的patient的 index
        patient_name, patient_patch_id = self.df[index]
        index_clincial = self.clincial[self.clincial['PatientID'] == np.int64(patient_name)]
        OS = index_clincial['OS'].values[0]
        try:
            OSState = np.array(index_clincial['OSState'].values[0],dtype=np.int64)
        except ValueError:
            OSState=index_clincial['OSStatus'].values[0]
        DFS = index_clincial['DFS'].values[0]
        DFSState = index_clincial['DFSState'].values[0]

        WSI_feature = adjust_matrix(self.Data_WSI[patient_patch_id], self.args.N)


        r = {}
        r['index'] = torch.tensor(index)
        r['patch_name']=self.name[patient_patch_id]
        r['patient_name'] = torch.tensor(np.int64(patient_name))
        r['OS']=torch.tensor(OS)
        r['OSState'] = torch.tensor(OSState)
        r['DFS'] = torch.tensor(DFS)
        r['DFSState'] = torch.tensor(DFSState)
        r['WSI_feature']=WSI_feature

        return r

class SCLCDataset_Val(Dataset):
    def __init__(self,df,WSI_feature,name,clincials,args,ratio=1):
        self.df=get_percent_subset(df,ratio)
        self.args=args
        self.name=np.array(name)
        self.clincial=clincials
        self.Data_WSI=WSI_feature
        self.length = len(self.df)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        # patient_feature_id=self.df[index]
        patient_name, patient_patch_id = self.df[index]
        index_clincial = self.clincial[self.clincial['PatientID'] == np.int64(patient_name)]
        OS = index_clincial['OS'].values[0]
        try:
            OSState = np.array(index_clincial['OSState'].values[0],dtype=np.int64)
        except ValueError:
            OSState=index_clincial['OSState'].values[0]

        DFS = index_clincial['DFS'].values[0]
        DFSState = index_clincial['DFSState'].values[0]

        WSI_feature = self.Data_WSI[patient_patch_id]
        r = {}
        r['index'] = torch.tensor(index)
        r['WSI_feature'] = WSI_feature
        r['patch_name']=self.name[patient_patch_id]
        r['patient_name'] = torch.tensor(np.int64(patient_name))
        r['OS']=torch.tensor(OS)
        try:
            r['OSState'] = torch.tensor(OSState)
        except TypeError:
            r['OSState'] = torch.tensor(np.nan)
        r['DFS'] = torch.tensor(DFS)
        r['DFSState'] = torch.tensor(DFSState)


        return r

tensor_list = [
     'index','patient_name','WSI_feature','OS','OSState','DFS','DFSState'
]





def image_to_tensor(image, mode='bgr'):  # image mode
    if mode == 'bgr':
        image = image[:, :, ::-1]
    x = image
    x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    return x


def tensor_to_image(x, mode='bgr'):
    image = x.data.cpu().numpy()
    image = image.transpose(1, 2, 0)
    if mode == 'bgr':
        image = image[:, :, ::-1]
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32)
    return image

tensor_list = [
     'index','patient_name','WSI_feature','OS','OSState','DFS','DFSState'
]


def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_list:
            try:
                v = torch.stack(v)
            except TypeError:
                v=None
        d[k] = v
    # d['organ'] = d['organ'].reshape(-1)
    return d

