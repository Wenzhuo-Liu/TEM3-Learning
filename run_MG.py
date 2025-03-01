import json
import cv2
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import math
from torch.nn import Parameter  

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import *
from torch.autograd import *
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

import shutil
import time
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from torch.optim import *
from torchvision.transforms import *
from mydata_xu import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
import random
import copy
from torch.utils.checkpoint import checkpoint
from keep_alive import EnhancedKeepAlive

EMOTION_LABEL = ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger']
DRIVER_BEHAVIOR_LABEL = ['Smoking', 'Making Phone', 'Looking Around', 'Dozing Off', 'Normal Driving', 'Talking',
                         'Body Movement']
SCENE_CENTRIC_CONTEXT_LABEL = ['Traffic Jam', 'Waiting', 'Smooth Traffic']
VEHICLE_BASED_CONTEXT_LABEL = ['Parking', 'Turning', 'Backward Moving', 'Changing Lane', 'Forward Moving']


class CarDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.path = pd.read_csv(csv_file)
        # self.path='/root/'+self.path
        self.transform = transform
        self.resize_height = 224
        self.resize_width = 224
        self.body_height = 112
        self.body_width = 112
        self.face_height = 64#56 #64
        self.face_width = 64#56 #64

    # /root/autodl-tmp/AIDE_Dataset/AIDE_Dataset/annotation/0006.json
    def __len__(self):
        return len(self.path)

    # 为数据加载器提供单个样本，包括图像数据和相关标签
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frames_path, label_path = self.path.iloc[idx]
        frames_path = './' + frames_path
        label_path = './' + label_path

        parts1 = frames_path.split('/')
        # parts1.insert(4, 'AIDE_Dataset')  # 在第四个元素后添加 'AIDE_Dataset'
        frames_path = '/'.join(parts1)

        parts2 = label_path.split('/')
        # parts2.insert(4, 'AIDE_Dataset')  # 在第四个元素后添加 'AIDE_Dataset'
        label_path = '/'.join(parts2)

        label_json = json.load(open(label_path))
        pose_list = label_json['pose_list']

        # buffer, buffer_front, buffer_left, buffer_right, buffer_face, buffer_body,keypoints = self.load_frames(frames_path, pose_list)  # 加载图像帧数据
        buffer, buffer_front, buffer_left, buffer_right, buffer_face, buffer_body, posture, gesture = self.load_frames(frames_path, pose_list)

        # buffer, buffer_front, buffer_left, buffer_right, buffer_body, buffer_face, keypoints = self.load_frames(
        #     frames_path, pose_list)  # 加载图像帧数据

        # 数据增强
        buffer = self.randomflip(buffer)
        buffer_front = self.randomflip(buffer_front)
        buffer_left = self.randomflip(buffer_left)
        buffer_right = self.randomflip(buffer_right)

        context = torch.cat([buffer, buffer_front, buffer_left, buffer_right], dim=0)  # 将四个张量沿批次维度拼接
        context = self.to_tensor(context)

        # 加载的图像数据-->PyTorch张量
        buffer = self.to_tensor(buffer)
        buffer_front = self.to_tensor(buffer_front)
        buffer_left = self.to_tensor(buffer_left)
        buffer_right = self.to_tensor(buffer_right)

        # 身体、面部、关节点
        buffer_body = self.to_tensor(buffer_body)
        buffer_face = self.to_tensor(buffer_face)
        # keypoints = keypoints.permute(2, 0, 1).contiguous()

        emotion_label = EMOTION_LABEL.index((label_json['emotion_label'].capitalize()))
        driver_behavior_label = DRIVER_BEHAVIOR_LABEL.index((label_json['driver_behavior_label']))
        scene_centric_context_label = SCENE_CENTRIC_CONTEXT_LABEL.index((label_json['scene_centric_context_label']))

        # 标签错误情况
        if label_json['vehicle_based_context_label'] == "Forward":
            label_json['vehicle_based_context_label'] = "Forward Moving"
        # print(label_json['vehicle_based_context_label'], label_path)
        vehicle_based_context_label = VEHICLE_BASED_CONTEXT_LABEL.index((label_json['vehicle_based_context_label']))

        sample = {
            'context': context,
            'body': buffer_body,
            'face': buffer_face,
            # 'keypoints': torch.stack([keypoints], dim=-1),
            'posture': posture,  # 使用 posture
            'gesture': gesture,  # 使用 gesture
            "emotion_label": emotion_label,
            "driver_behavior_label": driver_behavior_label,
            "scene_centric_context_label": scene_centric_context_label,
            "vehicle_based_context_label": vehicle_based_context_label
        }

        # keypoints = sample['keypoints']
        posture = sample['posture']  # 使用 posture
        gesture = sample['gesture']  # 使用 gesture
        context = sample['context']
        body = sample['body']
        face = sample['face']
        emotion_label = sample['emotion_label']
        behavior_label = sample['driver_behavior_label']
        context_label = sample['scene_centric_context_label']
        vehicle_label = sample['vehicle_based_context_label']
      

        # 返回图像数据和相关标签
        return buffer, buffer_front, buffer_left, buffer_right, buffer_face, buffer_body, posture,gesture, emotion_label, behavior_label, context_label, vehicle_label

    def load_frames(self, file_dir, pose_list):
        incar_path = os.path.join(file_dir, 'incarframes')
        front_frames = os.path.join(file_dir, 'frontframes')
        left_frames = os.path.join(file_dir, 'leftframes')
        right_frames = os.path.join(file_dir, 'rightframes')
        face_frames = os.path.join(file_dir, 'face')
        body_frames = os.path.join(file_dir, 'body')

        # 构建图像路径列表
        frames = [os.path.join(incar_path, img) for img in os.listdir(incar_path) if img.endswith('.jpg')]
        front_frames = [os.path.join(front_frames, img) for img in os.listdir(front_frames) if img.endswith('.jpg')]
        left_frames = [os.path.join(left_frames, img) for img in os.listdir(left_frames) if img.endswith('.jpg')]
        right_frames = [os.path.join(right_frames, img) for img in os.listdir(right_frames) if img.endswith('.jpg')]
        face_frames = [os.path.join(face_frames, img) for img in os.listdir(face_frames) if img.endswith('.jpg')]
        body_frames = [os.path.join(body_frames, img) for img in os.listdir(body_frames) if img.endswith('.jpg')]

        # 确保帧的数量一致，不足时填充
        if len(face_frames) != 45:
            face_frames.extend([face_frames[-1]] * (45 - len(face_frames)))
        if len(body_frames) != 45:
            body_frames.extend([body_frames[-1]] * (45 - len(body_frames)))

        # 对文件路径排序
        frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        front_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        left_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        right_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        face_frames.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
        body_frames.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

        # 初始化数据缓存
        buffer, buffer_front, buffer_left, buffer_right = [], [], [], []
        buffer_face, buffer_body = [], []
        posture_list, gesture_list = [], []

        # 遍历帧，加载图像和关键点
        for i, frame_name in enumerate(frames):
            if not i == 0 and not i % 3 == 2:
                continue
            if i >= 45:
                break

            # 加载帧并处理加载失败的情况
            def load_image(image_path, height, width):
                if not os.path.exists(image_path):
                    print(f"File not found: {image_path}")
                    return np.zeros((height, width, 3), dtype=np.uint8)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Failed to load image: {image_path}")
                    return np.zeros((height, width, 3), dtype=np.uint8)
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height))
                return img

            # 加载图像
            img = load_image(frame_name, self.resize_height, self.resize_width)
            front_img = load_image(front_frames[i], self.resize_height, self.resize_width)
            left_img = load_image(left_frames[i], self.resize_height, self.resize_width)
            right_img = load_image(right_frames[i], self.resize_height, self.resize_width)
            img_face = load_image(face_frames[i], self.face_height, self.face_width)
            img_body = load_image(body_frames[i], self.body_height, self.body_width)

            # 加载关键点数据
            keypoints = np.array(pose_list[i]['result'][0]['keypoints']).reshape(-1, 3)
            posture = keypoints[:26]
            gesture = keypoints[94:136]
            posture_list.append(posture)
            gesture_list.append(gesture)

            # 转换为 PyTorch 张量
            buffer.append(torch.from_numpy(img).float())
            buffer_front.append(torch.from_numpy(front_img).float())
            buffer_left.append(torch.from_numpy(left_img).float())
            buffer_right.append(torch.from_numpy(right_img).float())
            buffer_face.append(torch.from_numpy(img_face).float())
            buffer_body.append(torch.from_numpy(img_body).float())

        # 转换姿态和手势为 PyTorch 张量
        posture_tensor = torch.from_numpy(np.array(posture_list, dtype=np.float32))  # 转换为 PyTorch 张量
        gesture_tensor = torch.from_numpy(np.array(gesture_list, dtype=np.float32))

        # 返回加载的张量
        return (
            torch.stack(buffer),
            torch.stack(buffer_front),
            torch.stack(buffer_left),
            torch.stack(buffer_right),
            torch.stack(buffer_face),
            torch.stack(buffer_body),
            posture_tensor,
            gesture_tensor,
        )

    # def load_frames(self, file_dir, pose_list):

    #     incar_path = os.path.join(file_dir, 'incarframes')
    #     front_frames = os.path.join(file_dir, 'frontframes')
    #     left_frames = os.path.join(file_dir, 'leftframes')
    #     right_frames = os.path.join(file_dir, 'rightframes')
    #     face_frames = os.path.join(file_dir, 'face')
    #     body_frames = os.path.join(file_dir, 'body')


    #     frames = [os.path.join(incar_path, img) for img in os.listdir(incar_path) if img.endswith('.jpg')]
    #     front_frames = [os.path.join(front_frames, img) for img in os.listdir(front_frames) if img.endswith('.jpg')]
    #     left_frames = [os.path.join(left_frames, img) for img in os.listdir(left_frames) if img.endswith('.jpg')]
    #     right_frames = [os.path.join(right_frames, img) for img in os.listdir(right_frames) if img.endswith('.jpg')]

    #     face_frames = [os.path.join(face_frames, img) for img in os.listdir(face_frames) if img.endswith('.jpg')]
    #     if len(face_frames)!=45:
    #         face_frames.extend([face_frames[-1]] * (45 - len(face_frames)))
    #     body_frames = [os.path.join(body_frames, img) for img in os.listdir(body_frames) if img.endswith('.jpg')]
    #     if len(body_frames)!=45:
    #         body_frames.extend([body_frames[-1]] * (45 - len(body_frames)))


    #     frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    #     front_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    #     left_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    #     right_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    #     face_frames.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    #     body_frames.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))




    #     buffer, buffer_front, buffer_left, buffer_right, keypoints_list, buffer_face, buffer_body= [], [], [], [], [], [], []
    #     posture_list, gesture_list = [], [] 

    #     for i, frame_name in enumerate(frames):
    #         if not i == 0 and not i % 3 == 2:
    #             continue
    #         if i >= 45:
    #             break

    #         img = cv2.imread(frame_name)
    #         front_img = cv2.imread(front_frames[i])
    #         left_img = cv2.imread(left_frames[i])
    #         right_img = cv2.imread(right_frames[i])

    #         img_face = cv2.imread(face_frames[i])
    #         img_body = cv2.imread(body_frames[i])
    #         keypoints = np.array(pose_list[i]['result'][0]['keypoints']).reshape(-1, 3)
    #         # keypoints_list.append(torch.from_numpy(keypoints).float())
    #         # keypoint=keypoint[94:115]
    #         # posture =  keypoint[:,:,:,:26,:]
    #         # gesture = keypoint[:,:,:,94:,:]            
    #         posture =  keypoints[:26]
    #         gesture = keypoints[94:136]
            
    #         posture_list.append(posture)
    #         gesture_list.append(gesture)
            
            
            
    #         # img_body = img[int(body[1]):int(body[1] + max(body[3], 20)), int(body[0]):int(body[0] + max(body[2], 10))]
    #         # img_face = img[int(face[1]):int(face[1] + max(face[3], 10)), int(face[0]):int(face[0] + max(face[2], 10))]

    #         if img.shape[0] != self.resize_height or img.shape[1] != self.resize_width:
    #             img = cv2.resize(img, (self.resize_width, self.resize_height))
    #         if front_img.shape[0] != self.resize_height or front_img.shape[1] != self.resize_width:
    #             front_img = cv2.resize(front_img, (self.resize_width, self.resize_height))
    #         if left_img.shape[0] != self.resize_height or left_img.shape[1] != self.resize_width:
    #             left_img = cv2.resize(left_img, (self.resize_width, self.resize_height))
    #         if right_img.shape[0] != self.resize_height or right_img.shape[1] != self.resize_width:
    #             right_img = cv2.resize(right_img, (self.resize_width, self.resize_height))

    #         if img_body.shape[0] != self.resize_height or img_body.shape[1] != self.resize_width:
    #             img_body = cv2.resize(img_body, (self.resize_width, self.resize_height))

    #         try:
    #             if img_face.shape[0] != self.face_height or img_face.shape[1] != self.face_width:
    #                 img_face = cv2.resize(img_face, (self.face_width, self.face_height))
    #             # if img_face.shape[0] != self.resize_height or img_face.shape[1] != self.resize_width:
    #             #     img_face = cv2.resize(img_face, (self.resize_width, self.resize_height))
    #         except:
    #             img_face = img_body

    #         buffer.append(torch.from_numpy(img).float())
    #         buffer_front.append(torch.from_numpy(front_img).float())
    #         buffer_left.append(torch.from_numpy(left_img).float())
    #         buffer_right.append(torch.from_numpy(right_img).float())

    #         buffer_body.append(torch.from_numpy(img_body).float())
    #         # if len(face_frames)==45:
    #         buffer_face.append(torch.from_numpy(img_face).float())
    #         # keypoints.append(torch.from_numpy(keypoints).float())
    #         # keypoints_tensor = torch.stack(keypoints_list)
    #         # posture_tensor = torch.tensor(posture_list, dtype=torch.float)
    #         # gesture_tensor = torch.tensor(gesture_list, dtype=torch.float)

    #         posture_array = np.array(posture_list, dtype=np.float32)  # 将 posture_list 轈换为 numpy 数组
    #         gesture_array = np.array(gesture_list, dtype=np.float32)  # 将 gesture_list 转换为 numpy 数组
    #         posture_tensor = torch.from_numpy(posture_array)  # 将 numpy 数组转换为 PyTorch 张量
    #         gesture_tensor = torch.from_numpy(gesture_array)  # 将 numpy 数组转换为 PyTorch 张量

    #     return torch.stack(buffer), torch.stack(buffer_front), torch.stack(buffer_left), torch.stack(
    #         buffer_right), torch.stack(buffer_face), torch.stack(buffer_body), posture_tensor,gesture_tensor
    # #keypoints_tensor#torch.stack(keypoints) 

    # 随机翻转输入的PyTorch张量buffer(数据增强操作)
    def randomflip(self, buffer):

        # 以50%的概率在第二个维度上进行水平翻转
        if np.random.random() < 0.5:
            buffer = torch.flip(buffer, dims=[1])

        # 以50%的概率在第三个维度上进行垂直翻转
        if np.random.random() < 0.5:
            buffer = torch.flip(buffer, dims=[2])

        # 返回翻转后的张量buffer
        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):

        return buffer.permute(3, 0, 1, 2).contiguous()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)

train_dataset = CarDataset(csv_file='./training.csv')  # 'training.csv'
val_dataset = CarDataset(csv_file='./validation.csv')
test_dataset = CarDataset(csv_file='./testing.csv')

train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=24,shuffle=False, num_workers=4, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False)


class MSMambaBlock(nn.Module):
    def __init__(self, dim=256, d_state=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(dim, d_state))
        self.B = nn.Parameter(torch.randn(dim, d_state))
        self.C = nn.Parameter(torch.randn(dim, d_state))
        self.D = nn.Parameter(torch.zeros(dim))
        
        # Forward path
        self.conv_forward = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Backward path
        self.conv_backward = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Projections
        self.forward_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        self.backward_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def selective_scan(self, x, direction="forward"):
        """Simplified selective scan operation"""
        if direction == "forward":
            x = torch.einsum('bcd,ce->bed', x, self.B @ self.C.T)
        else:
            x = torch.einsum('bcd,ce->bed', x.flip(-1), self.B @ self.C.T).flip(-1)
        return x
    
    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        
        # Reshape for processing while preserving spatial information
        x_reshaped = x.view(B, C, H * W)
        
        # Forward path - 关注局部上下文和前向依赖
        x_f = self.conv_forward(x_reshaped)
        x_f = F.gelu(x_f)  # 非线性激活
        x_f = self.selective_scan(x_f, "forward")
        x_f = x_f.view(B, self.dim, H, W)
        # 添加局部注意力
        x_f = x_f + F.avg_pool2d(x_f, 3, stride=1, padding=1)
        x_f = x_f.permute(0, 2, 3, 1)
        x_f = self.forward_proj(x_f)
        
        # Backward path - 关注全局上下文和反向依赖
        x_b = self.conv_backward(x_reshaped)
        x_b = F.gelu(x_b)  # 非线性激活
        x_b = self.selective_scan(x_b, "backward")
        x_b = x_b.view(B, self.dim, H, W)
        # 添加全局注意力
        x_b = x_b + F.adaptive_avg_pool2d(x_b, (H, W))
        x_b = x_b.permute(0, 2, 3, 1)
        x_b = self.backward_proj(x_b)
        
        # Combine paths
        x = x_f + x_b
        
        # 确保维度匹配
        d_state_ones = torch.ones(self.d_state, device=x.device)  # [d_state]
        dim_ones = torch.ones(self.dim, device=x.device)  # [dim]
        
        ssm_weight = torch.sigmoid(
            self.D[None, None, None, :] + 
            (self.A @ d_state_ones).unsqueeze(0).unsqueeze(0).unsqueeze(0) +
            (self.B @ self.C.T @ dim_ones).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
        
        # ssm_weight = torch.sigmoid(
        #     # 1. 基础偏置项
        #     self.D[None, None, None, :] + 
            
        #     # 2. A 矩阵的影响
        #     (self.A @ d_state_ones).unsqueeze(0).unsqueeze(0).unsqueeze(0) +
            
        #     # 3. B和C矩阵的组合影响
        #     (self.B @ self.C.T @ dim_ones).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # )
        
        x = x * ssm_weight
        
        # Final processing
        x = self.norm(x)
        x = self.out_proj(x)
        x = x.permute(0, 3, 1, 2)
        
        return identity + 0.1 * x


def create_conv_sequence(in_channels=48, out_channels=512):
    return nn.Sequential(
        # 第一层：降维
        nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # 第二层
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        # 第三层
        nn.Conv2d(256, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class MultiTaskFusionGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super(MultiTaskFusionGate, self).__init__()
        self.channels = channels
        self.reduced_dim = channels // reduction
        
        # QKV transformations
        self.query = nn.Sequential(
            nn.Conv2d(channels*3, self.reduced_dim, 1, bias=False),
            nn.BatchNorm2d(self.reduced_dim),
            nn.ReLU(inplace=True)
        )
        
        self.key = nn.Sequential(
            nn.Conv2d(channels*3, self.reduced_dim, 1, bias=False),
            nn.BatchNorm2d(self.reduced_dim)
        )
        
        self.value = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # Task-specific gates
        self.gate1_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate1_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate1_3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        self.gate2_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate2_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate2_3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        self.gate3_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate3_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate3_3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        self.gate4_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate4_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate4_3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, h1, h2, h3):
        # Concatenate inputs
        concat = torch.cat([h1, h2, h3], dim=1)
        
        # Generate Q, K, V
        q = self.query(concat)
        k = self.key(concat)
        v = self.value(concat)
        
        # Attention computation
        B, C, H, W = q.shape
        q = q.flatten(2)
        k = k.flatten(2)
        v = v.flatten(2)
        
        attn = torch.bmm(q.transpose(1, 2), k)
        attn = F.softmax(attn / math.sqrt(self.reduced_dim), dim=-1)
        
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(B, -1, H, W)
        
        # Apply task-specific gates
        # emotion_feat = out * self.gate1_1(out) + out * self.gate1_2(out) + out * self.gate1_3(out)
        # behavior_feat = out * self.gate2_1(out) + out * self.gate2_2(out) + out * self.gate2_3(out)
        # scene_feat = out * self.gate3_1(out) + out * self.gate3_2(out) + out * self.gate3_3(out)
        # vehicle_feat = out * self.gate4_1(out) + out * self.gate4_2(out) + out * self.gate4_3(out)
        
        emotion_feat = h1 * self.gate1_1(out) + h2 * self.gate1_2(out) + h3 * self.gate1_3(out)
        behavior_feat = h1 * self.gate2_1(out) + h2 * self.gate2_2(out) + h3 * self.gate2_3(out)
        scene_feat = h1 * self.gate3_1(out) + h2 * self.gate3_2(out) + h3 * self.gate3_3(out)
        vehicle_feat = h1 * self.gate4_1(out) + h2 * self.gate4_2(out) + h3 * self.gate4_3(out)

        return emotion_feat, behavior_feat, scene_feat, vehicle_feat

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        
        # 降低初始通道数和特征维度
        self.initial_conv1 = create_conv_sequence(in_channels=48, out_channels=128)
        self.initial_conv2 = create_conv_sequence(in_channels=48, out_channels=128)
        self.initial_conv3 = create_conv_sequence(in_channels=48, out_channels=128)
        self.initial_conv4 = create_conv_sequence(in_channels=48, out_channels=128)
        self.face_conv = create_conv_sequence(in_channels=48, out_channels=128)
        self.body_conv = create_conv_sequence(in_channels=48, out_channels=128)
        
        # 归一化层
        self.bn = nn.BatchNorm2d(128)
        self.norm = nn.LayerNorm([128, 64, 64])
        
        # Mamba块
        self.mamba_block1 = MSMambaBlock(dim=128, d_state=4)
        self.mamba_block2 = MSMambaBlock(dim=128, d_state=4)
        
        # 3D卷积
        self.conv3d_gesture = ConvNet3D(num_keypoints=26, out_dim=128)
        self.conv3d_posture = ConvNet3D(num_keypoints=42, out_dim=128)
        
        # Replace RDF gates with attention fusion gates
        self.fusion_gate = MultiTaskFusionGate(channels=128)
        
        # 为每个任务创建独立的池化层
        self.avg_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in range(4)
        ])
        
        # 为每个任务创建独立的分类器
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(4)
        ])
        
        self.fc1 = nn.Linear(128, 5)
        self.fc2 = nn.Linear(128, 7)
        self.fc3 = nn.Linear(128, 3)
        self.fc4 = nn.Linear(128, 5)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用更保守的初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 更保守的线性层初始化
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, img1, img2, img3, img4, face, body, gesture, posture):
        # 特征提取
        def process_feature(x, conv):
            x = conv(x)
            x = F.adaptive_avg_pool2d(x, (64, 64))
            return x * 0.1  # 缩放因子
        
        # 处理所有输入
        img1 = process_feature(img1, self.initial_conv1)
        img2 = process_feature(img2, self.initial_conv2)
        img3 = process_feature(img3, self.initial_conv3)
        img4 = process_feature(img4, self.initial_conv4)
        face = process_feature(face, self.face_conv)
        body = process_feature(body, self.body_conv)
        
        # 特征融合
        group1 = self.bn(img1 + face + body)
        group2 = self.bn(img2 + img3 + img4)
        
        # Mamba处理
        h1 = self.mamba_block1(group1)
        h2 = self.mamba_block2(group2)
        
        # 处理姿态和手势
        h_gesture = self.conv3d_gesture(gesture)
        h_posture = self.conv3d_posture(posture)
        
        # 扩展维度
        h_gesture = h_gesture.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 64, 64) * 0.1
        h_posture = h_posture.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 64, 64) * 0.1
        h3 = self.bn(h_gesture + h_posture)
        
        # 使用新的多任务融合门控
        emotion_feat, behavior_feat, scene_feat, vehicle_feat = self.fusion_gate(h1, h2, h3)
        
        # 为每个任务独立处理特征
        emotion_out = self.classifiers[0](self.avg_pools[0](emotion_feat).flatten(1))
        behavior_out = self.classifiers[1](self.avg_pools[1](behavior_feat).flatten(1))
        scene_out = self.classifiers[2](self.avg_pools[2](scene_feat).flatten(1))
        vehicle_out = self.classifiers[3](self.avg_pools[3](vehicle_feat).flatten(1))
        
        # 返回每个任务的预测结果
        return (
            self.fc1(emotion_out),
            self.fc2(behavior_out),
            self.fc3(scene_out),
            self.fc4(vehicle_out)
        )



class ConvNet3D(nn.Module):
    def __init__(self, num_keypoints=26, out_dim=256):
        super(ConvNet3D, self).__init__()
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        
        # 添加投影层，将特征映射到指定维度
        self.projection = nn.Sequential(
            nn.Linear(128 * num_keypoints, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: [batch, channels, frames, keypoints, 1]
        B, C, F, K, _ = x.shape
        
        # 3D卷积处理
        x = self.conv3d(x)  # [B, 128, F, K, 1]
        
        # 重塑并投影到指定维度
        x = x.mean(dim=2)  # 在时间维度上平均池化 [B, 128, K, 1]
        x = x.view(B, -1)  # 展平 [B, 128*K]
        x = self.projection(x)  # 投影到指定维度 [B, out_dim]
        
        return x


choices = ["demo", "main", "test", "checkValidation", "getVideoEmbeddings", "generateEmbeddingsForVideoAudio",
           "imageToImageQueries", "crossModalQueries"]

# parser = argparse.ArgumentParser(description="Select code to run.")
# parser.add_argument('--mode', default="test", choices=choices, type=str)

checkpoint_dir = './'


class valConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        f1_list = []
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            # Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            f1_list.append(F1)
        return f1_list


class testConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity, F1])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


class LossAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n

    def getacc(self):
        return (self.sum * 100) / self.count


# Main function here
def main(use_cuda=True, EPOCHS=125, batch_size=48):
    # 创建 keep_alive 实例
    keep_alive = EnhancedKeepAlive(interval=60)  # 每60秒检查一次
    
    # model = ImageConvNet().cuda()
    # model = TotalNet().cuda()
    model = TotalNet()  # 创建模型实例
    model = nn.DataParallel(model)  # 使用 DataParallel 包装模型
    
    # # model.load_state_dict({k.replace('module.',''):v for k,v in model_dict.items()})
    # model.load_state_dict(model_dict, strict=False)
    
    model = model.cuda()  # 将模型移动到 CUDA 上



    crossEntropy1 = nn.CrossEntropyLoss()
    crossEntropy2 = nn.CrossEntropyLoss()
    crossEntropy3 = nn.CrossEntropyLoss()
    crossEntropy4 = nn.CrossEntropyLoss()
    print("Loaded dataloader and loss function.")

    # optim = Adam(model.parameters(), lr=lr, weight_decay=1e-7)
    # optim = SGD(model.parameters(), lr=0.25e-3, momentum=0.9, weight_decay=1e-4)
    optim = SGD(model.parameters(), lr=0.25e-4, momentum=0.9, weight_decay=1e-4)
    print("Optimizer loaded.")
    model.train()

    # state_dict = torch.load(os.path.join(checkpoint_dir, model_name))
    # model.load_state_dict(state_dict)
    # try:
    #     best_precision = 0
    #     lowest_loss = 100000
    #     best_avgf1 = 0
    #     # best_weightf1 = 0
    #     for epoch in range(EPOCHS):
    #         if (50 <= epoch < 100):
    #             optim = SGD(model.parameters(), lr=0.25e-4, momentum=0.9, weight_decay=1e-4)
    #         if (epoch >= 100):
    #             optim = SGD(model.parameters(), lr=0.25e-5, momentum=0.9, weight_decay=1e-4)
    # try:
    best_precision = 0
    lowest_loss = 100000
    best_avgf1 = 0
    best_weightf1 = 0


    try:
        # 启动监控
        keep_alive.start(model, optim, 0)
        
        for epoch in range(EPOCHS):
            if ( epoch <= 25):
                optim = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
            if (25 < epoch <= 50):
                optim = SGD(model.parameters(), lr=0.5e-3, momentum=0.9, weight_decay=1e-4)
            if (epoch > 50):
                optim = SGD(model.parameters(), lr=0.5e-4, momentum=0.9, weight_decay=1e-4)
            # Run algo

            # 训练损失
            train_losses = LossAverageMeter()

            # train_acc1 是一个AccAverageMeter类的实例，用于追踪训练准确率的平均值
            train_acc1 = AccAverageMeter()
            train_acc2 = AccAverageMeter()
            train_acc3 = AccAverageMeter()
            train_acc4 = AccAverageMeter()
            if (epoch == 0):
                end = time.time()
            # for subepoch, (img, aud, out) in enumerate(train_dataloader):#dataloader context,behavior_label

            # 遍历训练数据加载器，获取图像img1, img2, img3, img4和相应标签 , points
            for subepoch, (img1,img2,img3,img4,face,body,gesture,posture,emotion_label, behavior_label, context_label, vehicle_label) in enumerate(
                    train_dataloader):

                # 打印第一个epoch中第一个批次所用时间
                if (epoch == 0 and subepoch == 0):
                    print(time.time() - end)
                # 梯度清零
                optim.zero_grad()
                # 改变图像张量形状（五维->四维），使其与模型兼容
                B, _, _, H, W = img1.shape
                img1 = img1.view(B, -1,  H, W)  # [16, 3, 16, 224, 224]
                img2 = img2.view(B, -1,  H, W)
                img3 = img3.view(B, -1,  H, W)
                img4 = img4.view(B, -1,  H, W)
                
                face = face.view(B, -1,  64, 64)
                body = body.view(B, -1,  112,112)
# Gesture Skeleton Keypoint 3 (C)×16 (F)×42 (K)×1 (P)
# Posture Skeleton Keypoint 3 (C)×16 (F)×26 (K)×1 (P)
                gesture = gesture.view(B, 3,16,  26, 1)
                posture = posture.view(B,3,16, 42, 1)    
                # 批次大小
                M = img1.shape[0]
                # 将数据移到GPU上
                if use_cuda:
                    img1 = img1.cuda()
                    img2 = img2.cuda()
                    img3 = img3.cuda()
                    img4 = img4.cuda()
                    
                    face = face.cuda()
                    body = body.cuda()
                    
                    gesture = gesture.cuda()
                    posture = posture.cuda()
                    
                emotion_label = emotion_label.cuda()
                behavior_label = behavior_label.cuda()
                context_label = context_label.cuda()
                vehicle_label = vehicle_label.cuda()

                # 将输入图像传递到模型(前向传播)
                out1, out2, out3, out4 = model(img1,img2,img3,img4,face,body,gesture,posture)
                # if subepoch%400 == 0:
                # 	print(o)
                # 	print(out)
                # print(o.shape, out.shape)
                # print(out1.shape, emotion_label.shape)

                # 计算单独损失和总损失
                loss1 = crossEntropy1(out1, emotion_label)
                # print(out2.shape, behavior_label.shape)
                loss2 = crossEntropy2(out2, behavior_label)
                # print(out3.shape, context_label.shape)
                loss3 = crossEntropy3(out3, context_label)
                # print(out4.shape, vehicle_label.shape)
                loss4 = crossEntropy4(out4, vehicle_label)
                loss = loss1 + loss2 + loss3 + loss4
                # print(loss)

                # 更新训练损失
                train_losses.update(loss.item(), M)
                # 反向传播，进一步优化
                loss.backward()
                optim.step()

                # Calculate accuracy
                out1 = F.softmax(out1, 1)  # Softmax将一组数值转换为概率分布
                ind = out1.argmax(dim=1)  # ind保存每行中最大值所在的索引，即概率最大的类别
                # print(ind.data)
                # print(out1.data)

                # 计算当前批次的准确率
                accuracy1 = (ind.data == emotion_label.data).sum() * 1.0 / M
                # 更新整个训练过程的准确率平均值
                train_acc1.update((ind.data == emotion_label.data).sum() * 1.0, M)

                out2 = F.softmax(out2, 1)
                ind = out2.argmax(dim=1)
                accuracy2 = (ind.data == behavior_label.data).sum() * 1.0 / M
                train_acc2.update((ind.data == behavior_label.data).sum() * 1.0, M)

                out3 = F.softmax(out3, 1)
                ind = out3.argmax(dim=1)
                accuracy3 = (ind.data == context_label.data).sum() * 1.0 / M
                train_acc3.update((ind.data == context_label.data).sum() * 1.0, M)

                out4 = F.softmax(out4, 1)
                ind = out4.argmax(dim=1)
                accuracy4 = (ind.data == vehicle_label.data).sum() * 1.0 / M
                train_acc4.update((ind.data == vehicle_label.data).sum() * 1.0, M)


                # if subepoch % 1 == 0:
                print("Epoch: %d, Subepoch: %d, Loss: %f, "
                        "batch_size: %d, total_acc1: %f, total_acc2: %f, total_acc3: %f, total_acc4: %f" % (
                    epoch, subepoch, train_losses.avg, M,
                    train_acc1.getacc(),
                    train_acc2.getacc(),
                    train_acc3.getacc(),
                    train_acc4.getacc()))

                with open(file="./Logs/MGv6+.txt", mode="a+") as f:
                    f.write("Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d, total_acc1: %f,total_acc2: %f, total_acc3: %f, total_acc4: %f"\
                         %(epoch, subepoch, train_losses.avg, M, train_acc1.getacc(), train_acc2.getacc(), train_acc3.getacc(), train_acc4.getacc()))


            # 验证阶段
            print("Valing...")
            # val_losses = LossAverageMeter()

            val_losses1 = LossAverageMeter()
            val_losses2 = LossAverageMeter()
            val_losses3 = LossAverageMeter()
            val_losses4 = LossAverageMeter()
            # val_losses = (val_losses1.avg + val_losses2.avg + val_losses3.avg + val_losses4.avg) / 4.0

            val_acc1 = AccAverageMeter()
            val_acc2 = AccAverageMeter()
            val_acc3 = AccAverageMeter()
            val_acc4 = AccAverageMeter()
      
            # 混淆矩阵
            valconfusion1 = valConfusionMatrix(num_classes = 5, labels = EMOTION_LABEL)
            valconfusion2 = valConfusionMatrix(num_classes = 7, labels = DRIVER_BEHAVIOR_LABEL)
            valconfusion3 = valConfusionMatrix(num_classes = 3, labels = SCENE_CENTRIC_CONTEXT_LABEL)
            valconfusion4 = valConfusionMatrix(num_classes = 5, labels = VEHICLE_BASED_CONTEXT_LABEL)

            # 将模型设置为评估模式
            model.eval()


            for subepoch1, (img1,img2,img3,img4,face,body,gesture,posture,emotion_label, behavior_label, context_label, vehicle_label) in enumerate(
                    val_dataloader):

                if (epoch == 0 and subepoch1 == 0):
                    print(time.time() - end)
                with torch.no_grad():
                             
                    B, _, _, H, W = img1.shape
                    img1 = img1.view(B, -1,  H, W)  # [16, 3, 16, 224, 224]
                    img2 = img2.view(B, -1,  H, W)
                    img3 = img3.view(B, -1,  H, W)
                    img4 = img4.view(B, -1,  H, W)

                    face = face.view(B, -1,  64, 64)
                    body = body.view(B, -1,  112,112)
# Gesture Skeleton Keypoint 3 (C)×16 (F)×42 (K)×1 (P)
# Posture Skeleton Keypoint 3 (C)×16 (F)×26 (K)×1 (P)
                    gesture = gesture.view(B, 3,16,  26, 1)
                    posture = posture.view(B,3,16, 42, 1)    
                    # gesture = gesture.view(B, -1,  26, 1)
                    # posture = posture.view(B,-1, 21, 1)                 

                    # 批次大小
                    M = img1.shape[0]
                    # 将数据移到GPU上
                    if use_cuda:
                        img1 = img1.cuda()
                        img2 = img2.cuda()
                        img3 = img3.cuda()
                        img4 = img4.cuda()

                        face = face.cuda()
                        body = body.cuda()

                        gesture = gesture.cuda()
                        posture = posture.cuda()

                    emotion_label = emotion_label.cuda()
                    behavior_label = behavior_label.cuda()
                    context_label = context_label.cuda()
                    vehicle_label = vehicle_label.cuda()

                    # 将输入图像传递到模型(前向传播)
                    out1, out2, out3, out4 = model(img1,img2,img3,img4,face,body,gesture,posture)
                   

                    loss1 = crossEntropy1(out1, emotion_label)
                    # print(out2.shape, behavior_label.shape)
                    loss2 = crossEntropy2(out2, behavior_label)
                    # print(out3.shape, context_label.shape)
                    loss3 = crossEntropy3(out3, context_label)
                    # print(out4.shape, vehicle_label.shape)
                    loss4 = crossEntropy4(out4, vehicle_label)
                    loss = loss1 + loss2 + loss3 + loss4
                    # print(loss)
                    val_losses1.update(loss1.item(), M)
                    val_losses2.update(loss1.item(), M)
                    val_losses3.update(loss1.item(), M)
                    val_losses4.update(loss1.item(), M)

                    val_losses = (val_losses1.avg + val_losses2.avg + val_losses3.avg + val_losses4.avg) / 4.0
                    #早停，防止过拟合
                    # Calculate accuracy
                    out1 = F.softmax(out1, 1)
                    ind1 = out1.argmax(dim=1)
                    # print(ind.data)
                    # print(out1.data)
                    accuracy1 = (ind1.data == emotion_label.data).sum() * 1.0 / M
                    val_acc1.update((ind1.data == emotion_label.data).sum() * 1.0, M)
                    valconfusion1.update(ind1.to("cpu").numpy(), emotion_label.to("cpu").numpy())  # 更新混淆矩阵
                    avgf11 = (valconfusion1.summary()[0] + valconfusion1.summary()[1] + valconfusion1.summary()[2]+
                             valconfusion1.summary()[3]+valconfusion1.summary()[4]) / 5.0

                    out2 = F.softmax(out2, 1)
                    ind2 = out2.argmax(dim=1)
                    accuracy2 = (ind2.data == behavior_label.data).sum() * 1.0 / M
                    val_acc2.update((ind2.data == behavior_label.data).sum() * 1.0, M)
                    valconfusion2.update(ind2.to("cpu").numpy(), behavior_label.to("cpu").numpy())
                    avgf12 = (valconfusion2.summary()[0] + valconfusion2.summary()[1] + valconfusion2.summary()[2] +
                             valconfusion2.summary()[3] + valconfusion2.summary()[4] +
                             valconfusion2.summary()[5] + valconfusion2.summary()[6]) / 7.0

                    out3 = F.softmax(out3, 1)
                    ind3 = out3.argmax(dim=1)
                    accuracy3 = (ind3.data == context_label.data).sum() * 1.0 / M
                    val_acc3.update((ind3.data == context_label.data).sum() * 1.0, M)
                    valconfusion3.update(ind3.to("cpu").numpy(), context_label.to("cpu").numpy())
                    avgf13 = (valconfusion3.summary()[0] + valconfusion3.summary()[1] + valconfusion3.summary()[2]) / 3.0

                    out4 = F.softmax(out4, 1)
                    ind4 = out4.argmax(dim=1)
                    accuracy4 = (ind4.data == vehicle_label.data).sum() * 1.0 / M
                    val_acc4.update((ind4.data == vehicle_label.data).sum() * 1.0, M)
                    valconfusion4.update(ind4.to("cpu").numpy(), vehicle_label.to("cpu").numpy())
                    avgf14 = (valconfusion4.summary()[0] + valconfusion4.summary()[1] + valconfusion4.summary()[2] +
                             valconfusion4.summary()[3] + valconfusion4.summary()[4]) / 5.0

                    total_avgf1 = (avgf11 + avgf12 + avgf13 + avgf14) / 4.0

                    # if subepoch1 % 1 == 0:
                    print(
                        "Val Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d,total_acc1: %f, total_acc2: %f, "
                        "total_acc3: %f, total_acc4: %f, avgf11: %f, avgf12: %f, avgf13: %f, avgf14: %f" % (
                            epoch, subepoch1, val_losses, M,
                            val_acc1.getacc(),
                            val_acc2.getacc(),
                            val_acc3.getacc(),
                            val_acc4.getacc(), avgf11, avgf12, avgf13, avgf14))
                    
                    with open(file="./Logs/MGv6+.txt", mode="a+") as f:
                            f.write("Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d, total_acc1: %f,total_acc2: %f, total_acc3: %f, total_acc4: %f, \
                                    avgf11: %f, avgf12: %f, avgf13: %f, avgf14: %f\n"\
                             %(epoch, subepoch1, val_losses, M, val_acc1.getacc(), val_acc2.getacc(), val_acc3.getacc(), val_acc4.getacc(),avgf11, avgf12, avgf13, avgf14))

            # 更新最佳模型
            is_best_avgf1 = total_avgf1 > best_avgf1
            # is_best_weightf1 = weightf1 > best_weightf1
            val_acc = (val_acc1.getacc() + val_acc2.getacc() + val_acc3.getacc() + val_acc4.getacc()) / 4.0
            is_best = val_acc > best_precision
            is_lowest_loss = val_losses < lowest_loss
            best_precision = max(val_acc, best_precision)
            lowest_loss = min(val_losses, lowest_loss)
            best_avgf1 = max(total_avgf1, best_avgf1)


            print("Epoch: %d,best_precision: %f,lowest_loss: %f,best_avgf1: %f" % (
            epoch, best_precision, lowest_loss, best_avgf1))


            # 保存最佳模型(将当前模型的权重保存到best_model.pt文件中)
            best_path = os.path.join(checkpoint_dir, './Logs/MGv6+.pt')
            if is_best:
                with open(file="./Logs/MGv6+.txt", mode="w") as f:
                        f.write("Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d, total_acc1: %f,total_acc2: %f, total_acc3: %f, total_acc4: %f, \
                                avgf11: %f, avgf12: %f, avgf13: %f, avgf14: %f\n"\
                         %(epoch, subepoch1, val_losses, M, val_acc1.getacc(), val_acc2.getacc(), val_acc3.getacc(), val_acc4.getacc(),avgf11, avgf12, avgf13, avgf14))
                # shutil.copyfile(save_path, best_path)
                torch.save(model.state_dict(), best_path)
                print("Successfully saved the model with the best precision!")

            # 保存最低损失模型
            # lowest_path = os.path.join(checkpoint_dir, 'lowest_loss_swin_block_context.pt')
            # if is_lowest_loss:
            #     shutil.copyfile(save_path, lowest_path)
                # torch.save(model.state_dict(), lowest_path)
                print("Successfully saved the model with the lowest loss!")

            # 保存最佳平均F1模型
            # best_avgf1_path = os.path.join(checkpoint_dir, 'best_avgf1_swin_block_context.pt')
            # if is_best_avgf1:
            #     shutil.copyfile(save_path, best_avgf1_path)
                # torch.save(model.state_dict(), best_avgf1_path)
                print("Successfully saved the model with the best avgf1!")


            # 更新活动状态
            keep_alive.update_activity()

    except Exception as e:
        print(f"Training interrupted: {e}")
        # 保存紧急检查点
        keep_alive.save_emergency_checkpoint(model, optim, epoch)
    finally:
        # 停止监控
        keep_alive.stop()



class TestMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n

    def getacc(self):
        return (self.sum * 100) / self.count


def test(use_cuda=True, batch_size=16, model_name="./Logs/MGv6+.pt"):

    model = TotalNet()  # 创建模型实例
    model = nn.DataParallel(model)  # 使用 DataParallel 包装模型
    model = model.cuda()  # 将模型移动到 CUDA 上
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name))
        print("Loading from previous checkpoint.")
        

    test_dataset = CarDataset(csv_file='./testing.csv')

    testdataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    crossEntropy = nn.CrossEntropyLoss()
    print("Loaded dataloader and loss function.")

    test_losses = LossAverageMeter()
    test_acc1 = TestMeter()
    test_acc2 = TestMeter()
    test_acc3 = TestMeter()
    test_acc4 = TestMeter()




    testconfusion1 = valConfusionMatrix(num_classes=5, labels=EMOTION_LABEL)
    testconfusion2 = valConfusionMatrix(num_classes=7, labels=DRIVER_BEHAVIOR_LABEL)
    testconfusion3 = valConfusionMatrix(num_classes=3, labels=SCENE_CENTRIC_CONTEXT_LABEL)
    testconfusion4 = valConfusionMatrix(num_classes=5, labels=VEHICLE_BASED_CONTEXT_LABEL)

    model.eval()

    best_precision = 0
    lowest_loss = 100000
    best_avgf1 = 0
    for subepoch2, (img1,img2,img3,img4,face,body,gesture,posture,emotion_label, behavior_label, context_label, vehicle_label) in enumerate(
                test_dataloader):

        # if (epoch == 0 and subepoch2 == 0):
        #     print(time.time() - end)
        with torch.no_grad():

            B, _, _, H, W = img1.shape
            img1 = img1.view(B, -1,  H, W)  # [16, 3, 16, 224, 224]
            img2 = img2.view(B, -1,  H, W)
            img3 = img3.view(B, -1,  H, W)
            img4 = img4.view(B, -1,  H, W)

            face = face.view(B, -1,  64, 64)
            body = body.view(B, -1,  112,112)
# Gesture Skeleton Keypoint 3 (C)×16 (F)×42 (K)×1 (P)
# Posture Skeleton Keypoint 3 (C)×16 (F)×26 (K)×1 (P)
            gesture = gesture.view(B, 3,16,  26, 1)
            posture = posture.view(B,3,16, 42, 1)    
            # gesture = gesture.view(B, -1,  26, 1)
            # posture = posture.view(B,-1, 21, 1)             

            # 批次大小
            M = img1.shape[0]
            # 将数据移到GPU上
            if use_cuda:
                img1 = img1.cuda()
                img2 = img2.cuda()
                img3 = img3.cuda()
                img4 = img4.cuda()

                face = face.cuda()
                body = body.cuda()

                gesture = gesture.cuda()
                posture = posture.cuda()

            emotion_label = emotion_label.cuda()
            behavior_label = behavior_label.cuda()
            context_label = context_label.cuda()
            vehicle_label = vehicle_label.cuda()

            # 将输入图像传递到模型(前向传播)
            out1, out2, out3, out4 = model(img1,img2,img3,img4,face,body,gesture,posture)


            # Calculate accuracy
            out1 = F.softmax(out1, 1)
            ind1 = out1.argmax(dim=1)
            # print(ind.data)
            # print(out1.data)
            accuracy1 = (ind1.data == emotion_label.data).sum() * 1.0 / M
            test_acc1.update((ind1.data == emotion_label.data).sum() * 1.0, M)
            testconfusion1.update(ind1.to("cpu").numpy(), emotion_label.to("cpu").numpy())
            avgf11 = (testconfusion1.summary()[0] + testconfusion1.summary()[1] + testconfusion1.summary()[2] +
                      testconfusion1.summary()[3] + testconfusion1.summary()[4]) / 5.0

            out2 = F.softmax(out2, 1)
            ind2 = out2.argmax(dim=1)
            accuracy2 = (ind2.data == behavior_label.data).sum() * 1.0 / M
            test_acc2.update((ind2.data == behavior_label.data).sum() * 1.0, M)
            testconfusion2.update(ind2.to("cpu").numpy(), behavior_label.to("cpu").numpy())
            avgf12 = (testconfusion2.summary()[0] + testconfusion2.summary()[1] + testconfusion2.summary()[2] +
                      testconfusion2.summary()[3] + testconfusion2.summary()[4] +
                      testconfusion2.summary()[5] + testconfusion2.summary()[6]) / 7.0

            out3 = F.softmax(out3, 1)
            ind3 = out3.argmax(dim=1)
            accuracy3 = (ind3.data == context_label.data).sum() * 1.0 / M
            test_acc3.update((ind3.data == context_label.data).sum() * 1.0, M)
            testconfusion3.update(ind3.to("cpu").numpy(), context_label.to("cpu").numpy())
            avgf13 = (testconfusion3.summary()[0] + testconfusion3.summary()[1] + testconfusion3.summary()[2]) / 3.0

            out4 = F.softmax(out4, 1)
            ind4 = out4.argmax(dim=1)
            accuracy4 = (ind4.data == vehicle_label.data).sum() * 1.0 / M
            test_acc4.update((ind4.data == vehicle_label.data).sum() * 1.0, M)
            testconfusion4.update(ind4.to("cpu").numpy(), vehicle_label.to("cpu").numpy())
            avgf14 = (testconfusion4.summary()[0] + testconfusion4.summary()[1] + testconfusion4.summary()[2] +
                      testconfusion4.summary()[3] + testconfusion4.summary()[4]) / 5.0

            total_avgf1 = (avgf11 + avgf12 + avgf13 + avgf14) / 4.0
            mAcc = (test_acc1.getacc() + test_acc2.getacc() + test_acc3.getacc() + test_acc4.getacc()) / 4.0

            # if subepoch1 % 1 == 0:
            print(
                "Test  Subepoch: %d, batch_size: %d,total_acc1: %f, total_acc2: %f, "
                "total_acc3: %f, total_acc4: %f, avgf11: %f, avgf12: %f, avgf13: %f, avgf14: %f,  mAcc: %f" % (
                    subepoch2, M,
                    test_acc1.getacc(),
                    test_acc2.getacc(),
                    test_acc3.getacc(),
                    test_acc4.getacc(), avgf11, avgf12, avgf13, avgf14, mAcc))
            
            with open(file="./Logs/MGv6test.txt", mode="a+") as f:
                    f.write("Test  Subepoch: %d, batch_size: %d,total_acc1: %f, total_acc2: %f, "
                "total_acc3: %f, total_acc4: %f, avgf11: %f, avgf12: %f, avgf13: %f, avgf14: %f, mAcc: %f\n"
                     %(subepoch2, M,
                    test_acc1.getacc(),
                    test_acc2.getacc(),
                    test_acc3.getacc(),
                    test_acc4.getacc(), avgf11, avgf12, avgf13, avgf14,mAcc))


    testconfusion1.summary()
    testconfusion2.summary()
    testconfusion3.summary()
    testconfusion4.summary()
    # testconfusion1.plot()
    # testconfusion2.plot()
    # testconfusion3.plot()
    # testconfusion4.plot()







if __name__ == "__main__":
    cuda = True

mode = "train"
print("running...")
cuda = torch.cuda.is_available()

if mode == "train":
    main(use_cuda=cuda, batch_size=64)
    print("main mode...")
elif mode == "test":
    test(use_cuda=cuda, batch_size=64)
    print("test mode...")
