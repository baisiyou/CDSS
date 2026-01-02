"""
数据预处理模块
处理eICU/MIMIC数据，提取特征用于模型训练
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, file_path):
        """加载CSV数据"""
        print(f"正在加载数据: {file_path}")
        df = pd.read_csv(file_path)
        print(f"数据形状: {df.shape}")
        return df
    
    def extract_features(self, df):
        """提取特征列"""
        # 排除ID列和目标列
        exclude_cols = ['patientunitstayid', 'hospitalid', 'death', 'length_of_stay']
        
        # 获取所有特征列
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        return df[feature_cols]
    
    def create_target_labels(self, df):
        """
        创建目标标签：肝肾功能异常
        基于BUN（血尿素氮）和实验室指标判断
        """
        # 定义异常阈值（标准化后的值）
        # BUN > 1.5 (约等于正常上限的1.5倍)
        # 或者存在其他异常指标
        
        targets = {}
        
        # 肾功能异常：BUN升高
        if 'bun' in df.columns:
            bun_abnormal = (df['bun'] > 1.5).astype(int)
            targets['kidney_abnormal'] = bun_abnormal
        
        # 肝功能异常：基于转氨酶等指标（如果数据中有）
        # 这里使用INR和albumin作为代理指标
        liver_abnormal = np.zeros(len(df))
        if 'inr' in df.columns:
            # INR升高可能表示肝功能异常
            liver_abnormal = (df['inr'] > 1.2).astype(int)
        if 'albu_lab' in df.columns:
            # 白蛋白降低可能表示肝功能异常
            liver_abnormal = liver_abnormal | (df['albu_lab'] < -1.0).astype(int)
        
        targets['liver_abnormal'] = liver_abnormal.astype(int)
        
        # 综合异常
        targets['organ_abnormal'] = (targets.get('kidney_abnormal', 0) | 
                                     targets.get('liver_abnormal', 0)).astype(int)
        
        return pd.DataFrame(targets)
    
    def prepare_training_data(self, df):
        """准备训练数据"""
        # 提取特征
        X = self.extract_features(df)
        
        # 创建目标标签
        y = self.create_target_labels(df)
        
        # 处理缺失值
        X = X.fillna(0)
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def scale_features(self, X_train, X_test=None):
        """标准化特征"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled

