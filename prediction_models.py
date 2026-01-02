"""
预测模型模块
构建肝肾功能异常预测模型
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

class OrganFunctionPredictor:
    def __init__(self, model_type='random_forest'):
        """
        初始化预测器
        model_type: 'random_forest', 'gradient_boosting', 'logistic_regression'
        """
        self.model_type = model_type
        self.models = {}
        self.feature_importance = {}
        
    def _get_model(self, target_name):
        """获取指定类型的模型"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:  # logistic_regression
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练模型
        y_train可以是多列DataFrame（多个目标）
        """
        if isinstance(y_train, pd.DataFrame):
            targets = y_train.columns
        else:
            targets = ['target']
            y_train = pd.DataFrame({targets[0]: y_train})
        
        results = {}
        
        for target in targets:
            print(f"\n训练 {target} 预测模型...")
            model = self._get_model(target)
            
            # 训练
            model.fit(X_train, y_train[target])
            self.models[target] = model
            
            # 计算特征重要性
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[target] = model.feature_importances_
            
            # 验证集评估
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1] if len(np.unique(y_val[target])) > 1 else None
                
                results[target] = {
                    'accuracy': accuracy_score(y_val[target], y_pred),
                    'precision': precision_score(y_val[target], y_pred, zero_division=0),
                    'recall': recall_score(y_val[target], y_pred, zero_division=0),
                    'f1': f1_score(y_val[target], y_pred, zero_division=0),
                }
                
                if y_pred_proba is not None:
                    try:
                        results[target]['roc_auc'] = roc_auc_score(y_val[target], y_pred_proba)
                    except:
                        results[target]['roc_auc'] = 0.0
                
                print(f"  Accuracy: {results[target]['accuracy']:.4f}")
                print(f"  Precision: {results[target]['precision']:.4f}")
                print(f"  Recall: {results[target]['recall']:.4f}")
                print(f"  F1-Score: {results[target]['f1']:.4f}")
                if 'roc_auc' in results[target]:
                    print(f"  ROC-AUC: {results[target]['roc_auc']:.4f}")
        
        return results
    
    def predict(self, X, target_name='organ_abnormal'):
        """预测"""
        if target_name not in self.models:
            raise ValueError(f"模型 {target_name} 未训练")
        
        model = self.models[target_name]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        return predictions, probabilities
    
    def predict_all(self, X):
        """预测所有目标"""
        results = {}
        for target_name in self.models.keys():
            pred, proba = self.predict(X, target_name)
            results[target_name] = {
                'prediction': pred,
                'probability': proba
            }
        return results
    
    def get_feature_importance(self, target_name='organ_abnormal', top_n=20):
        """获取特征重要性"""
        if target_name not in self.feature_importance:
            return None
        
        importance = self.feature_importance[target_name]
        indices = np.argsort(importance)[::-1][:top_n]
        
        return {
            'indices': indices,
            'importance': importance[indices]
        }
    
    def save(self, filepath):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump({
            'models': self.models,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        data = joblib.load(filepath)
        self.models = data['models']
        self.model_type = data['model_type']
        self.feature_importance = data.get('feature_importance', {})
        print(f"模型已从 {filepath} 加载")

