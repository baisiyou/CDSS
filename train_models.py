"""
模型训练脚本
训练肝肾功能异常预测模型
"""

import os
import pandas as pd
from data_preprocessing import DataPreprocessor
from prediction_models import OrganFunctionPredictor

def main():
    print("=" * 60)
    print("临床决策支持系统 - 模型训练")
    print("=" * 60)
    
    # 初始化
    preprocessor = DataPreprocessor()
    
    # 加载数据
    data_file = 'eicu_mimic_drug_lab.csv'
    if not os.path.exists(data_file):
        print(f"错误：找不到数据文件 {data_file}")
        return
    
    df = preprocessor.load_data(data_file)
    
    # 准备数据
    print("\n准备训练数据...")
    X, y = preprocessor.prepare_training_data(df)
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    print(f"目标变量: {y.columns.tolist()}")
    print(f"\n目标变量分布:")
    for col in y.columns:
        print(f"  {col}: {y[col].sum()} 阳性样本 ({y[col].mean()*100:.2f}%)")
    
    # 划分数据集
    print("\n划分训练集和验证集...")
    X_train, X_val, y_train, y_val = preprocessor.split_data(X, y, test_size=0.2)
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    
    # 标准化特征
    print("\n标准化特征...")
    X_train_scaled, X_val_scaled = preprocessor.scale_features(X_train, X_val)
    
    # 训练模型
    print("\n开始训练模型...")
    predictor = OrganFunctionPredictor(model_type='random_forest')
    results = predictor.train(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 保存模型
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'organ_function_predictor.pkl')
    predictor.save(model_path)
    
    # 保存预处理器
    import joblib
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\n预处理器已保存到: {preprocessor_path}")
    
    # 显示特征重要性
    print("\n" + "=" * 60)
    print("特征重要性 (Top 20)")
    print("=" * 60)
    for target in y.columns:
        importance = predictor.get_feature_importance(target, top_n=20)
        if importance:
            print(f"\n{target}:")
            for i, idx in enumerate(importance['indices']):
                if idx < len(X.columns):
                    feature_name = X.columns[idx]
                    imp_value = importance['importance'][i]
                    print(f"  {i+1:2d}. {feature_name:30s} {imp_value:.4f}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()

