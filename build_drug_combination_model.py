#!/usr/bin/env python3
"""
构建药物组合数据模型
将大型CSV文件转换为轻量级预计算模型，保持功能效果
"""

import pandas as pd
import numpy as np
import joblib
from collections import Counter
from itertools import combinations
import os
import sys

def identify_drug_columns(df):
    """识别药物列"""
    exclude_cols = [
        'Unnamed: 0', 'patientunitstayid', 'hospitalid', 'time_window',
        'death', 'ventilator', 'sepsis',
        'bmi_underweight', 'bmi_normal', 'bmi_overweight', 'bmi_obesity',
        'race_african', 'race_hispanic', 'race_caucasion', 'race_asian', 'race_native',
        'sex_is_male', 'sex_is_female',
        '< 30', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70 - 79', '80 - 89', '> 89',
        'o2sat', 'pao2', 'paco2', 'ph', 'albu_lab', 'bands', 'bun', 'hct', 
        'inr', 'lactate', 'platelets', 'wbc'
    ]
    return [col for col in df.columns if col not in exclude_cols]

def calculate_combination_stats(df, drug_columns, outcome_columns, min_support=0.01):
    """计算所有药物组合的统计信息"""
    print("正在计算药物组合统计...")
    
    combination_stats = {}
    total_patients = len(df)
    min_count = int(total_patients * min_support)
    
    # 统计所有2-药物组合
    combination_counts = Counter()
    for idx, row in df.iterrows():
        used_drugs = [drug for drug in drug_columns if row[drug] > 0]
        if len(used_drugs) >= 2:
            for combo in combinations(sorted(used_drugs), 2):
                combination_counts[combo] += 1
    
    # 过滤并计算统计信息
    for (drug1, drug2), count in combination_counts.items():
        if count >= min_count:
            support = count / total_patients
            
            # 计算与各结局的关联
            combo_mask = (df[drug1] > 0) & (df[drug2] > 0)
            combo_data = df[combo_mask]
            control_mask = ~combo_mask
            control_data = df[control_mask]
            
            outcome_stats = {}
            for outcome in outcome_columns:
                if outcome in df.columns:
                    combo_rate = combo_data[outcome].mean() if len(combo_data) > 0 else 0
                    control_rate = control_data[outcome].mean() if len(control_data) > 0 else 0
                    
                    if control_rate > 0:
                        relative_risk = combo_rate / control_rate
                    else:
                        relative_risk = np.inf if combo_rate > 0 else 1.0
                    
                    outcome_stats[outcome] = {
                        'relative_risk': float(relative_risk),
                        'combo_rate': float(combo_rate),
                        'control_rate': float(control_rate),
                        'combo_count': int(combo_data[outcome].sum()),
                        'combo_total': len(combo_data),
                        'control_count': int(control_data[outcome].sum()),
                        'control_total': len(control_data)
                    }
            
            combination_stats[(drug1, drug2)] = {
                'count': count,
                'support': support,
                'outcomes': outcome_stats
            }
    
    print(f"计算了 {len(combination_stats)} 个常见组合的统计信息")
    return combination_stats

def build_model(data_path, output_path='models/drug_combination_model.pkl', min_support=0.01):
    """构建药物组合模型"""
    print("=" * 60)
    print("构建药物组合数据模型")
    print("=" * 60)
    print(f"数据文件: {data_path}")
    print(f"输出模型: {output_path}")
    print()
    
    # 读取数据
    print("正在读取数据文件...")
    df = pd.read_csv(data_path)
    print(f"数据形状: {df.shape}")
    
    # 识别药物列和结局列
    drug_columns = identify_drug_columns(df)
    outcome_columns = ['death', 'ventilator', 'sepsis']
    organ_outcome_columns = ['kidney_abnormal', 'liver_abnormal', 'organ_abnormal']
    
    # 添加器官功能异常列（如果存在）
    for col in organ_outcome_columns:
        if col in df.columns and col not in outcome_columns:
            outcome_columns.append(col)
    
    print(f"识别到 {len(drug_columns)} 种药物")
    print(f"结局变量: {outcome_columns}")
    print()
    
    # 计算组合统计
    combination_stats = calculate_combination_stats(df, drug_columns, outcome_columns, min_support)
    
    # 构建模型数据
    model_data = {
        'drug_columns': drug_columns,
        'outcome_columns': outcome_columns,
        'combination_stats': combination_stats,
        'total_patients': len(df),
        'data_shape': df.shape,
        'version': '1.0'
    }
    
    # 保存模型
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model_data, output_path)
    
    # 计算模型大小
    model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    data_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
    
    print()
    print("=" * 60)
    print("模型构建完成！")
    print("=" * 60)
    print(f"原始数据大小: {data_size:.1f} MB")
    print(f"模型大小: {model_size:.1f} MB")
    print(f"压缩比: {data_size/model_size:.1f}x")
    print(f"组合数量: {len(combination_stats)}")
    print(f"模型文件: {output_path}")
    print()
    print("✅ 模型已保存，可以替代原始数据文件使用")
    
    return model_data

if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'eicu_mimic_lab_time.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'models/drug_combination_model.pkl'
    min_support = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    
    if not os.path.exists(data_path):
        print(f"❌ 错误：数据文件不存在: {data_path}")
        sys.exit(1)
    
    try:
        model_data = build_model(data_path, output_path, min_support)
        print("\n✅ 成功！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

