# 临床决策支持系统 (CDSS)

## 系统概述

本系统是一个基于机器学习的临床决策支持系统，旨在辅助医生：
- 判断是否需要调整用药
- 预测药物不良反应（肝肾功能异常）
- 推荐治疗方案
- 预警高风险药物组合

## 主要功能

### 1. 肝肾功能异常预测
- 基于患者用药情况和实验室指标，预测是否会出现：
  - 肾功能异常（肌酐升高、BUN异常）
  - 肝功能异常（转氨酶异常、INR升高）

### 2. 药物组合风险预警
- 检测高风险药物组合：
  - 抗生素 + 肾毒性药物联用
  - 多种肾毒性药物联用
  - 肝毒性药物组合
- 基于实验室指标评估当前风险状态

## 系统架构

```
临床决策支持系统（CDSS）/
├── data_preprocessing.py          # 数据预处理模块
├── prediction_models.py          # 预测模型模块
├── drug_interaction_warning.py    # 药物相互作用预警模块
├── drug_combination_analyzer.py   # 药物组合分析模块
├── train_models.py               # 模型训练脚本
├── cdss_api.py                   # Flask API服务
├── drug_combination_analyzer.html # 药物组合分析前端界面
├── requirements.txt              # Python依赖
├── README.md                     # 说明文档
├── 一键启动并打开.sh              # 一键启动脚本
├── eicu_mimic_lab_time.csv        # 训练数据
└── models/                       # 模型保存目录（训练后生成）
    ├── organ_function_predictor.pkl
    └── preprocessor.pkl
```

## 安装和使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

首先需要训练预测模型：

```bash
python train_models.py
```

这将：
- 加载 `eicu_mimic_lab_time.csv` 数据
- 预处理数据并创建目标标签
- 训练随机森林模型
- 保存模型到 `models/` 目录

**注意**：如果模型文件已存在，可以跳过此步骤。

### 3. 启动系统

#### 方法1：一键启动（推荐）

```bash
chmod +x 一键启动并打开.sh
./一键启动并打开.sh
```

这个脚本会：
- 自动检查并停止旧进程
- 启动API服务（端口5003）
- 等待服务就绪
- 自动打开前端界面

#### 方法2：手动启动

```bash
# 启动API服务
python cdss_api.py
```

API服务将在 `http://localhost:5003` 启动。

### 4. 使用Web界面

1. 打开 `drug_combination_analyzer.html` 文件（在浏览器中打开）
2. 选择或搜索药物（支持中文和英文）
3. 点击"开始分析"按钮
4. 查看分析结果：
   - 总体风险评估
   - 多器官功能障碍预测
   - 推荐药物

**注意**：系统支持多语言切换（中文/English/Français），可在页面右上角切换。

## API接口

### 健康检查
```
GET /health
```

### 预测肝肾功能异常
```
POST /predict
Content-Type: application/json

{
  "bun": 1.2,
  "inr": 0.8,
  "vancomycin": 1,
  "furosemide": 1,
  ...
}
```

### 药物组合风险预警
```
POST /warn
Content-Type: application/json

{
  "vancomycin": 1,
  "furosemide": 1,
  "bun": 2.0,
  ...
}
```

### 综合分析
```
POST /analyze
Content-Type: application/json

{
  "patientunitstayid": "12345",
  "bun": 1.2,
  "vancomycin": 1,
  ...
}
```

### 药物组合分析
```
POST /drug_combinations
Content-Type: application/json

{
  "aspirin": 1,
  "prednisone": 1,
  "piperacillin": 1
}
```

### 获取药物列表
```
GET /drugs/list?limit=1000
```

### 获取推荐药物
```
POST /drugs/recommend
Content-Type: application/json

{
  "drugs": ["aspirin", "prednisone"]
}
```

## 技术实现

### 预测模型
- **算法**: 随机森林 (Random Forest)
- **特征**: 药物使用情况、实验室指标、患者基本信息
- **目标**: 二分类（正常/异常）

### 风险预警规则
- **肾毒性药物**: 氨基糖苷类、利尿剂、NSAIDs、ACE抑制剂、造影剂等
- **肝毒性药物**: 对乙酰氨基酚、胺碘酮、他汀类等
- **高风险组合**: 抗生素+肾毒性药物、多种肾毒性药物联用

### 实验室指标评估
- BUN (血尿素氮): 正常范围 (-2, 1.5)
- INR (国际标准化比值): 正常范围 (-2, 1.2)
- 白蛋白: 正常范围 (-1.0, 2)

## 使用示例

### Python代码示例

```python
from prediction_models import OrganFunctionPredictor
from drug_interaction_warning import DrugInteractionWarning
import pandas as pd

# 加载模型
predictor = OrganFunctionPredictor()
predictor.load('models/organ_function_predictor.pkl')

# 预测
patient_data = pd.DataFrame([{
    'vancomycin': 1,
    'furosemide': 1,
    'bun': 2.0,
    # ... 其他特征
}])

X = preprocessor.extract_features(patient_data)
X_scaled = preprocessor.scale_features(X)
predictions = predictor.predict_all(X_scaled)

# 预警
warning_system = DrugInteractionWarning()
warning = warning_system.generate_warning(
    patient_data, 
    drug_columns, 
    lab_columns
)
```

## 注意事项

1. **数据标准化**: 实验室指标需要标准化后才能输入模型
2. **模型更新**: 建议定期使用新数据重新训练模型
3. **临床验证**: 系统预测结果仅供参考，需结合临床判断
4. **数据隐私**: 确保患者数据的安全和隐私保护

## 未来改进

- [ ] 支持更多实验室指标
- [ ] 增加时间序列预测
- [ ] 集成更多药物相互作用数据库
- [ ] 优化模型性能
- [ ] 添加模型解释性分析

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请联系开发团队。

