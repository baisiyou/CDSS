"""
药物相互作用和风险预警模块
检测高风险药物组合（如抗生素+肾毒性药物）
"""

import pandas as pd
import numpy as np

class DrugInteractionWarning:
    def __init__(self):
        """初始化药物风险预警系统"""
        # 定义肾毒性药物列表
        self.nephrotoxic_drugs = [
            'vancomycin', 'gentamicin', 'tobramycin', 'amikacin',  # 氨基糖苷类
            'furosemide', 'bumetanide', 'bumex',  # 利尿剂
            'acyclovir', 'ganciclovir',  # 抗病毒药
            'amphotericin',  # 抗真菌药
            'cyclosporine', 'tacrolimus',  # 免疫抑制剂
            'nsaids', 'ibuprofen', 'ketorolac', 'toradol',  # NSAIDs
            'ace_inhibitors', 'lisinopril', 'enalapril',  # ACE抑制剂
            'arbs', 'losartan', 'cozaar',  # ARB
            'contrast_agents', 'iohexol', 'iopamidol', 'optiray', 'definity'  # 造影剂
        ]
        
        # 定义肝毒性药物列表
        self.hepatotoxic_drugs = [
            'acetaminophen', 'acetamin',  # 对乙酰氨基酚
            'amiodarone',  # 胺碘酮
            'methotrexate',  # 甲氨蝶呤
            'isoniazid',  # 异烟肼
            'valproic_acid',  # 丙戊酸
            'statins', 'atorvastatin', 'simvastatin', 'lipitor', 'zocor'  # 他汀类
        ]
        
        # 定义抗生素列表
        self.antibiotics = [
            'vancomycin', 'cefazolin', 'ceftriaxone', 'cefepime', 'cephulac',
            'piperacillin', 'meropenem', 'merrem', 'zosyn',
            'azithromycin', 'ciprofloxacin', 'levofloxacin', 'levaquin',
            'clindamycin', 'metronidazole', 'flagyl', 'nafcillin'
        ]
        
        # 高风险组合规则
        self.high_risk_combinations = {
            'antibiotic_nephrotoxic': {
                'description': '抗生素 + 肾毒性药物',
                'risk_level': 'high',
                'warning': '同时使用抗生素和肾毒性药物可能增加急性肾损伤风险，建议监测肾功能指标'
            },
            'multiple_nephrotoxic': {
                'description': '多种肾毒性药物联用',
                'risk_level': 'high',
                'warning': '多种肾毒性药物同时使用，肾功能损伤风险显著增加'
            },
            'hepatotoxic_combination': {
                'description': '肝毒性药物组合',
                'risk_level': 'medium',
                'warning': '肝毒性药物联用可能增加肝功能异常风险，建议监测肝功能指标'
            }
        }
    
    def normalize_drug_name(self, drug_name):
        """标准化药物名称（处理空格和大小写）"""
        return drug_name.lower().replace(' ', '_').replace('-', '_')
    
    def check_drug_category(self, drug_name, category_list):
        """检查药物是否属于某个类别"""
        normalized = self.normalize_drug_name(drug_name)
        return any(cat in normalized for cat in category_list)
    
    def analyze_patient_drugs(self, patient_data, drug_columns):
        """
        分析患者用药情况
        patient_data: 单行DataFrame或字典，包含药物使用情况（0/1编码）
        drug_columns: 药物列名列表
        """
        # 获取患者使用的药物
        used_drugs = []
        for col in drug_columns:
            if col in patient_data:
                if isinstance(patient_data, pd.DataFrame):
                    value = patient_data[col].iloc[0] if len(patient_data) > 0 else 0
                else:
                    value = patient_data.get(col, 0)
                
                if value > 0:
                    used_drugs.append(col)
        
        # 分类药物
        nephrotoxic_used = []
        hepatotoxic_used = []
        antibiotics_used = []
        
        for drug in used_drugs:
            if self.check_drug_category(drug, self.nephrotoxic_drugs):
                nephrotoxic_used.append(drug)
            if self.check_drug_category(drug, self.hepatotoxic_drugs):
                hepatotoxic_used.append(drug)
            if self.check_drug_category(drug, self.antibiotics):
                antibiotics_used.append(drug)
        
        return {
            'all_drugs': used_drugs,
            'nephrotoxic': nephrotoxic_used,
            'hepatotoxic': hepatotoxic_used,
            'antibiotics': antibiotics_used
        }
    
    def check_high_risk_combinations(self, drug_analysis):
        """检查高风险药物组合"""
        warnings = []
        risk_score = 0
        
        # 检查：抗生素 + 肾毒性药物
        if drug_analysis['antibiotics'] and drug_analysis['nephrotoxic']:
            warnings.append({
                'type': 'antibiotic_nephrotoxic',
                'drugs': drug_analysis['antibiotics'] + drug_analysis['nephrotoxic'],
                **self.high_risk_combinations['antibiotic_nephrotoxic']
            })
            risk_score += 3
        
        # 检查：多种肾毒性药物
        if len(drug_analysis['nephrotoxic']) >= 2:
            warnings.append({
                'type': 'multiple_nephrotoxic',
                'drugs': drug_analysis['nephrotoxic'],
                **self.high_risk_combinations['multiple_nephrotoxic']
            })
            risk_score += 2
        
        # 检查：肝毒性药物组合
        if len(drug_analysis['hepatotoxic']) >= 2:
            warnings.append({
                'type': 'hepatotoxic_combination',
                'drugs': drug_analysis['hepatotoxic'],
                **self.high_risk_combinations['hepatotoxic_combination']
            })
            risk_score += 1
        
        return warnings, risk_score
    
    def assess_lab_indicators(self, patient_data, lab_columns):
        """
        评估实验室指标
        lab_columns: 实验室指标列名列表，如 ['bun', 'inr', 'albu_lab', 'creatinine']
        """
        lab_status = {}
        abnormal_count = 0
        
        # 定义正常范围（标准化后的值）
        normal_ranges = {
            'bun': (-2, 1.5),  # BUN正常范围
            'inr': (-2, 1.2),  # INR正常范围
            'albu_lab': (-1.0, 2),  # 白蛋白正常范围
            'creatinine': (-2, 1.5),  # 肌酐正常范围
            'lactate': (-2, 1.5),  # 乳酸正常范围
        }
        
        for lab in lab_columns:
            if lab in patient_data:
                if isinstance(patient_data, pd.DataFrame):
                    value = patient_data[lab].iloc[0] if len(patient_data) > 0 else None
                else:
                    value = patient_data.get(lab, None)
                
                if value is not None and not np.isnan(value):
                    normal_range = normal_ranges.get(lab, (-3, 3))
                    is_abnormal = value < normal_range[0] or value > normal_range[1]
                    
                    lab_status[lab] = {
                        'value': value,
                        'normal_range': normal_range,
                        'is_abnormal': is_abnormal
                    }
                    
                    if is_abnormal:
                        abnormal_count += 1
        
        return lab_status, abnormal_count
    
    def generate_warning(self, patient_data, drug_columns, lab_columns=None):
        """
        生成综合预警
        """
        # 分析药物
        drug_analysis = self.analyze_patient_drugs(patient_data, drug_columns)
        
        # 检查高风险组合
        warnings, risk_score = self.check_high_risk_combinations(drug_analysis)
        
        # 评估实验室指标
        lab_status = {}
        lab_abnormal_count = 0
        if lab_columns:
            lab_status, lab_abnormal_count = self.assess_lab_indicators(patient_data, lab_columns)
            # 如果实验室指标异常，增加风险评分
            if lab_abnormal_count > 0:
                risk_score += lab_abnormal_count
        
        # 确定总体风险等级
        if risk_score >= 5:
            overall_risk = 'high'
        elif risk_score >= 3:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        return {
            'overall_risk': overall_risk,
            'risk_score': risk_score,
            'drug_analysis': drug_analysis,
            'warnings': warnings,
            'lab_status': lab_status,
            'lab_abnormal_count': lab_abnormal_count,
            'recommendations': self._generate_recommendations(warnings, lab_status, overall_risk)
        }
    
    def _generate_recommendations(self, warnings, lab_status, overall_risk):
        """生成建议"""
        recommendations = []
        
        if overall_risk == 'high':
            recommendations.append("⚠️ 高风险：建议立即评估用药方案，考虑调整药物剂量或更换替代药物")
            recommendations.append("建议密切监测肝肾功能指标，每24-48小时复查")
        
        if warnings:
            for warning in warnings:
                recommendations.append(f"💊 {warning['warning']}")
        
        if lab_status:
            abnormal_labs = [lab for lab, status in lab_status.items() if status.get('is_abnormal', False)]
            if abnormal_labs:
                recommendations.append(f"🔬 实验室指标异常：{', '.join(abnormal_labs)}，建议复查并评估")
        
        if not recommendations:
            recommendations.append("✅ 当前用药方案风险较低，继续监测")
        
        return recommendations

