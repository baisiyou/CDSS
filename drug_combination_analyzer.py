"""
药物组合分析系统
基于数据挖掘分析药物组合模式、疗效和风险
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class DrugCombinationAnalyzer:
    def __init__(self, data_path=None):
        """
        初始化药物组合分析器
        data_path: 数据文件路径，如果提供则自动加载数据
        """
        self.data = None
        self.drug_columns = None
        self.outcome_columns = ['death', 'ventilator', 'sepsis']
        # 器官功能异常结局（如果数据中有）
        self.organ_outcome_columns = ['kidney_abnormal', 'liver_abnormal', 'organ_abnormal']
        self.combination_stats = {}
        self.association_rules = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path, load_full_data=True):
        """
        加载数据
        load_full_data: 如果为False，只读取列名（用于节省内存）
        """
        if load_full_data:
            print(f"正在加载数据: {data_path}")
            self.data = pd.read_csv(data_path)
            print(f"数据形状: {self.data.shape}")
            self._identify_drug_columns()
            return self.data
        else:
            # 只读取列名，不加载数据（节省内存）
            print(f"正在读取数据文件列名: {data_path}")
            import pandas as pd
            # 只读取第一行来获取列名
            self.data = None  # 不加载完整数据
            df_columns = pd.read_csv(data_path, nrows=0)  # nrows=0 只读取列名
            self._identify_drug_columns_from_columns(df_columns.columns)
            return None
    
    def _identify_drug_columns_from_columns(self, columns):
        """从列名列表中识别药物列（不依赖self.data）"""
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
        
        self.drug_columns = [col for col in columns if col not in exclude_cols]
        print(f"识别到 {len(self.drug_columns)} 种药物（仅列名，未加载数据）")
        return self.drug_columns
    
    def _identify_drug_columns(self):
        """识别药物列（需要self.data已加载）"""
        if self.data is None:
            raise ValueError("数据未加载，无法识别药物列")
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
        
        self.drug_columns = [col for col in self.data.columns if col not in exclude_cols]
        print(f"识别到 {len(self.drug_columns)} 种药物")
        return self.drug_columns
    
    def get_drug_combinations(self, patient_data=None, min_support=0.01, max_combinations=1000):
        """
        获取药物组合统计
        patient_data: 如果提供，分析特定患者的药物组合
        min_support: 最小支持度（组合出现频率）
        max_combinations: 最大组合数量
        """
        if patient_data is not None:
            # 分析单个患者
            used_drugs = [drug for drug in self.drug_columns 
                         if drug in patient_data and patient_data[drug] > 0]
            return {
                'drugs': used_drugs,
                'count': len(used_drugs),
                'combinations': list(combinations(used_drugs, 2))
            }
        
        # 分析整个数据集
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print("正在分析药物组合...")
        combination_counts = Counter()
        total_patients = len(self.data)
        
        # 统计所有2-药物组合
        for idx, row in self.data.iterrows():
            used_drugs = [drug for drug in self.drug_columns if row[drug] > 0]
            if len(used_drugs) >= 2:
                for combo in combinations(sorted(used_drugs), 2):
                    combination_counts[combo] += 1
        
        # 过滤并排序
        min_count = int(total_patients * min_support)
        filtered_combinations = {
            combo: count for combo, count in combination_counts.items() 
            if count >= min_count
        }
        
        # 按频率排序
        sorted_combinations = sorted(
            filtered_combinations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_combinations]
        
        result = []
        for (drug1, drug2), count in sorted_combinations:
            support = count / total_patients
            result.append({
                'drug1': drug1,
                'drug2': drug2,
                'count': count,
                'support': support,
                'frequency': f"{support*100:.2f}%"
            })
        
        self.combination_stats = result
        print(f"发现 {len(result)} 个常见药物组合（支持度 >= {min_support*100:.1f}%）")
        return result
    
    def analyze_combination_outcomes(self, drug1, drug2, outcome='death'):
        """
        分析特定药物组合与结局的关联
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 支持所有结局类型（包括器官功能异常）
        all_outcomes = self.outcome_columns + self.organ_outcome_columns
        if outcome not in all_outcomes:
            # 如果数据中有该列，也允许使用
            if outcome not in self.data.columns:
                raise ValueError(f"结局变量必须是以下之一: {all_outcomes}，或数据中的其他列")
        
        # 筛选使用该组合的患者
        combo_mask = (self.data[drug1] > 0) & (self.data[drug2] > 0)
        combo_data = self.data[combo_mask]
        
        if len(combo_data) == 0:
            return {
                'error': f"未找到同时使用 {drug1} 和 {drug2} 的记录"
            }
        
        # 计算结局发生率
        outcome_rate = combo_data[outcome].mean()
        outcome_count = combo_data[outcome].sum()
        total_count = len(combo_data)
        
        # 计算对照组（不使用该组合）的结局发生率
        control_mask = ~combo_mask
        control_data = self.data[control_mask]
        control_outcome_rate = control_data[outcome].mean()
        control_count = len(control_data)
        
        # 计算相对风险
        if control_outcome_rate > 0:
            relative_risk = outcome_rate / control_outcome_rate
        else:
            relative_risk = np.inf if outcome_rate > 0 else 1.0
        
        # 计算风险差异
        risk_difference = outcome_rate - control_outcome_rate
        
        # 计算置信区间（简化版）
        se = np.sqrt(outcome_rate * (1 - outcome_rate) / total_count + 
                    control_outcome_rate * (1 - control_outcome_rate) / control_count)
        ci_lower = risk_difference - 1.96 * se
        ci_upper = risk_difference + 1.96 * se
        
        return {
            'drug1': drug1,
            'drug2': drug2,
            'outcome': outcome,
            'combo_outcome_rate': outcome_rate,
            'combo_outcome_count': int(outcome_count),
            'combo_total_count': total_count,
            'control_outcome_rate': control_outcome_rate,
            'control_outcome_count': int(control_data[outcome].sum()),
            'control_total_count': control_count,
            'relative_risk': relative_risk,
            'risk_difference': risk_difference,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'interpretation': self._interpret_risk(relative_risk, risk_difference)
        }
    
    def _interpret_risk(self, relative_risk, risk_difference):
        """解释风险"""
        if relative_risk > 1.5:
            return "高风险：该药物组合显著增加不良结局风险"
        elif relative_risk > 1.2:
            return "中等风险：该药物组合可能增加不良结局风险"
        elif relative_risk < 0.8:
            return "保护性：该药物组合可能降低不良结局风险"
        elif relative_risk < 0.9:
            return "可能保护性：该药物组合可能略微降低不良结局风险"
        else:
            return "中性：该药物组合对结局影响不明显"
    
    def find_effective_combinations(self, outcome='death', min_improvement=0.1, top_n=20):
        """
        发现有效的药物组合（降低不良结局风险）
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print(f"正在寻找有效药物组合（降低{outcome}风险）...")
        
        # 获取常见组合
        common_combos = self.get_drug_combinations(min_support=0.01, max_combinations=500)
        
        effective_combos = []
        
        for combo in common_combos:
            drug1, drug2 = combo['drug1'], combo['drug2']
            analysis = self.analyze_combination_outcomes(drug1, drug2, outcome)
            
            if 'error' not in analysis:
                # 如果相对风险 < 1，说明是保护性的
                if analysis['relative_risk'] < 1.0:
                    improvement = 1 - analysis['relative_risk']
                    if improvement >= min_improvement:
                        effective_combos.append({
                            'drug1': drug1,
                            'drug2': drug2,
                            'relative_risk': analysis['relative_risk'],
                            'risk_reduction': improvement,
                            'outcome_rate': analysis['combo_outcome_rate'],
                            'control_rate': analysis['control_outcome_rate'],
                            'count': analysis['combo_total_count'],
                            'interpretation': analysis['interpretation']
                        })
        
        # 按风险降低程度排序
        effective_combos.sort(key=lambda x: x['risk_reduction'], reverse=True)
        
        print(f"发现 {len(effective_combos)} 个有效药物组合")
        return effective_combos[:top_n]
    
    def find_risky_combinations(self, outcome='death', min_risk_increase=0.2, top_n=20):
        """
        发现高风险药物组合（增加不良结局风险）
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print(f"正在寻找高风险药物组合（增加{outcome}风险）...")
        
        # 获取常见组合
        common_combos = self.get_drug_combinations(min_support=0.01, max_combinations=500)
        
        risky_combos = []
        
        for combo in common_combos:
            drug1, drug2 = combo['drug1'], combo['drug2']
            analysis = self.analyze_combination_outcomes(drug1, drug2, outcome)
            
            if 'error' not in analysis:
                # 如果相对风险 > 1，说明是风险性的
                if analysis['relative_risk'] > 1.0:
                    risk_increase = analysis['relative_risk'] - 1.0
                    if risk_increase >= min_risk_increase:
                        risky_combos.append({
                            'drug1': drug1,
                            'drug2': drug2,
                            'relative_risk': analysis['relative_risk'],
                            'risk_increase': risk_increase,
                            'outcome_rate': analysis['combo_outcome_rate'],
                            'control_rate': analysis['control_outcome_rate'],
                            'count': analysis['combo_total_count'],
                            'interpretation': analysis['interpretation']
                        })
        
        # 按风险增加程度排序
        risky_combos.sort(key=lambda x: x['risk_increase'], reverse=True)
        
        print(f"发现 {len(risky_combos)} 个高风险药物组合")
        return risky_combos[:top_n]
    
    def analyze_patient_combination(self, patient_data, outcome='death'):
        """
        分析患者当前药物组合的风险和疗效
        patient_data: 字典或DataFrame，包含患者用药情况
        """
        if isinstance(patient_data, pd.DataFrame):
            patient_data = patient_data.iloc[0].to_dict()
        
        # 获取患者使用的药物
        used_drugs = [drug for drug in self.drug_columns 
                     if drug in patient_data and patient_data.get(drug, 0) > 0]
        
        if len(used_drugs) < 2:
            return {
                'warning': '患者使用的药物少于2种，无法进行组合分析',
                'drugs': used_drugs
            }
        
        # 分析所有药物组合
        combination_analyses = []
        risky_combos = []
        effective_combos = []
        
        for drug1, drug2 in combinations(used_drugs, 2):
            analysis = self.analyze_combination_outcomes(drug1, drug2, outcome)
            if 'error' not in analysis:
                combination_analyses.append(analysis)
                
                if analysis['relative_risk'] > 1.2:
                    risky_combos.append({
                        'drug1': drug1,
                        'drug2': drug2,
                        'relative_risk': analysis['relative_risk'],
                        'interpretation': analysis['interpretation']
                    })
                elif analysis['relative_risk'] < 0.9:
                    effective_combos.append({
                        'drug1': drug1,
                        'drug2': drug2,
                        'relative_risk': analysis['relative_risk'],
                        'interpretation': analysis['interpretation']
                    })
        
        # 计算总体风险评分
        if combination_analyses:
            avg_relative_risk = np.mean([a['relative_risk'] for a in combination_analyses])
            max_relative_risk = max([a['relative_risk'] for a in combination_analyses])
        else:
            avg_relative_risk = 1.0
            max_relative_risk = 1.0
        
        # 确定风险等级
        if max_relative_risk > 1.5:
            overall_risk = 'high'
        elif max_relative_risk > 1.2:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        return {
            'patient_drugs': used_drugs,
            'total_drugs': len(used_drugs),
            'total_combinations': len(list(combinations(used_drugs, 2))),
            'analyzed_combinations': len(combination_analyses),
            'overall_risk': overall_risk,
            'average_relative_risk': avg_relative_risk,
            'max_relative_risk': max_relative_risk,
            'risky_combinations': risky_combos,
            'effective_combinations': effective_combos,
            'all_combinations': combination_analyses[:10],  # 只返回前10个
            'recommendations': self._generate_combination_recommendations(
                risky_combos, effective_combos, overall_risk
            )
        }
    
    def _generate_combination_recommendations(self, risky_combos, effective_combos, overall_risk):
        """生成药物组合建议"""
        recommendations = []
        
        if overall_risk == 'high':
            recommendations.append("⚠️ 高风险：患者当前药物组合存在高风险，建议重新评估用药方案")
        elif overall_risk == 'medium':
            recommendations.append("🟡 中等风险：建议密切监测患者状况，考虑调整用药")
        
        if risky_combos:
            recommendations.append(f"🔴 发现 {len(risky_combos)} 个高风险药物组合：")
            for combo in risky_combos[:3]:  # 只显示前3个
                recommendations.append(
                    f"   - {combo['drug1']} + {combo['drug2']}: "
                    f"相对风险 {combo['relative_risk']:.2f}"
                )
        
        if effective_combos:
            recommendations.append(f"✅ 发现 {len(effective_combos)} 个有效药物组合：")
            for combo in effective_combos[:3]:  # 只显示前3个
                recommendations.append(
                    f"   - {combo['drug1']} + {combo['drug2']}: "
                    f"相对风险 {combo['relative_risk']:.2f}（保护性）"
                )
        
        if not recommendations:
            recommendations.append("✅ 当前药物组合风险较低，继续监测")
        
        return recommendations
    
    def analyze_drug_protective_effects(self, drug_name, min_risk_reduction=0.05, top_n=20):
        """
        分析特定药物与其他药物联用时，可能降低哪些不良结局风险
        
        Args:
            drug_name: 要分析的药物名称
            min_risk_reduction: 最小风险降低比例（0-1）
            top_n: 返回前N个保护性组合
        
        Returns:
            字典，包含每个不良结局的保护性组合列表
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        if drug_name not in self.drug_columns:
            return {
                'error': f'药物 {drug_name} 不在数据集中'
            }
        
        print(f"正在分析 {drug_name} 的保护性联用效果...")
        
        # 所有要分析的结局
        all_outcomes = self.outcome_columns.copy()
        
        # 如果数据中有器官功能异常列，也分析
        for outcome in self.organ_outcome_columns:
            if outcome in self.data.columns:
                all_outcomes.append(outcome)
        
        results = {}
        
        # 获取与目标药物联用的所有其他药物
        drug_mask = self.data[drug_name] > 0
        patients_with_drug = self.data[drug_mask]
        
        if len(patients_with_drug) == 0:
            return {
                'error': f'未找到使用 {drug_name} 的患者记录'
            }
        
        # 找出与目标药物经常联用的其他药物
        co_used_drugs = {}
        for other_drug in self.drug_columns:
            if other_drug == drug_name:
                continue
            if other_drug in patients_with_drug.columns:
                co_usage = ((patients_with_drug[other_drug] > 0).sum())
                if co_usage >= 10:  # 至少10个患者同时使用
                    co_used_drugs[other_drug] = co_usage
        
        # 按联用频率排序
        sorted_co_drugs = sorted(co_used_drugs.items(), key=lambda x: x[1], reverse=True)[:100]  # 只分析前100个
        
        # 对每个结局进行分析
        for outcome in all_outcomes:
            if outcome not in self.data.columns:
                continue
            
            protective_combos = []
            
            for other_drug, co_usage_count in sorted_co_drugs:
                try:
                    # 分析该组合与结局的关联
                    analysis = self.analyze_combination_outcomes(drug_name, other_drug, outcome)
                    
                    if 'error' not in analysis:
                        rr = analysis['relative_risk']
                        
                        # 如果相对风险 < 1，说明是保护性的
                        if rr < 1.0:
                            risk_reduction = 1 - rr
                            if risk_reduction >= min_risk_reduction:
                                protective_combos.append({
                                    'drug': other_drug,
                                    'relative_risk': rr,
                                    'risk_reduction': risk_reduction,
                                    'risk_reduction_percent': risk_reduction * 100,
                                    'combo_outcome_rate': analysis['combo_outcome_rate'],
                                    'control_outcome_rate': analysis['control_outcome_rate'],
                                    'combo_total_count': analysis['combo_total_count'],
                                    'interpretation': analysis['interpretation']
                                })
                except Exception as e:
                    # 如果该结局不支持，跳过
                    continue
            
            # 按风险降低程度排序
            protective_combos.sort(key=lambda x: x['risk_reduction'], reverse=True)
            results[outcome] = protective_combos[:top_n]
        
        # 生成总结
        summary = {
            'drug': drug_name,
            'total_outcomes_analyzed': len([k for k in results.keys() if len(results[k]) > 0]),
            'protective_combinations': {}
        }
        
        for outcome, combos in results.items():
            if len(combos) > 0:
                summary['protective_combinations'][outcome] = {
                    'count': len(combos),
                    'best_risk_reduction': combos[0]['risk_reduction_percent'] if combos else 0,
                    'best_combo_drug': combos[0]['drug'] if combos else None
                }
        
        results['summary'] = summary
        print(f"分析完成，发现 {summary['total_outcomes_analyzed']} 个结局有保护性组合")
        
        return results
    
    def get_drug_recommendations(self, current_drugs, outcome='death', top_n=10):
        """
        基于当前用药，推荐可能有益的额外药物
        current_drugs: 当前使用的药物列表
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 获取所有药物
        all_drugs = set(self.drug_columns)
        unused_drugs = all_drugs - set(current_drugs)
        
        recommendations = []
        
        for drug in unused_drugs:
            # 分析添加该药物后的效果
            # 简化：分析该药物与当前药物的组合效果
            best_risk_reduction = 0
            best_combo = None
            
            for current_drug in current_drugs:
                analysis = self.analyze_combination_outcomes(current_drug, drug, outcome)
                if 'error' not in analysis:
                    if analysis['relative_risk'] < 1.0:
                        risk_reduction = 1 - analysis['relative_risk']
                        if risk_reduction > best_risk_reduction:
                            best_risk_reduction = risk_reduction
                            best_combo = current_drug
            
            if best_risk_reduction > 0:
                recommendations.append({
                    'drug': drug,
                    'best_combo_with': best_combo,
                    'risk_reduction': best_risk_reduction,
                    'potential_benefit': f"与{best_combo}联用可能降低{outcome}风险{best_risk_reduction*100:.1f}%"
                })
        
        # 按风险降低程度排序
        recommendations.sort(key=lambda x: x['risk_reduction'], reverse=True)
        
        return recommendations[:top_n]
    
    def generate_summary_report(self, output_file=None):
        """生成药物组合分析摘要报告"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print("正在生成药物组合分析报告...")
        
        report = []
        report.append("=" * 60)
        report.append("药物组合分析报告")
        report.append("=" * 60)
        report.append(f"\n数据概览:")
        report.append(f"  总记录数: {len(self.data):,}")
        report.append(f"  药物种类: {len(self.drug_columns)}")
        report.append(f"  平均每患者用药数: {self.data[self.drug_columns].sum(axis=1).mean():.1f}")
        
        # 常见组合
        report.append(f"\n常见药物组合（前10）:")
        common_combos = self.get_drug_combinations(min_support=0.01, max_combinations=10)
        for i, combo in enumerate(common_combos[:10], 1):
            report.append(f"  {i}. {combo['drug1']} + {combo['drug2']}: {combo['frequency']}")
        
        # 高风险组合
        report.append(f"\n高风险药物组合（增加死亡风险，前5）:")
        risky = self.find_risky_combinations(outcome='death', min_risk_increase=0.2, top_n=5)
        for i, combo in enumerate(risky, 1):
            report.append(
                f"  {i}. {combo['drug1']} + {combo['drug2']}: "
                f"相对风险 {combo['relative_risk']:.2f} "
                f"(风险增加 {combo['risk_increase']*100:.1f}%)"
            )
        
        # 有效组合
        report.append(f"\n有效药物组合（降低死亡风险，前5）:")
        effective = self.find_effective_combinations(outcome='death', min_improvement=0.1, top_n=5)
        for i, combo in enumerate(effective, 1):
            report.append(
                f"  {i}. {combo['drug1']} + {combo['drug2']}: "
                f"相对风险 {combo['relative_risk']:.2f} "
                f"(风险降低 {combo['risk_reduction']*100:.1f}%)"
            )
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存到: {output_file}")
        else:
            print(report_text)
        
        return report_text

