"""
临床决策支持系统 API
提供预测和预警服务的Flask API
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sys
from prediction_models import OrganFunctionPredictor
from drug_interaction_warning import DrugInteractionWarning
from drug_combination_analyzer import DrugCombinationAnalyzer

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
predictor = None
preprocessor = None
warning_system = None
combination_analyzer = None

def load_models():
    """加载模型和预处理器"""
    global predictor, preprocessor, warning_system, combination_analyzer
    
    try:
        model_dir = 'models'
        model_path = os.path.join(model_dir, 'organ_function_predictor.pkl')
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        
        if os.path.exists(model_path):
            try:
                predictor = OrganFunctionPredictor()
                predictor.load(model_path)
                print("模型加载成功")
                if hasattr(predictor, 'models') and predictor.models:
                    print(f"  已加载模型: {list(predictor.models.keys())}")
                else:
                    print("  警告：模型对象为空")
            except Exception as e:
                print(f"  错误：模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                predictor = None
        else:
            print("警告：模型文件不存在，请先运行 train_models.py")
        
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            print("预处理器加载成功")
        else:
            print("警告：预处理器文件不存在")
        
        warning_system = DrugInteractionWarning()
        print("药物预警系统初始化成功")
        
        # 初始化药物组合分析器
        combination_analyzer = DrugCombinationAnalyzer()
        data_path = 'eicu_mimic_lab_time.csv'
        if os.path.exists(data_path):
            try:
                # 通过环境变量控制是否加载完整数据
                # LOAD_FULL_DATA=true 时加载完整数据（需要更多内存，可能超过免费版512MB限制）
                # 默认加载完整数据，如果内存不足会失败
                load_full_data = os.environ.get('LOAD_FULL_DATA', 'true').lower() == 'true'
                
                if load_full_data:
                    print("正在加载完整数据（这可能需要一些时间和内存）...")
                    combination_analyzer.load_data(data_path, load_full_data=True)
                    print("药物组合分析系统初始化成功（完整数据已加载）")
                else:
                    # 只读取列名，不加载完整数据（节省内存，适用于免费版512MB限制）
                    print("正在读取药物列表（仅列名，不加载完整数据以节省内存）...")
                    combination_analyzer.load_data(data_path, load_full_data=False)
                    print("药物列表加载成功（完整数据分析功能不可用，但药物列表可用）")
            except Exception as e:
                print(f"警告：数据加载失败: {e}")
                print("如果内存不足，可以设置环境变量 LOAD_FULL_DATA=false 来只加载列名")
                print("药物组合分析功能将不可用，但其他功能正常")
                import traceback
                traceback.print_exc()
        else:
            print("警告：数据文件不存在，药物组合分析功能将不可用")
    except Exception as e:
        print(f"错误：加载模型时发生异常: {e}")
        import traceback
        traceback.print_exc()
        print("继续启动服务，但某些功能可能不可用")

@app.route('/', methods=['GET'])
def index():
    """API根路径，返回API文档"""
    return jsonify({
        'name': '临床决策支持系统 (CDSS) API',
        'version': '1.0.0',
        'description': '辅助医生判断用药调整、预测不良反应、推荐治疗方案',
        'endpoints': {
            'GET /': 'API文档（当前页面）',
            'GET /health': '健康检查',
            'POST /predict': '预测肝肾功能异常',
            'POST /warn': '药物组合风险预警',
            'POST /analyze': '综合分析（预测+预警）',
            'POST /drug_combinations': '分析患者药物组合',
            'GET /drug_combinations/common': '获取常见药物组合',
            'GET /drug_combinations/risky': '获取高风险药物组合',
            'GET /drug_combinations/effective': '获取有效药物组合'
        },
        'usage': {
            'predict': {
                'method': 'POST',
                'url': '/predict',
                'content_type': 'application/json',
                'example': {
                    'bun': 1.2,
                    'inr': 0.8,
                    'vancomycin': 1,
                    'furosemide': 1
                }
            },
            'warn': {
                'method': 'POST',
                'url': '/warn',
                'content_type': 'application/json',
                'example': {
                    'vancomycin': 1,
                    'furosemide': 1,
                    'bun': 2.0
                }
            },
            'analyze': {
                'method': 'POST',
                'url': '/analyze',
                'content_type': 'application/json',
                'description': '综合分析，包含预测和预警'
            }
        },
        'web_interface': '打开 web_interface.html 文件使用Web界面',
        'documentation': '查看 README.md 获取完整文档'
    })

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    """处理favicon请求"""
    return '', 204  # 返回空响应，避免404错误

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'preprocessor_loaded': preprocessor is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """预测肝肾功能异常"""
    try:
        data = request.json
        
        if predictor is None:
            return jsonify({
                'error': '预测模型未加载，请先运行 train_models.py 训练模型',
                'model_status': 'not_loaded'
            }), 500
        
        if preprocessor is None:
            return jsonify({
                'error': '预处理器未加载，请先运行 train_models.py 训练模型',
                'preprocessor_status': 'not_loaded'
            }), 500
        
        # 检查模型是否有效
        if not hasattr(predictor, 'models') or not predictor.models:
            return jsonify({
                'error': '模型对象无效，请重新训练模型',
                'model_status': 'invalid'
            }), 500
        
        # 将输入转换为DataFrame
        patient_data = pd.DataFrame([data])
        
        # 确保所有特征列都存在，并按照训练时的顺序排列
        if hasattr(preprocessor, 'feature_columns') and preprocessor.feature_columns:
            # 创建完整的特征DataFrame，缺失的列填充为0
            feature_dict = {}
            for col in preprocessor.feature_columns:
                if col in patient_data.columns:
                    feature_dict[col] = patient_data[col].iloc[0] if len(patient_data) > 0 else 0
                else:
                    feature_dict[col] = 0
            
            # 按照feature_columns的顺序创建DataFrame
            X = pd.DataFrame([feature_dict])[preprocessor.feature_columns]
        else:
            # 如果没有feature_columns，使用extract_features方法
            X = preprocessor.extract_features(patient_data)
        
        X = X.fillna(0)
        
        # 标准化（使用scaler的transform方法）
        # 预处理器在训练时已经fit过scaler，这里只需要transform
        X_scaled = preprocessor.scaler.transform(X)
        
        # 预测
        results = predictor.predict_all(X_scaled[0:1])
        
        # 格式化结果
        predictions = {}
        for target, result in results.items():
            pred_value = result['prediction']
            prob_value = result['probability']
            
            # 处理numpy数组
            if hasattr(pred_value, '__len__') and not isinstance(pred_value, str):
                pred_value = int(pred_value[0]) if len(pred_value) > 0 else 0
            else:
                pred_value = int(pred_value)
            
            if prob_value is not None:
                if hasattr(prob_value, '__len__') and not isinstance(prob_value, str):
                    prob_value = float(prob_value[0]) if len(prob_value) > 0 else None
                else:
                    prob_value = float(prob_value)
            
            predictions[target] = {
                'prediction': pred_value,
                'probability': prob_value
            }
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/warn', methods=['POST'])
def warn():
    """药物组合风险预警"""
    try:
        data = request.json
        
        if warning_system is None:
            return jsonify({
                'error': '预警系统未初始化'
            }), 500
        
        # 获取药物列和实验室指标列
        patient_data = pd.DataFrame([data])
        
        # 从预处理器获取所有特征列
        if preprocessor is not None:
            all_features = preprocessor.feature_columns
        else:
            # 如果没有预处理器，从数据中推断
            all_features = list(data.keys())
        
        # 分离药物列和实验室指标列
        lab_columns = ['o2sat', 'pao2', 'paco2', 'ph', 'albu_lab', 'bands', 
                      'bun', 'hct', 'inr', 'lactate', 'platelets', 'wbc']
        drug_columns = [col for col in all_features if col not in lab_columns]
        
        # 生成预警
        warning_result = warning_system.generate_warning(
            patient_data, 
            drug_columns, 
            lab_columns
        )
        
        # 转换为可序列化的格式
        result = {
            'overall_risk': warning_result['overall_risk'],
            'risk_score': warning_result['risk_score'],
            'drug_analysis': {
                'all_drugs': warning_result['drug_analysis']['all_drugs'],
                'nephrotoxic': warning_result['drug_analysis']['nephrotoxic'],
                'hepatotoxic': warning_result['drug_analysis']['hepatotoxic'],
                'antibiotics': warning_result['drug_analysis']['antibiotics']
            },
            'warnings': warning_result['warnings'],
            'lab_status': {k: {
                'value': float(v['value']) if not np.isnan(v['value']) else None,
                'is_abnormal': v['is_abnormal']
            } for k, v in warning_result['lab_status'].items()},
            'lab_abnormal_count': warning_result['lab_abnormal_count'],
            'recommendations': warning_result['recommendations']
        }
        
        return jsonify({
            'success': True,
            'warning': result
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """综合分析：预测 + 预警"""
    try:
        data = request.json
        
        # 预测
        predict_response = predict()
        if predict_response[1] != 200:
            return predict_response
        
        # 预警
        warn_response = warn()
        if warn_response[1] != 200:
            return warn_response
        
        # 合并结果
        predict_data = predict_response[0].get_json()
        warn_data = warn_response[0].get_json()
        
        return jsonify({
            'success': True,
            'prediction': predict_data.get('predictions', {}),
            'warning': warn_data.get('warning', {}),
            'summary': _generate_summary(predict_data.get('predictions', {}), 
                                       warn_data.get('warning', {}))
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

def _generate_summary(predictions, warning):
    """生成综合摘要"""
    summary = []
    
    # 预测结果摘要
    if predictions:
        for target, result in predictions.items():
            if result.get('prediction') == 1:
                prob = result.get('probability', 0)
                summary.append(f"⚠️ 预测{target}异常 (概率: {prob:.2%})")
    
    # 预警摘要
    if warning:
        risk_level = warning.get('overall_risk', 'unknown')
        if risk_level == 'high':
            summary.append("🔴 高风险：药物组合存在高风险")
        elif risk_level == 'medium':
            summary.append("🟡 中等风险：建议密切监测")
        
        if warning.get('lab_abnormal_count', 0) > 0:
            summary.append(f"🔬 {warning['lab_abnormal_count']} 项实验室指标异常")
    
    if not summary:
        summary.append("✅ 当前状态良好，继续监测")
    
    return summary

@app.route('/drug_combinations', methods=['POST'])
def analyze_drug_combinations():
    """分析患者药物组合"""
    try:
        data = request.json
        
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': '药物组合分析系统未初始化',
                'message': '数据文件未加载，分析功能不可用（免费版内存限制，仅支持药物列表功能）'
            }), 503  # 503 Service Unavailable 更合适
        
        if combination_analyzer.data is None:
            return jsonify({
                'success': False,
                'error': '完整数据未加载',
                'message': '由于内存限制，完整数据分析功能不可用。当前仅支持药物列表查询。如需完整功能，请升级到付费计划。'
            }), 503  # 503 Service Unavailable
        
        # 将输入转换为DataFrame
        patient_data = pd.DataFrame([data])
        
        # 获取结局变量（默认为death）
        outcome = request.json.get('outcome', 'death')
        
        # 分析药物组合
        result = combination_analyzer.analyze_patient_combination(patient_data, outcome)
        
        # 转换为可序列化的格式
        serializable_result = {
            'patient_drugs': result.get('patient_drugs', []),
            'total_drugs': result.get('total_drugs', 0),
            'total_combinations': result.get('total_combinations', 0),
            'analyzed_combinations': result.get('analyzed_combinations', 0),
            'overall_risk': result.get('overall_risk', 'unknown'),
            'average_relative_risk': float(result.get('average_relative_risk', 1.0)),
            'max_relative_risk': float(result.get('max_relative_risk', 1.0)),
            'risky_combinations': result.get('risky_combinations', []),
            'effective_combinations': result.get('effective_combinations', []),
            'recommendations': result.get('recommendations', [])
        }
        
        return jsonify({
            'success': True,
            'analysis': serializable_result
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/drug_combinations/common', methods=['GET'])
def get_common_combinations():
    """获取常见药物组合"""
    try:
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': '药物组合分析系统未初始化',
                'message': '数据文件未加载，此功能不可用'
            }), 503
        
        if combination_analyzer.data is None:
            return jsonify({
                'success': False,
                'error': '完整数据未加载',
                'message': '由于内存限制，此功能不可用。当前仅支持药物列表查询。'
            }), 503
        
        min_support = float(request.args.get('min_support', 0.01))
        max_combinations = int(request.args.get('max_combinations', 50))
        
        combinations = combination_analyzer.get_drug_combinations(
            min_support=min_support,
            max_combinations=max_combinations
        )
        
        return jsonify({
            'success': True,
            'combinations': combinations
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/drug_combinations/risky', methods=['GET'])
def get_risky_combinations():
    """获取高风险药物组合"""
    try:
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': '药物组合分析系统未初始化',
                'message': '数据文件未加载，此功能不可用'
            }), 503
        
        if combination_analyzer.data is None:
            return jsonify({
                'success': False,
                'error': '完整数据未加载',
                'message': '由于内存限制，此功能不可用。当前仅支持药物列表查询。'
            }), 503
        
        outcome = request.args.get('outcome', 'death')
        min_risk_increase = float(request.args.get('min_risk_increase', 0.2))
        top_n = int(request.args.get('top_n', 20))
        
        risky = combination_analyzer.find_risky_combinations(
            outcome=outcome,
            min_risk_increase=min_risk_increase,
            top_n=top_n
        )
        
        return jsonify({
            'success': True,
            'risky_combinations': risky
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/drug_combinations/effective', methods=['GET'])
def get_effective_combinations():
    """获取有效药物组合"""
    try:
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': '药物组合分析系统未初始化',
                'message': '数据文件未加载，此功能不可用'
            }), 503
        
        if combination_analyzer.data is None:
            return jsonify({
                'success': False,
                'error': '完整数据未加载',
                'message': '由于内存限制，此功能不可用。当前仅支持药物列表查询。'
            }), 503
        
        outcome = request.args.get('outcome', 'death')
        min_improvement = float(request.args.get('min_improvement', 0.1))
        top_n = int(request.args.get('top_n', 20))
        
        effective = combination_analyzer.find_effective_combinations(
            outcome=outcome,
            min_improvement=min_improvement,
            top_n=top_n
        )
        
        return jsonify({
            'success': True,
            'effective_combinations': effective
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/drugs/list', methods=['GET'])
def get_drugs_list():
    """获取所有可用药物列表"""
    try:
        # 如果数据未加载，返回空列表（而不是500错误）
        if combination_analyzer is None or combination_analyzer.drug_columns is None:
            return jsonify({
                'success': True,
                'drugs': [],
                'total': 0,
                'filtered': 0,
                'warning': '药物组合分析系统未初始化，数据文件可能未加载'
            })
        
        search = request.args.get('search', '').lower()
        limit = int(request.args.get('limit', 1000))
        
        drugs = combination_analyzer.drug_columns
        
        # 搜索过滤
        if search:
            drugs = [d for d in drugs if search in d.lower()]
        
        # 限制数量
        drugs = sorted(drugs)[:limit]
        
        return jsonify({
            'success': True,
            'drugs': drugs,
            'total': len(combination_analyzer.drug_columns),
            'filtered': len(drugs)
        })
    
    except Exception as e:
        # 即使出错也返回200，但包含错误信息
        return jsonify({
            'success': False,
            'drugs': [],
            'total': 0,
            'filtered': 0,
            'error': str(e)
        })

@app.route('/drugs/protective-effects', methods=['GET', 'POST'])
def get_drug_protective_effects():
    """分析特定药物与其他药物联用时，可能降低哪些不良结局风险"""
    try:
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': '药物组合分析系统未初始化',
                'message': '数据文件未加载，此功能不可用'
            }), 503
        
        if combination_analyzer.data is None:
            return jsonify({
                'success': False,
                'error': '完整数据未加载',
                'message': '由于内存限制，此功能不可用。当前仅支持药物列表查询。'
            }), 503
        
        # 支持GET和POST请求
        if request.method == 'GET':
            drug_name = request.args.get('drug', '')
        else:
            data = request.json or {}
            drug_name = data.get('drug', '')
        
        if not drug_name:
            return jsonify({
                'error': '请提供药物名称（drug参数）'
            }), 400
        
        min_risk_reduction = float(request.args.get('min_risk_reduction', 0.05) if request.method == 'GET' else data.get('min_risk_reduction', 0.05))
        top_n = int(request.args.get('top_n', 20) if request.method == 'GET' else data.get('top_n', 20))
        
        results = combination_analyzer.analyze_drug_protective_effects(
            drug_name=drug_name,
            min_risk_reduction=min_risk_reduction,
            top_n=top_n
        )
        
        if 'error' in results:
            return jsonify(results), 400
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/drugs/recommend', methods=['POST'])
def get_drug_recommendations():
    """获取防止多器官功能障碍的推荐药物"""
    try:
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': '药物组合分析系统未初始化',
                'message': '数据文件未加载，此功能不可用'
            }), 503
        
        if combination_analyzer.data is None:
            return jsonify({
                'success': False,
                'error': '完整数据未加载',
                'message': '由于内存限制，此功能不可用。当前仅支持药物列表查询。'
            }), 503
        
        data = request.json
        current_drugs = data.get('drugs', [])
        
        if not current_drugs:
            return jsonify({
                'error': '请提供当前使用的药物列表'
            }), 400
        
        # 获取推荐药物（针对器官功能异常）
        recommendations = []
        
        # 针对肾功能异常推荐
        kidney_recs = combination_analyzer.get_drug_recommendations(
            current_drugs, 
            outcome='death',  # 使用death作为代理，实际应该基于器官功能
            top_n=5
        )
        
        # 分析每个推荐药物与当前药物的组合，看是否能降低器官功能异常风险
        protective_drugs = []
        for rec in kidney_recs:
            # 检查该药物是否与当前药物组合有保护性
            drug = rec['drug']
            has_protective_combo = False
            
            for current_drug in current_drugs:
                try:
                    # 分析组合效果（这里简化处理，实际可以更复杂）
                    analysis = combination_analyzer.analyze_combination_outcomes(
                        current_drug, drug, 'death'
                    )
                    if 'error' not in analysis and analysis['relative_risk'] < 0.9:
                        has_protective_combo = True
                        break
                except:
                    continue
            
            if has_protective_combo:
                protective_drugs.append({
                    'drug': drug,
                    'best_combo_with': rec.get('best_combo_with', ''),
                    'risk_reduction': rec.get('risk_reduction', 0),
                    'potential_benefit': rec.get('potential_benefit', ''),
                    'reason': f"与{rec.get('best_combo_with', '当前药物')}联用可能降低不良结局风险"
                })
        
        # 添加一些已知的保护性药物（基于临床知识）
        known_protective_drugs = {
            'n-acetylcysteine': {
                'reason': 'N-乙酰半胱氨酸：抗氧化剂，可能保护肝肾功能，减少氧化应激损伤',
                'category': '保护性药物'
            },
            'vitamin_e': {
                'reason': '维生素E：抗氧化，可能降低器官损伤风险',
                'category': '保护性药物'
            },
            'magnesium': {
                'reason': '镁：可能保护肾功能，维持电解质平衡',
                'category': '保护性药物'
            },
            'vitamin': {
                'reason': '维生素：可能支持器官功能，增强机体抵抗力',
                'category': '保护性药物'
            },
            'ascorbic': {
                'reason': '维生素C：抗氧化，可能保护器官功能',
                'category': '保护性药物'
            },
            'thiamine': {
                'reason': '维生素B1：可能支持器官代谢功能',
                'category': '保护性药物'
            },
            'folic': {
                'reason': '叶酸：可能支持器官功能',
                'category': '保护性药物'
            }
        }
        
        # 检查这些药物是否在数据中
        for drug_name, info in known_protective_drugs.items():
            matching_drugs = [d for d in combination_analyzer.drug_columns 
                            if drug_name.lower() in d.lower()]
            if matching_drugs and matching_drugs[0] not in current_drugs:
                protective_drugs.append({
                    'drug': matching_drugs[0],
                    'reason': info['reason'],
                    'category': info['category'],
                    'risk_reduction': 0.1,  # 默认值
                    'potential_benefit': info['reason']
                })
        
        # 去重并排序
        seen = set()
        unique_recs = []
        for rec in protective_drugs:
            if rec['drug'] not in seen:
                seen.add(rec['drug'])
                unique_recs.append(rec)
        
        unique_recs.sort(key=lambda x: x.get('risk_reduction', 0), reverse=True)
        
        return jsonify({
            'success': True,
            'recommendations': unique_recs[:10],  # 返回前10个
            'current_drugs': current_drugs,
            'total_recommendations': len(unique_recs)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# 在模块加载时初始化（gunicorn 启动时会执行）
# 使用延迟初始化，避免在导入时立即加载（可能影响启动速度）
_models_loaded = False

def initialize_models():
    """初始化模型和数据（用于 gunicorn 启动时调用）"""
    global _models_loaded
    if not _models_loaded:
        print("正在加载模型...")
        load_models()
        _models_loaded = True

# 使用 gunicorn 的 on_starting 钩子会在 worker 启动前执行
# 但更好的方式是在模块级别调用（每个 worker 都需要加载数据）
# 在应用启动时初始化（gunicorn 会在导入模块时执行这部分代码）
try:
    initialize_models()
except Exception as e:
    print(f"警告：初始化时加载模型失败: {e}")
    import traceback
    traceback.print_exc()
    # 不退出，允许服务启动，但功能会受限

if __name__ == '__main__':
    try:
        # 如果直接运行，确保模型已加载
        if not _models_loaded:
            initialize_models()
        
        # 验证路由是否注册
        print("\n验证路由注册...")
        routes = list(app.url_map.iter_rules())
        print(f"✅ 已注册 {len(routes)} 个路由:")
        for rule in routes:
            if 'static' not in str(rule):
                print(f"  {rule}")
        
        print("\n启动CDSS API服务器...")
        print("=" * 60)
        print("API端点:")
        print("  GET  /                      - API文档")
        print("  GET  /health                 - 健康检查")
        print("  POST /predict                - 预测肝肾功能异常")
        print("  POST /warn                   - 药物组合风险预警")
        print("  POST /analyze                - 综合分析")
        print("  POST /drug_combinations      - 分析患者药物组合")
        print("  GET  /drug_combinations/common   - 获取常见药物组合")
        print("  GET  /drug_combinations/risky    - 获取高风险药物组合")
        print("  GET  /drug_combinations/effective - 获取有效药物组合")
        print("  GET  /drugs/list             - 获取药物列表")
        print("  POST /drugs/recommend        - 获取药物推荐")
        print("  GET/POST /drugs/protective-effects - 分析药物保护性效果")
        print("=" * 60)
        
        # 获取端口号，优先使用环境变量（Render会设置PORT环境变量）
        PORT = int(os.environ.get('PORT', 5003))
        HOST = os.environ.get('HOST', '0.0.0.0')  # Render需要监听0.0.0.0
        
        print(f"\n服务器运行在 http://{HOST}:{PORT}")
        print(f"访问 http://localhost:{PORT} 查看API文档")
        print("打开 drug_combination_analyzer.html 使用药物组合分析界面")
        print("\n按 Ctrl+C 停止服务器")
        print("=" * 60)
        
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

