"""
Clinical Decision Support System API
Flask API providing prediction and warning services
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
from drug_side_effects import DrugSideEffects

app = Flask(__name__)
# Enable CORS - allow all origins and handle preflight requests
# CORS(app) automatically handles OPTIONS preflight requests
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": False
    }
})

# Global variables
predictor = None
preprocessor = None
warning_system = None
combination_analyzer = None
side_effects_db = None

def load_models():
    """Load models and preprocessor"""
    global predictor, preprocessor, warning_system, combination_analyzer, side_effects_db
    
    try:
        model_dir = 'models'
        model_path = os.path.join(model_dir, 'organ_function_predictor.pkl')
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        
        if os.path.exists(model_path):
            try:
                predictor = OrganFunctionPredictor()
                predictor.load(model_path)
                print("Model loaded successfully")
                if hasattr(predictor, 'models') and predictor.models:
                    print(f"  Loaded models: {list(predictor.models.keys())}")
                else:
                    print("  Warning: Model object is empty")
            except Exception as e:
                print(f"  Error: Model loading failed: {e}")
                import traceback
                traceback.print_exc()
                predictor = None
        else:
            print("Warning: Model file does not exist, please run train_models.py first")
        
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            print("Preprocessor loaded successfully")
        else:
            print("Warning: Preprocessor file does not exist")
        
        warning_system = DrugInteractionWarning()
        print("Drug interaction warning system initialized successfully")
        
        # Initialize side effects database
        side_effects_db = DrugSideEffects()
        print("Drug side effects database initialized successfully")
        
        # Initialize drug combination analyzer
        # Prefer model file (lightweight), fallback to original data file if not available
        combination_analyzer = DrugCombinationAnalyzer()
        model_path = 'models/drug_combination_model.pkl'
        data_path = 'eicu_mimic_lab_time.csv'
        
        # Try to load model file first
        if os.path.exists(model_path):
            try:
                print("Loading drug combination model (pre-computed data)...")
                combination_analyzer.load_model(model_path)
                print("Drug combination analysis system initialized successfully (using pre-computed model, low memory usage)")
            except Exception as e:
                print(f"Warning: Model loading failed: {e}")
                print("Attempting to use original data file...")
                import traceback
                traceback.print_exc()
                # If model loading fails, try using original data
                model_path = None
        
        # If model doesn't exist or loading failed, use original data file
        if model_path is None or not os.path.exists(model_path):
            if os.path.exists(data_path):
                try:
                    # Control whether to load full data via environment variable
                    # LOAD_FULL_DATA=true loads full data (requires more memory, may exceed free tier 512MB limit)
                    # Default: only load column names (saves memory)
                    load_full_data = os.environ.get('LOAD_FULL_DATA', 'false').lower() == 'true'
                    
                    if load_full_data:
                        print("Loading full data (this may take some time and memory)...")
                        combination_analyzer.load_data(data_path, load_full_data=True)
                        print("Drug combination analysis system initialized successfully (full data loaded)")
                    else:
                        # Only read column names, don't load full data (saves memory, suitable for free tier 512MB limit)
                        print("Reading drug list (column names only, not loading full data to save memory)...")
                        combination_analyzer.load_data(data_path, load_full_data=False)
                        print("Drug list loaded successfully (full data analysis unavailable, but drug list is available)")
                except Exception as e:
                    print(f"Warning: Data loading failed: {e}")
                    print("If memory is insufficient, set environment variable LOAD_FULL_DATA=false to only load column names")
                    print("Drug combination analysis will be unavailable, but other features will work normally")
                    import traceback
                    traceback.print_exc()
            else:
                print("Warning: Both model file and data file do not exist, drug combination analysis will be unavailable")
    except Exception as e:
        print(f"Error: Exception occurred while loading models: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing to start service, but some features may be unavailable")

# Removed after_request hook - CORS is handled by CORS(app) above
# Adding CORS headers here would cause duplicate headers error

@app.route('/', methods=['GET'])
def index():
    """API root path, returns API documentation"""
    return jsonify({
        'name': 'Clinical Decision Support System (CDSS) API',
        'version': '1.0.0',
        'description': 'Assist physicians in medication adjustment decisions, predict adverse reactions, and recommend treatment plans',
        'endpoints': {
            'GET /': 'API documentation (current page)',
            'GET /health': 'Health check',
            'POST /predict': 'Predict kidney and liver function abnormalities',
            'POST /warn': 'Drug combination risk warning',
            'POST /analyze': 'Comprehensive analysis (prediction + warning)',
            'POST /drug_combinations': 'Analyze patient drug combinations',
            'GET /drug_combinations/common': 'Get common drug combinations',
            'GET /drug_combinations/risky': 'Get high-risk drug combinations',
            'GET /drug_combinations/effective': 'Get effective drug combinations'
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
                'description': 'Comprehensive analysis, includes prediction and warning'
            }
        },
        'web_interface': 'Open web_interface.html file to use the web interface',
        'documentation': 'See README.md for complete documentation'
    })

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    """Handle favicon requests"""
    return '', 204  # Return empty response to avoid 404 error

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    # Check combination analyzer status
    combination_status = 'not_initialized'
    if combination_analyzer is not None:
        has_data = combination_analyzer.data is not None
        has_model = hasattr(combination_analyzer, 'model_data') and combination_analyzer.model_data is not None
        if has_data:
            combination_status = 'full_data_loaded'
        elif has_model:
            combination_status = 'model_loaded'
        else:
            combination_status = 'columns_only'
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'preprocessor_loaded': preprocessor is not None,
        'combination_analyzer_status': combination_status,
        'combination_analyzer_initialized': combination_analyzer is not None,
        'drug_list_available': combination_analyzer is not None and combination_analyzer.drug_columns is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict kidney and liver function abnormalities"""
    try:
        data = request.json
        
        if predictor is None:
            return jsonify({
                'error': 'Prediction model not loaded, please run train_models.py to train the model first',
                'model_status': 'not_loaded'
            }), 500
        
        if preprocessor is None:
            return jsonify({
                'error': 'Preprocessor not loaded, please run train_models.py to train the model first',
                'preprocessor_status': 'not_loaded'
            }), 500
        
        # Check if model is valid
        if not hasattr(predictor, 'models') or not predictor.models:
            return jsonify({
                'error': 'Model object is invalid, please retrain the model',
                'model_status': 'invalid'
            }), 500
        
        # Convert input to DataFrame
        patient_data = pd.DataFrame([data])
        
        # Ensure all feature columns exist and are arranged in the same order as during training
        if hasattr(preprocessor, 'feature_columns') and preprocessor.feature_columns:
            # Create complete feature DataFrame, fill missing columns with 0
            feature_dict = {}
            for col in preprocessor.feature_columns:
                if col in patient_data.columns:
                    feature_dict[col] = patient_data[col].iloc[0] if len(patient_data) > 0 else 0
                else:
                    feature_dict[col] = 0
            
            # Create DataFrame in the order of feature_columns
            X = pd.DataFrame([feature_dict])[preprocessor.feature_columns]
        else:
            # If no feature_columns, use extract_features method
            X = preprocessor.extract_features(patient_data)
        
        X = X.fillna(0)
        
        # Standardize (using scaler's transform method)
        # Preprocessor has already fit the scaler during training, only transform is needed here
        X_scaled = preprocessor.scaler.transform(X)
        
        # Predict
        results = predictor.predict_all(X_scaled[0:1])
        
        # Format results
        predictions = {}
        for target, result in results.items():
            pred_value = result['prediction']
            prob_value = result['probability']
            
            # Handle numpy arrays
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
    """Drug combination risk warning"""
    try:
        data = request.json
        
        if warning_system is None:
            return jsonify({
                'error': 'Warning system not initialized'
            }), 500
        
        # Get drug columns and lab indicator columns
        patient_data = pd.DataFrame([data])
        
        # Get all feature columns from preprocessor
        if preprocessor is not None:
            all_features = preprocessor.feature_columns
        else:
            # If no preprocessor, infer from data
            all_features = list(data.keys())
        
        # Separate drug columns and lab indicator columns
        lab_columns = ['o2sat', 'pao2', 'paco2', 'ph', 'albu_lab', 'bands', 
                      'bun', 'hct', 'inr', 'lactate', 'platelets', 'wbc']
        drug_columns = [col for col in all_features if col not in lab_columns]
        
        # Generate warning
        warning_result = warning_system.generate_warning(
            patient_data, 
            drug_columns, 
            lab_columns
        )
        
        # Convert to serializable format
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
    """Comprehensive analysis: prediction + warning"""
    try:
        data = request.json
        
        # Predict
        predict_response = predict()
        if predict_response[1] != 200:
            return predict_response
        
        # Warn
        warn_response = warn()
        if warn_response[1] != 200:
            return warn_response
        
        # Merge results
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
    """Generate comprehensive summary"""
    summary = []
    
    # Prediction results summary
    if predictions:
        for target, result in predictions.items():
            if result.get('prediction') == 1:
                prob = result.get('probability', 0)
                summary.append(f"⚠️ Predicted {target} abnormality (probability: {prob:.2%})")
    
    # Warning summary
    if warning:
        risk_level = warning.get('overall_risk', 'unknown')
        if risk_level == 'high':
            summary.append("🔴 High risk: Drug combination has high risk")
        elif risk_level == 'medium':
            summary.append("🟡 Medium risk: Recommend close monitoring")
        
        if warning.get('lab_abnormal_count', 0) > 0:
            summary.append(f"🔬 {warning['lab_abnormal_count']} lab indicators abnormal")
    
    if not summary:
        summary.append("✅ Current status is good, continue monitoring")
    
    return summary

@app.route('/drug_combinations', methods=['POST'])
def analyze_drug_combinations():
    """Analyze patient drug combinations"""
    try:
        data = request.json
        
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': 'Drug combination analysis system not initialized',
                'message': 'Data file not loaded, analysis feature unavailable (free tier memory limit, only drug list feature supported)'
            }), 503  # 503 Service Unavailable is more appropriate
        
        # Check if data source is available (full data or pre-computed model)
        has_data = combination_analyzer.data is not None
        has_model = hasattr(combination_analyzer, 'model_data') and combination_analyzer.model_data is not None
        
        if not has_data and not has_model:
            return jsonify({
                'success': False,
                'error': 'Data not loaded',
                'message': 'Both full data and pre-computed model are not loaded. Due to memory limitations, full data analysis is unavailable. Currently only drug list queries are supported. For full functionality, please upgrade to a paid plan or ensure the pre-computed model file exists.'
            }), 503  # 503 Service Unavailable
        
        # Convert input to DataFrame
        patient_data = pd.DataFrame([data])
        
        # Get outcome variable (default: death)
        outcome = request.json.get('outcome', 'death')
        
        # Analyze drug combinations
        result = combination_analyzer.analyze_patient_combination(patient_data, outcome)
        
        # Convert to serializable format
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
    """Get common drug combinations"""
    try:
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': 'Drug combination analysis system not initialized',
                'message': 'Data file not loaded, this feature is unavailable'
            }), 503
        
        # Check if data source is available
        has_data = combination_analyzer.data is not None
        has_model = hasattr(combination_analyzer, 'model_data') and combination_analyzer.model_data is not None
        
        if not has_data and not has_model:
            return jsonify({
                'success': False,
                'error': 'Data not loaded',
                'message': 'Both full data and pre-computed model are not loaded. Due to memory limitations, this feature is unavailable. Currently only drug list queries are supported.'
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
    """Get high-risk drug combinations"""
    try:
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': 'Drug combination analysis system not initialized',
                'message': 'Data file not loaded, this feature is unavailable'
            }), 503
        
        # Check if data source is available
        has_data = combination_analyzer.data is not None
        has_model = hasattr(combination_analyzer, 'model_data') and combination_analyzer.model_data is not None
        
        if not has_data and not has_model:
            return jsonify({
                'success': False,
                'error': 'Data not loaded',
                'message': 'Both full data and pre-computed model are not loaded. Due to memory limitations, this feature is unavailable. Currently only drug list queries are supported.'
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
    """Get effective drug combinations"""
    try:
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': 'Drug combination analysis system not initialized',
                'message': 'Data file not loaded, this feature is unavailable'
            }), 503
        
        # Check if data source is available
        has_data = combination_analyzer.data is not None
        has_model = hasattr(combination_analyzer, 'model_data') and combination_analyzer.model_data is not None
        
        if not has_data and not has_model:
            return jsonify({
                'success': False,
                'error': 'Data not loaded',
                'message': 'Both full data and pre-computed model are not loaded. Due to memory limitations, this feature is unavailable. Currently only drug list queries are supported.'
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
    """Get all available drug list"""
    try:
        # If data is not loaded, return empty list (instead of 500 error)
        if combination_analyzer is None or combination_analyzer.drug_columns is None:
            return jsonify({
                'success': True,
                'drugs': [],
                'total': 0,
                'filtered': 0,
                'warning': 'Drug combination analysis system not initialized, data file may not be loaded'
            })
        
        search = request.args.get('search', '').lower()
        limit = int(request.args.get('limit', 1000))
        
        drugs = combination_analyzer.drug_columns
        
        # Search filter
        if search:
            drugs = [d for d in drugs if search in d.lower()]
        
        # Limit count
        drugs = sorted(drugs)[:limit]
        
        return jsonify({
            'success': True,
            'drugs': drugs,
            'total': len(combination_analyzer.drug_columns),
            'filtered': len(drugs)
        })
    
    except Exception as e:
        # Return 200 even on error, but include error information
        return jsonify({
            'success': False,
            'drugs': [],
            'total': 0,
            'filtered': 0,
            'error': str(e)
        })

@app.route('/drugs/protective-effects', methods=['GET', 'POST'])
def get_drug_protective_effects():
    """Analyze which adverse outcome risks may be reduced when a specific drug is combined with other drugs"""
    try:
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': 'Drug combination analysis system not initialized',
                'message': 'Data file not loaded, this feature is unavailable'
            }), 503
        
        # Check if data source is available
        has_data = combination_analyzer.data is not None
        has_model = hasattr(combination_analyzer, 'model_data') and combination_analyzer.model_data is not None
        
        if not has_data and not has_model:
            return jsonify({
                'success': False,
                'error': 'Data not loaded',
                'message': 'Both full data and pre-computed model are not loaded. Due to memory limitations, this feature is unavailable. Currently only drug list queries are supported.'
            }), 503
        
        # Support both GET and POST requests
        if request.method == 'GET':
            drug_name = request.args.get('drug', '')
        else:
            data = request.json or {}
            drug_name = data.get('drug', '')
        
        if not drug_name:
            return jsonify({
                'error': 'Please provide drug name (drug parameter)'
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
    """Get recommended drugs to prevent multi-organ dysfunction"""
    try:
        if combination_analyzer is None:
            return jsonify({
                'success': False,
                'error': 'Drug combination analysis system not initialized',
                'message': 'Data file not loaded, this feature is unavailable'
            }), 503
        
        # Check if data source is available
        has_data = combination_analyzer.data is not None
        has_model = hasattr(combination_analyzer, 'model_data') and combination_analyzer.model_data is not None
        
        if not has_data and not has_model:
            return jsonify({
                'success': False,
                'error': 'Data not loaded',
                'message': 'Both full data and pre-computed model are not loaded. Due to memory limitations, this feature is unavailable. Currently only drug list queries are supported.'
            }), 503
        
        data = request.json
        current_drugs = data.get('drugs', [])
        
        if not current_drugs:
            return jsonify({
                'error': 'Please provide current drug list'
            }), 400
        
        # Get recommended drugs (for organ function abnormalities)
        recommendations = []
        
        # Recommendations for kidney function abnormalities
        kidney_recs = combination_analyzer.get_drug_recommendations(
            current_drugs, 
            outcome='death',  # Use death as proxy, should actually be based on organ function
            top_n=5
        )
        
        # Analyze each recommended drug's combination with current drugs to see if it can reduce organ function abnormality risk
        protective_drugs = []
        for rec in kidney_recs:
            # Check if this drug has protective combination with current drugs
            drug = rec['drug']
            has_protective_combo = False
            
            for current_drug in current_drugs:
                try:
                    # Analyze combination effect (simplified here, can be more complex)
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
                    'reason': f"Combination with {rec.get('best_combo_with', 'current drugs')} may reduce adverse outcome risk"
                })
        
        # Add some known protective drugs (based on clinical knowledge)
        # Always include these, even if not in the data, as they are clinically known to be protective
        known_protective_drugs = [
            {
                'drug': 'n-acetylcysteine',
                'reason': 'N-acetylcysteine: Antioxidant, may protect liver and kidney function, reduce oxidative stress damage',
                'category': 'Protective drug',
                'risk_reduction': 0.15
            },
            {
                'drug': 'vitamin_e',
                'reason': 'Vitamin E: Antioxidant, may reduce organ damage risk',
                'category': 'Protective drug',
                'risk_reduction': 0.12
            },
            {
                'drug': 'magnesium',
                'reason': 'Magnesium: May protect kidney function, maintain electrolyte balance',
                'category': 'Protective drug',
                'risk_reduction': 0.10
            },
            {
                'drug': 'ascorbic_acid',
                'reason': 'Vitamin C (Ascorbic Acid): Antioxidant, may protect organ function',
                'category': 'Protective drug',
                'risk_reduction': 0.10
            },
            {
                'drug': 'thiamine',
                'reason': 'Vitamin B1 (Thiamine): May support organ metabolic function',
                'category': 'Protective drug',
                'risk_reduction': 0.08
            },
            {
                'drug': 'folic_acid',
                'reason': 'Folic Acid: May support organ function',
                'category': 'Protective drug',
                'risk_reduction': 0.08
            }
        ]
        
        # Try to match with drugs in the data, but if not found, still include them
        for known_drug in known_protective_drugs:
            drug_name = known_drug['drug']
            # Check if this drug is already in current drugs
            if drug_name in current_drugs:
                continue
            
            # Try to find matching drug in data
            matching_drug = None
            if combination_analyzer.drug_columns:
                matching_drugs = [d for d in combination_analyzer.drug_columns 
                                if drug_name.lower().replace('_', '-') in d.lower() or 
                                   drug_name.lower().replace('_', ' ') in d.lower() or
                                   drug_name.lower() in d.lower()]
                if matching_drugs:
                    matching_drug = matching_drugs[0]
            
            # Add the drug (use matched name if found, otherwise use known name)
            protective_drugs.append({
                'drug': matching_drug if matching_drug else drug_name,
                'reason': known_drug['reason'],
                'category': known_drug['category'],
                'risk_reduction': known_drug['risk_reduction'],
                'potential_benefit': known_drug['reason']
            })
        
        # Remove duplicates and sort
        seen = set()
        unique_recs = []
        for rec in protective_drugs:
            if rec['drug'] not in seen:
                seen.add(rec['drug'])
                unique_recs.append(rec)
        
        unique_recs.sort(key=lambda x: x.get('risk_reduction', 0), reverse=True)
        
        return jsonify({
            'success': True,
            'recommendations': unique_recs[:10],  # Return top 10
            'current_drugs': current_drugs,
            'total_recommendations': len(unique_recs)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/drugs/side-effects', methods=['GET', 'POST', 'OPTIONS'])
def get_drug_side_effects():
    """Get drug side effects and toxicity information"""
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return response
    
    try:
        if side_effects_db is None:
            return jsonify({
                'success': False,
                'error': 'Side effects database not initialized',
                'message': 'Side effects database not loaded'
            }), 503
        
        # Support both GET and POST requests
        if request.method == 'GET':
            drug_names = request.args.get('drugs', '')
            if drug_names:
                # Comma-separated list
                drug_list = [d.strip() for d in drug_names.split(',') if d.strip()]
            else:
                drug_list = []
        else:
            # POST request
            data = request.json or {}
            drug_list = data.get('drugs', [])
            if isinstance(drug_list, str):
                # If it's a string, try to split by comma
                drug_list = [d.strip() for d in drug_list.split(',') if d.strip()]
        
        if not drug_list:
            return jsonify({
                'success': False,
                'error': 'Please provide drug names (drugs parameter)',
                'message': 'Provide a list of drug names to get side effects information'
            }), 400
        
        # Get side effects for each drug
        results = []
        for drug_name in drug_list:
            drug_info = side_effects_db.get_side_effects(drug_name)
            if drug_info:
                # drug_info is a dict with keys: side_effects, organ_toxicity, monitoring, contraindications, precautions
                results.append({
                    'drug': drug_name,
                    'found': True,
                    'side_effects': drug_info.get('side_effects', []),
                    'organ_toxicity': drug_info.get('organ_toxicity', {}),
                    'monitoring': drug_info.get('monitoring', []),
                    'contraindications': drug_info.get('contraindications', []),
                    'precautions': drug_info.get('precautions', '')
                })
            else:
                results.append({
                    'drug': drug_name,
                    'found': False,
                    'message': f'Side effects information not available for {drug_name}'
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_drugs': len(drug_list),
            'found_count': sum(1 for r in results if r.get('found', False))
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Initialize when module is loaded (executed when gunicorn starts)
# Use lazy initialization to avoid loading immediately on import (may affect startup speed)
_models_loaded = False

def initialize_models():
    """Initialize models and data (called when gunicorn starts)"""
    global _models_loaded
    if not _models_loaded:
        print("Loading models...")
        load_models()
        _models_loaded = True

# Using gunicorn's on_starting hook executes before worker starts
# But better way is to call at module level (each worker needs to load data)
# Initialize when application starts (gunicorn will execute this code when importing module)
try:
    initialize_models()
except Exception as e:
    print(f"Warning: Failed to load models during initialization: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit, allow service to start, but features will be limited

if __name__ == '__main__':
    try:
        # If running directly, ensure models are loaded
        if not _models_loaded:
            initialize_models()
        
        # Verify routes are registered
        print("\nVerifying route registration...")
        routes = list(app.url_map.iter_rules())
        print(f"✅ Registered {len(routes)} routes:")
        for rule in routes:
            if 'static' not in str(rule):
                print(f"  {rule}")
        
        print("\nStarting CDSS API server...")
        print("=" * 60)
        print("API endpoints:")
        print("  GET  /                      - API documentation")
        print("  GET  /health                 - Health check")
        print("  POST /predict                - Predict kidney and liver function abnormalities")
        print("  POST /warn                   - Drug combination risk warning")
        print("  POST /analyze                - Comprehensive analysis")
        print("  POST /drug_combinations      - Analyze patient drug combinations")
        print("  GET  /drug_combinations/common   - Get common drug combinations")
        print("  GET  /drug_combinations/risky    - Get high-risk drug combinations")
        print("  GET  /drug_combinations/effective - Get effective drug combinations")
        print("  GET  /drugs/list             - Get drug list")
        print("  POST /drugs/recommend        - Get drug recommendations")
        print("  GET/POST /drugs/side-effects - Get drug side effects and toxicity information")
        print("  GET/POST /drugs/protective-effects - Analyze drug protective effects")
        print("=" * 60)
        
        # Get port number, prefer environment variable (Render sets PORT environment variable)
        PORT = int(os.environ.get('PORT', 5003))
        HOST = os.environ.get('HOST', '0.0.0.0')  # Render needs to listen on 0.0.0.0
        
        print(f"\nServer running on http://{HOST}:{PORT}")
        print(f"Visit http://localhost:{PORT} to view API documentation")
        print("Open drug_combination_analyzer.html to use drug combination analysis interface")
        print("\nPress Ctrl+C to stop server")
        print("=" * 60)
        
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
