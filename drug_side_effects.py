"""
Drug Side Effects Database
Provides detailed side effects and toxicity information for drugs
"""

class DrugSideEffects:
    def __init__(self):
        """Initialize drug side effects database"""
        # Drug side effects database
        # Format: drug_name: {side_effects: [], organ_toxicity: {}, monitoring: [], contraindications: []}
        self.side_effects_db = {
            'vancomycin': {
                'side_effects': [
                    'Nephrotoxicity (kidney damage)',
                    'Ototoxicity (hearing loss)',
                    'Red man syndrome (flushing, rash)',
                    'Thrombophlebitis (vein inflammation)',
                    'Neutropenia (low white blood cells)'
                ],
                'organ_toxicity': {
                    'kidney': 'high',
                    'ear': 'high',
                    'blood': 'medium'
                },
                'monitoring': [
                    'Monitor serum creatinine and BUN',
                    'Monitor hearing function',
                    'Monitor vancomycin trough levels (target: 10-20 mg/L)',
                    'Monitor complete blood count'
                ],
                'contraindications': [
                    'Hypersensitivity to vancomycin',
                    'Severe renal impairment (adjust dose)'
                ],
                'precautions': 'Use with caution in patients with renal impairment, hearing loss, or receiving other nephrotoxic drugs'
            },
            'gentamicin': {
                'side_effects': [
                    'Nephrotoxicity (kidney damage)',
                    'Ototoxicity (hearing and balance problems)',
                    'Neuromuscular blockade',
                    'Hypersensitivity reactions'
                ],
                'organ_toxicity': {
                    'kidney': 'high',
                    'ear': 'high',
                    'neuromuscular': 'medium'
                },
                'monitoring': [
                    'Monitor serum creatinine and BUN',
                    'Monitor hearing and vestibular function',
                    'Monitor gentamicin peak and trough levels',
                    'Monitor urine output'
                ],
                'contraindications': [
                    'Hypersensitivity to aminoglycosides',
                    'Myasthenia gravis (relative contraindication)'
                ],
                'precautions': 'Use with caution in renal impairment, elderly patients, and when combined with other nephrotoxic drugs'
            },
            'furosemide': {
                'side_effects': [
                    'Dehydration and electrolyte imbalance',
                    'Hypokalemia (low potassium)',
                    'Hypotension (low blood pressure)',
                    'Ototoxicity (with high doses)',
                    'Hyperuricemia (high uric acid)',
                    'Renal function deterioration (in some cases)'
                ],
                'organ_toxicity': {
                    'kidney': 'medium',
                    'ear': 'low',
                    'electrolyte': 'high'
                },
                'monitoring': [
                    'Monitor electrolytes (K+, Na+, Cl-, Mg2+)',
                    'Monitor blood pressure',
                    'Monitor renal function (creatinine, BUN)',
                    'Monitor fluid balance and weight'
                ],
                'contraindications': [
                    'Anuria (no urine output)',
                    'Hypersensitivity to furosemide or sulfonamides',
                    'Severe electrolyte depletion'
                ],
                'precautions': 'Monitor electrolytes closely, especially potassium. Use with caution in hepatic impairment and diabetes'
            },
            'acetaminophen': {
                'side_effects': [
                    'Hepatotoxicity (liver damage) - especially with overdose',
                    'Allergic reactions (rare)',
                    'Skin reactions (rare)'
                ],
                'organ_toxicity': {
                    'liver': 'high',
                    'skin': 'low'
                },
                'monitoring': [
                    'Monitor liver function tests (ALT, AST, bilirubin)',
                    'Monitor for signs of overdose (nausea, vomiting, abdominal pain)',
                    'Check acetaminophen levels if overdose suspected'
                ],
                'contraindications': [
                    'Severe hepatic impairment',
                    'Hypersensitivity to acetaminophen'
                ],
                'precautions': 'Do not exceed 4g/day in adults. Use with caution in chronic alcohol use, malnutrition, or hepatic impairment'
            },
            'aspirin': {
                'side_effects': [
                    'Gastrointestinal bleeding',
                    'Peptic ulcer disease',
                    'Renal impairment (with high doses)',
                    'Reye syndrome (in children with viral infections)',
                    'Hypersensitivity reactions',
                    'Increased bleeding risk'
                ],
                'organ_toxicity': {
                    'gastrointestinal': 'high',
                    'kidney': 'medium',
                    'blood': 'high'
                },
                'monitoring': [
                    'Monitor for signs of GI bleeding',
                    'Monitor renal function with chronic use',
                    'Monitor bleeding time if on anticoagulants'
                ],
                'contraindications': [
                    'Active peptic ulcer',
                    'Severe hepatic impairment',
                    'Bleeding disorders',
                    'Children with viral infections (Reye syndrome risk)'
                ],
                'precautions': 'Use with caution in elderly, patients with GI disorders, or on anticoagulants'
            },
            'prednisone': {
                'side_effects': [
                    'Hyperglycemia (high blood sugar)',
                    'Hypertension (high blood pressure)',
                    'Osteoporosis and bone fractures',
                    'Increased infection risk',
                    'Cushing syndrome (with long-term use)',
                    'Adrenal suppression',
                    'Gastrointestinal ulcers',
                    'Cataracts and glaucoma',
                    'Mood changes and insomnia'
                ],
                'organ_toxicity': {
                    'endocrine': 'high',
                    'bone': 'high',
                    'eye': 'medium',
                    'gastrointestinal': 'medium'
                },
                'monitoring': [
                    'Monitor blood glucose',
                    'Monitor blood pressure',
                    'Monitor bone density (with long-term use)',
                    'Monitor for signs of infection',
                    'Monitor eye exams (cataracts, glaucoma)'
                ],
                'contraindications': [
                    'Systemic fungal infections',
                    'Hypersensitivity to prednisone'
                ],
                'precautions': 'Taper dose gradually to avoid adrenal insufficiency. Monitor for infections, hyperglycemia, and bone health'
            },
            'piperacillin': {
                'side_effects': [
                    'Hypersensitivity reactions',
                    'Diarrhea and Clostridium difficile infection',
                    'Hematologic abnormalities (neutropenia, thrombocytopenia)',
                    'Seizures (with high doses in renal impairment)',
                    'Electrolyte imbalances'
                ],
                'organ_toxicity': {
                    'blood': 'medium',
                    'gastrointestinal': 'medium',
                    'nervous_system': 'low'
                },
                'monitoring': [
                    'Monitor for allergic reactions',
                    'Monitor complete blood count',
                    'Monitor renal function and adjust dose',
                    'Monitor for diarrhea and C. difficile'
                ],
                'contraindications': [
                    'Hypersensitivity to penicillins or beta-lactams'
                ],
                'precautions': 'Use with caution in renal impairment. Monitor for superinfections and C. difficile'
            },
            'ceftriaxone': {
                'side_effects': [
                    'Hypersensitivity reactions',
                    'Diarrhea and C. difficile infection',
                    'Gallbladder complications (biliary sludge)',
                    'Hematologic abnormalities',
                    'Renal impairment (rare)'
                ],
                'organ_toxicity': {
                    'gastrointestinal': 'medium',
                    'gallbladder': 'low',
                    'blood': 'low'
                },
                'monitoring': [
                    'Monitor for allergic reactions',
                    'Monitor for diarrhea',
                    'Monitor complete blood count',
                    'Monitor renal function'
                ],
                'contraindications': [
                    'Hypersensitivity to cephalosporins',
                    'Severe hypersensitivity to penicillins (cross-reactivity)'
                ],
                'precautions': 'Use with caution in patients with penicillin allergy. Monitor for C. difficile infection'
            },
            'warfarin': {
                'side_effects': [
                    'Bleeding (major risk)',
                    'Skin necrosis (rare)',
                    'Purple toe syndrome (rare)',
                    'Hair loss',
                    'Teratogenicity (birth defects)'
                ],
                'organ_toxicity': {
                    'blood': 'high',
                    'skin': 'low'
                },
                'monitoring': [
                    'Monitor INR regularly (target: 2.0-3.0 for most indications)',
                    'Monitor for signs of bleeding',
                    'Monitor liver function',
                    'Monitor for drug interactions'
                ],
                'contraindications': [
                    'Active bleeding',
                    'Severe hepatic impairment',
                    'Pregnancy (teratogenic)',
                    'Recent surgery or trauma'
                ],
                'precautions': 'Many drug interactions. Monitor INR closely. Avoid sudden diet changes (vitamin K)'
            },
            'heparin': {
                'side_effects': [
                    'Bleeding (major risk)',
                    'Heparin-induced thrombocytopenia (HIT)',
                    'Osteoporosis (with long-term use)',
                    'Hypersensitivity reactions',
                    'Alopecia (hair loss)'
                ],
                'organ_toxicity': {
                    'blood': 'high',
                    'bone': 'low'
                },
                'monitoring': [
                    'Monitor aPTT (activated partial thromboplastin time)',
                    'Monitor platelet count (for HIT)',
                    'Monitor for signs of bleeding',
                    'Monitor for HIT antibodies if platelets drop'
                ],
                'contraindications': [
                    'Active bleeding',
                    'Heparin-induced thrombocytopenia',
                    'Severe thrombocytopenia',
                    'Uncontrolled hypertension'
                ],
                'precautions': 'Monitor platelets daily for first 2 weeks. Watch for HIT (platelet drop >50%)'
            },
            'insulin': {
                'side_effects': [
                    'Hypoglycemia (low blood sugar) - most common',
                    'Weight gain',
                    'Lipodystrophy at injection sites',
                    'Allergic reactions (rare)',
                    'Hypokalemia (low potassium)'
                ],
                'organ_toxicity': {
                    'metabolic': 'high',
                    'skin': 'low'
                },
                'monitoring': [
                    'Monitor blood glucose frequently',
                    'Monitor for signs of hypoglycemia',
                    'Monitor HbA1c',
                    'Monitor injection sites'
                ],
                'contraindications': [
                    'Hypoglycemia',
                    'Hypersensitivity to insulin'
                ],
                'precautions': 'Risk of severe hypoglycemia. Educate patients on recognition and treatment. Monitor glucose closely'
            },
            'atorvastatin': {
                'side_effects': [
                    'Myopathy and rhabdomyolysis (muscle damage)',
                    'Hepatotoxicity (liver damage)',
                    'Increased blood glucose',
                    'Memory problems (rare)',
                    'Muscle pain and weakness'
                ],
                'organ_toxicity': {
                    'muscle': 'medium',
                    'liver': 'medium',
                    'metabolic': 'low'
                },
                'monitoring': [
                    'Monitor liver function tests (ALT, AST)',
                    'Monitor for muscle pain and weakness',
                    'Monitor creatine kinase (CK) if muscle symptoms',
                    'Monitor blood glucose'
                ],
                'contraindications': [
                    'Active liver disease',
                    'Pregnancy',
                    'Hypersensitivity to statins'
                ],
                'precautions': 'Monitor for myopathy, especially with high doses or drug interactions. Avoid in pregnancy'
            },
            'metformin': {
                'side_effects': [
                    'Lactic acidosis (rare but serious)',
                    'Gastrointestinal upset (nausea, diarrhea)',
                    'Vitamin B12 deficiency (with long-term use)',
                    'Hypoglycemia (when combined with other diabetes drugs)'
                ],
                'organ_toxicity': {
                    'metabolic': 'high',
                    'gastrointestinal': 'medium',
                    'hematologic': 'low'
                },
                'monitoring': [
                    'Monitor renal function (contraindicated if eGFR <30)',
                    'Monitor for signs of lactic acidosis',
                    'Monitor vitamin B12 levels (with long-term use)',
                    'Monitor blood glucose'
                ],
                'contraindications': [
                    'Severe renal impairment (eGFR <30)',
                    'Metabolic acidosis',
                    'Severe hepatic impairment',
                    'Conditions predisposing to lactic acidosis'
                ],
                'precautions': 'Contraindicated in renal impairment. Hold before contrast studies. Monitor for lactic acidosis'
            }
        }
    
    def get_side_effects(self, drug_name):
        """
        Get side effects information for a drug
        Returns None if drug not found
        """
        # Normalize drug name
        normalized = self._normalize_drug_name(drug_name)
        
        # Direct match
        if normalized in self.side_effects_db:
            return self.side_effects_db[normalized]
        
        # Partial match
        for db_drug, info in self.side_effects_db.items():
            if normalized in db_drug or db_drug in normalized:
                return info
        
        return None
    
    def get_multiple_drugs_side_effects(self, drug_names):
        """Get side effects for multiple drugs"""
        results = {}
        for drug in drug_names:
            effects = self.get_side_effects(drug)
            if effects:
                results[drug] = effects
        return results
    
    def _normalize_drug_name(self, drug_name):
        """Normalize drug name for matching"""
        return drug_name.lower().replace(' ', '_').replace('-', '_').strip()
    
    def get_organ_toxicity_summary(self, drug_names):
        """Get summary of organ toxicity for multiple drugs"""
        summary = {
            'kidney': [],
            'liver': [],
            'blood': [],
            'gastrointestinal': [],
            'heart': [],
            'other': []
        }
        
        for drug in drug_names:
            effects = self.get_side_effects(drug)
            if effects and 'organ_toxicity' in effects:
                for organ, level in effects['organ_toxicity'].items():
                    if organ == 'kidney':
                        summary['kidney'].append({
                            'drug': drug,
                            'level': level
                        })
                    elif organ == 'liver':
                        summary['liver'].append({
                            'drug': drug,
                            'level': level
                        })
                    elif organ == 'blood':
                        summary['blood'].append({
                            'drug': drug,
                            'level': level
                        })
                    elif organ == 'gastrointestinal':
                        summary['gastrointestinal'].append({
                            'drug': drug,
                            'level': level
                        })
                    elif organ == 'heart' or organ == 'cardiovascular':
                        summary['heart'].append({
                            'drug': drug,
                            'level': level
                        })
                    else:
                        summary['other'].append({
                            'drug': drug,
                            'organ': organ,
                            'level': level
                        })
        
        return summary

