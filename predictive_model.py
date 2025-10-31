import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
from colorama import Fore, Style, init

# Initialize colored output
init(autoreset=True)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. SYNTHETIC DATA GENERATION ---
def generate_synthetic_data(num_samples=50000):
    """
    Generates a large, plausible-looking dataset of mock patient records
    with all the features used in the frontend.
    """
    print(Fore.CYAN + "Generating advanced synthetic data pool...")
    np.random.seed(42)
    data_list = []
    
    # Helper for random numbers
    randn = lambda: (np.random.rand() + np.random.rand() + np.random.rand() - 1.5) * 2

    for _ in range(num_samples):
        age = np.random.randint(18, 85)
        height_m = 1.5 + np.random.rand() * 0.5
        weight_kg = (50 + randn() * 10) + (age / 3)
        bmi = weight_kg / (height_m ** 2)

        history_diabetes = 1 if np.random.rand() < 0.2 else 0
        history_cancer = 1 if np.random.rand() < 0.1 else 0
        
        smoker = 1 if np.random.rand() < (0.4 - (age / 300)) else 0
        
        # --- Simplified Heart Features ---
        sex = 1 if np.random.rand() < 0.7 else 0 # More male
        cp = np.random.randint(0, 4) # Chest Pain
        chol = 150 + randn() * 50 + (age - 30) * 0.5
        trestbps = 100 + randn() * 20 + (chol - 200) * 0.1 + (age-30) * 0.3
        fbs = 1 if (chol > 200 and np.random.rand() < 0.3) else 0
        thalach = 180 - (age * 0.8)
        exang = 1 if (cp > 0 and np.random.rand() < 0.4) else 0
        hasEKG = 1 if (exang == 1 and np.random.rand() < 0.5) else 0 # EKG anomaly
        
        # --- Diabetes Features ---
        glucose = 80 + randn() * 20 + (bmi - 22) * 1.5 + (age - 21) * 0.5
        systolic_bp = 100 + randn() * 15 + (bmi - 22) * 1.0 + (age - 21) * 0.3
        
        # --- Cancer Features ---
        hasImage = 1 if (age > 60 and smoker == 1 and np.random.rand() < 0.4) else 0 # Image anomaly

        # --- Define Disease Outcomes ---
        DISEASE_DIABETES = (glucose > 126) or \
                           (bmi > 35 and glucose > 110) or \
                           (age > 50 and glucose > 120) or \
                           (history_diabetes > 0 and glucose > 115)

        DISEASE_HEART = (cp > 0) or \
                        (thalach < 140 and age > 50) or \
                        (exang == 1) or \
                        (chol > 240 and trestbps > 140) or \
                        (hasEKG == 1)

        DISEASE_CANCER = (age > 60 and smoker > 0) or \
                         (age > 50 and history_cancer > 0) or \
                         (bmi > 35 and age > 55) or \
                         (hasImage == 1)
        
        data_list.append({
            # Diabetes features
            'Age_D': int(age), 'BMI_D': round(bmi, 1), 'Glucose': int(glucose), 
            'SystolicBP': int(systolic_bp), 'FamilyHistory_D': history_diabetes,
            
            # Heart features
            'age_H': int(age), 'sex': sex, 'cp': cp, 'trestbps': int(trestbps), 
            'chol': int(chol), 'fbs': fbs, 'thalach': int(thalach), 'exang': exang, 'hasEKG': hasEKG,
            
            # Cancer features
            'Age_C': int(age), 'BMI_C': round(bmi, 1), 'Smoker': smoker, 
            'FamilyHistory_C': history_cancer, 'hasImage': hasImage,
            
            # Outcomes
            'DISEASE_DIABETES': 1 if DISEASE_DIABETES else 0,
            'DISEASE_HEART': 1 if DISEASE_HEART else 0,
            'DISEASE_CANCER': 1 if DISEASE_CANCER else 0
        })
    
    print(Fore.GREEN + f" ✓ Generated {len(data_list)} synthetic records.")
    return pd.DataFrame(data_list)

# --- 2. MODEL TRAINING & PREDICTION ---

# Define the features for each model
FEATURES_DIABETES = ['Age_D', 'BMI_D', 'Glucose', 'SystolicBP', 'FamilyHistory_D']
FEATURES_HEART = ['age_H', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'hasEKG']
FEATURES_CANCER = ['Age_C', 'BMI_C', 'Smoker', 'FamilyHistory_C', 'hasImage']

# This dictionary will hold our trained models and scalers
trained_tools = {}
# Global list to store patient history
patient_history = []

# Generate the full dataset once
full_dataset = generate_synthetic_data()

# --- Precautions Dictionary ---
PRECAUTIONS = {
    'diabetes': {
        'Low': "Maintain a healthy lifestyle. Continue regular check-ups and be mindful of your diet.",
        'Medium': "Proactive lifestyle changes are key. Monitor your diet (reduce sugar/carbs) and increase physical activity.",
        'High': "Consult your doctor immediately. Focus on strict glucose monitoring, diet, and exercise."
    },
    'heart': {
        'Low': "Excellent. Continue a heart-healthy diet, stay active, and manage stress.",
        'Medium': "Focus on heart-healthy habits. Reduce sodium and saturated fats, manage stress, and aim for 150 mins of exercise/week.",
        'High': "Immediate medical consultation is advised. Your doctor may suggest EKG, stress tests, and medication."
    },
    'cancer': {
        'Low': "Continue a healthy lifestyle, avoid smoking, and follow standard screening guidelines for your age.",
        'Medium': "Discuss your risk factors (smoking, family history) with your doctor. Adopt a healthy lifestyle and schedule regular check-ups.",
        'High': "Speak to your doctor about a screening plan (e.g., mammogram, colonoscopy, lung CT scan) immediately."
    }
}

# --- NEW: Next Steps Dictionary ---
NEXT_STEPS = {
    'diabetes': {
        'Low': ["- Continue annual wellness checks.", "- Reinforce healthy diet and exercise."],
        'Medium': ["- Order fasting glucose test.", "- Recommend lifestyle changes.", "- Schedule 6-month follow-up."],
        'High': ["- Order HbA1c lab test.", "- Refer to endocrinologist.", "- Provide nutritional counseling."]
    },
    'heart': {
        'Low': ["- Continue annual wellness checks.", "- Reinforce heart-healthy lifestyle."],
        'Medium': ["- Order full lipid panel.", "- Recommend lifestyle changes (diet/exercise).", "- Schedule 6-month follow-up."],
        'High': ["- Order 12-lead EKG.", "- Refer to cardiologist.", "- Order full lipid panel & troponin test.", "- Schedule stress test."]
    },
    'cancer': {
        'Low': ["- Follow standard age-appropriate screening guidelines (e.g., colonoscopy, mammogram)."],
        'Medium': ["- Develop a personalized screening plan.", "- Strongly advise smoking cessation.", "- Discuss genetic counseling if family history is strong."],
        'High': ["- Refer to oncologist/specialist.", "- Order relevant imaging (e.g., Low-Dose CT for smoker, Mammogram).", "- Order relevant blood work (e.g., CEA, PSA)."]
    }
}


# --- Risk Level Interpreter ---
def interpret_risk_level(prob):
    if prob < 0.33:
        return ("Low", Fore.GREEN)
    elif prob < 0.66:
        return ("Medium", Fore.YELLOW)
    else:
        return ("High", Fore.RED)

def train_all_models():
    """
    Trains all three models on the full dataset.
    """
    print(Fore.CYAN + f"\n--- Training all models on {len(full_dataset)} records ---")
    
    # --- Train Diabetes Model ---
    try:
        y_diabetes = full_dataset['DISEASE_DIABETES']
        if len(y_diabetes.unique()) > 1:
            X_diabetes = full_dataset[FEATURES_DIABETES]
            scaler_d = StandardScaler().fit(X_diabetes)
            X_scaled_d = scaler_d.transform(X_diabetes)
            model_d = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10).fit(X_scaled_d, y_diabetes)
            trained_tools['diabetes'] = {'model': model_d, 'scaler': scaler_d, 'features': FEATURES_DIABETES}
            print(Fore.GREEN + " ✓ Diabetes model trained.")
        else:
            print(Fore.YELLOW + " ⚠ Skipped Diabetes (only one class present).")
    except Exception as e:
        print(Fore.RED + f" ✗ Error training Diabetes model: {e}")

    # --- Train Heart Model ---
    try:
        y_heart = full_dataset['DISEASE_HEART']
        if len(y_heart.unique()) > 1:
            X_heart = full_dataset[FEATURES_HEART]
            scaler_h = StandardScaler().fit(X_heart)
            X_scaled_h = scaler_h.transform(X_heart)
            model_h = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10).fit(X_scaled_h, y_heart)
            trained_tools['heart'] = {'model': model_h, 'scaler': scaler_h, 'features': FEATURES_HEART}
            print(Fore.GREEN + " ✓ Heart model trained.")
        else:
            print(Fore.YELLOW + " ⚠ Skipped Heart (only one class present).")
    except Exception as e:
        print(Fore.RED + f" ✗ Error training Heart model: {e}")
        
    # --- Train Cancer Model ---
    try:
        y_cancer = full_dataset['DISEASE_CANCER']
        if len(y_cancer.unique()) > 1:
            X_cancer = full_dataset[FEATURES_CANCER]
            scaler_c = StandardScaler().fit(X_cancer)
            X_scaled_c = scaler_c.transform(X_cancer)
            model_c = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10).fit(X_scaled_c, y_cancer)
            trained_tools['cancer'] = {'model': model_c, 'scaler': scaler_c, 'features': FEATURES_CANCER}
            print(Fore.GREEN + " ✓ Cancer model trained.")
        else:
            print(Fore.YELLOW + " ⚠ Skipped Cancer (only one class present).")
    except Exception as e:
        print(Fore.RED + f" ✗ Error training Cancer model: {e}")
        
    return True

# --- UPDATED: Risk Assessment Function ---
def get_risk_assessment(patient_data, model_key):
    """
    Uses the correct trained model to predict risk for one patient
    and (in a real app) would integrate multimodal analysis.
    Here, the multimodal data is already part of the feature set.
    """
    if model_key not in trained_tools:
        print(Fore.RED + f"Model '{model_key}' is not trained.")
        return
        
    tools = trained_tools[model_key]
    model_features = tools['features']
    
    # Create a DataFrame for the patient, ensuring feature order
    try:
        patient_df = pd.DataFrame([patient_data])[model_features]
    except KeyError as e:
        print(Fore.RED + f"Error: Patient data is missing feature: {e}.")
        return

    # --- 1. Structured Data Analysis (Sklearn) ---
    scaled_features = tools['scaler'].transform(patient_df)
    prob = tools['model'].predict_proba(scaled_features)[0][1] # Probability of "1"
    
    # --- Get Precaution & Next Steps ---
    level, color = interpret_risk_level(prob)
    precaution = PRECAUTIONS[model_key][level]
    next_steps = NEXT_STEPS[model_key][level]

    # XAI Report (from structured data)
    importances = tools['model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(Style.BRIGHT + f"\n--- {model_key.title()} Risk Report ---")
    print(f"Risk Score: {prob*100:.0f}% ({level})")
    
    # Print Precaution
    print(color + f"Recommended Precaution: {precaution}")
    
    # --- NEW: Print Next Steps ---
    print(Style.BRIGHT + "Recommended Next Steps:")
    for step in next_steps:
        print(Fore.WHITE + f"  {step}")
    
    print("\nKey Risk Contributors (from Structured & Multimodal Data):")
    for i in range(len(model_features)):
        # Format multimodal features for clarity
        feature_name = model_features[indices[i]]
        if feature_name == 'hasEKG':
            feature_name = 'Wearable EKG (Multimodal)'
        elif feature_name == 'hasImage':
            feature_name = 'Image Scan (Multimodal)'
            
        print(f"  - {feature_name}: {importances[indices[i]]:.2f}")
    
    return prob

# --- 3. HELPER FUNCTIONS FOR USER INPUT ---

def get_diabetes_input():
    print(Style.BRIGHT + "\n--- Enter Diabetes Risk Data ---")
    try:
        name = input("Patient Name: ")
        age = int(input("Age: "))
        height_cm = float(input("Height (cm): "))
        weight_kg = float(input("Weight (kg): "))
        glucose = int(input("Glucose (mg/dL): "))
        systolic_bp = int(input("Systolic BP: "))
        family_history = 1 if input("Family History of Diabetes? (y/n): ").lower() == 'y' else 0
        
        bmi = weight_kg / ((height_cm / 100) ** 2)
        
        return {
            "name": name,
            "age": age,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "Age_D": age, "BMI_D": round(bmi, 1), "Glucose": glucose,
            "SystolicBP": systolic_bp, "FamilyHistory_D": family_history
        }
    except ValueError:
        print(Fore.RED + "Invalid input. Please enter numbers only.")
        return get_diabetes_input()

def get_heart_input():
    print(Style.BRIGHT + "\n--- Enter Heart Risk Data ---")
    try:
        name = input("Patient Name: ")
        patient_data = {
            "name": name,
            "age_H": int(input("Age: ")),
            "sex": 1 if input("Sex (m/f): ").lower() == 'm' else 0,
            "cp": int(input("Chest Pain Type (0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic): ")),
            "trestbps": int(input("Resting BP (SBP): ")),
            "chol": int(input("Cholesterol (mg/dL): ")),
            "fbs": 1 if input("Fasting Blood Sugar > 120? (y/n): ").lower() == 'y' else 0,
            "thalach": int(input("Max Heart Rate Achieved: ")),
            "exang": 1 if input("Exercise-induced Angina? (y/n): ").lower() == 'y' else 0,
            "hasEKG": 1 if input("Simulate EKG anomaly? (y/n): ").lower() == 'y' else 0, # Multimodal
        }
        patient_data["age"] = patient_data["age_H"]
        return patient_data
    except ValueError:
        print(Fore.RED + "Invalid input. Please enter numbers only.")
        return get_heart_input()

def get_cancer_input():
    print(Style.BRIGHT + "\n--- Enter Cancer Risk Data ---")
    try:
        name = input("Patient Name: ")
        age = int(input("Age: "))
        height_cm = float(input("Height (cm): "))
        weight_kg = float(input("Weight (kg): "))
        smoker = 1 if input("Current Smoker? (y/n): ").lower() == 'y' else 0
        family_history = 1 if input("Family History of Cancer? (y/n): ").lower() == 'y' else 0
        hasImage = 1 if input("Simulate Image anomaly? (y/n): ").lower() == 'y' else 0 # Multimodal

        bmi = weight_kg / ((height_cm / 100) ** 2)

        return {
            "name": name,
            "age": age,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "Age_C": age, "BMI_C": round(bmi, 1), "Smoker": smoker,
            "FamilyHistory_C": family_history,
            "hasImage": hasImage
        }
    except ValueError:
        print(Fore.RED + "Invalid input. Please enter. Please enter numbers only.")
        return get_cancer_input()

# --- Function to display patient history ---
def show_patient_history():
    print(Style.BRIGHT + Fore.CYAN + "\n--- Patient History Log ---")
    
    if not patient_history:
        print(Fore.YELLOW + "No patients have been assessed yet.")
        return

    for i, patient in enumerate(patient_history):
        print(Style.BRIGHT + f"\n--- Patient #{i + 1} ---")
        # Make a copy to avoid modifying the original
        details = patient.copy()
        
        # Print key details first
        print(f"  Name:           {Fore.WHITE}{details.pop('name', 'N/A')}")
        print(f"  Age:            {Fore.WHITE}{details.pop('age', 'N/A')}")
        print(f"  Assessment:     {Fore.WHITE}{details.pop('assessment_type', 'N/A')}")
        
        risk_score = details.pop('risk_score', None)
        if risk_score is not None:
            level, _ = interpret_risk_level(risk_score)
            print(f"  Risk Score:     {Fore.WHITE}{risk_score*100:.0f}% ({level})")
        
        print(Style.BRIGHT + "  Details:")
        # Print remaining details
        for key, value in details.items():
            # Format booleans nicely
            if isinstance(value, int) and key in ['FamilyHistory_D', 'fbs', 'exang', 'Smoker', 'FamilyHistory_C', 'sex', 'hasEKG', 'hasImage']:
                value_str = "Yes" if value == 1 else "No"
                if key == 'sex':
                    value_str = "Male" if value == 1 else "Female"
            else:
                value_str = str(value)
            
            # Clean up keys for display
            key_str = key.replace('_D', '').replace('_H', '').replace('_C', '')
            print(f"    - {key_str}: {Fore.WHITE}{value_str}")

# --- 4. MAIN EXECUTION (INTERACTIVE LOOP) ---

if __name__ == "__main__":
    
    # Train all models on startup
    train_all_models()
    
    while True:
        print(Style.BRIGHT + Fore.CYAN + "\n--- Main Menu ---")
        print("1. Assess Diabetes Risk")
        print("2. Assess Heart Disease Risk")
        print("3. Assess Cancer Risk")
        print("4. Show Patient History")
        print("5. Exit")
        
        choice = input("Please select an option (1-5): ")
        
        if choice == '1':
            if 'diabetes' in trained_tools:
                patient_inputs = get_diabetes_input()
                if patient_inputs is None: continue 
                
                patient_for_history = patient_inputs.copy()
                patient_for_history['assessment_type'] = 'Diabetes'
                
                # Fill in blanks for other models
                patient_inputs.update({
                    "age_H": 0, "sex": 0, "cp": 0, "trestbps": 0, "chol": 0, "fbs": 0, "thalach": 0, "exang": 0, "hasEKG": 0,
                    "Age_C": 0, "BMI_C": 0, "Smoker": 0, "FamilyHistory_C": 0, "hasImage": 0
                })
                risk_score = get_risk_assessment(patient_inputs, 'diabetes')
                
                if risk_score is not None:
                    patient_for_history['risk_score'] = risk_score
                patient_history.append(patient_for_history)
            else:
                print(Fore.RED + "Diabetes model is not available.")
                
        elif choice == '2':
            if 'heart' in trained_tools:
                patient_inputs = get_heart_input()
                if patient_inputs is None: continue 

                patient_for_history = patient_inputs.copy()
                patient_for_history['assessment_type'] = 'Heart Disease'

                # Fill in blanks for other models
                patient_inputs.update({
                    "Age_D": 0, "BMI_D": 0, "Glucose": 0, "SystolicBP": 0, "FamilyHistory_D": 0,
                    "Age_C": 0, "BMI_C": 0, "Smoker": 0, "FamilyHistory_C": 0, "hasImage": 0
                })
                risk_score = get_risk_assessment(patient_inputs, 'heart')
                
                if risk_score is not None:
                    patient_for_history['risk_score'] = risk_score
                patient_history.append(patient_for_history)
            else:
                print(Fore.RED + "Heart Disease model is not available.")
                
        elif choice == '3':
            if 'cancer' in trained_tools:
                patient_inputs = get_cancer_input()
                if patient_inputs is None: continue 

                patient_for_history = patient_for_history.copy()
                patient_for_history['assessment_type'] = 'Cancer'
                
                # Fill in blanks for other models
                patient_inputs.update({
                    "Age_D": 0, "BMI_D": 0, "Glucose": 0, "SystolicBP": 0, "FamilyHistory_D": 0,
                    "age_H": 0, "sex": 0, "cp": 0, "trestbps": 0, "chol": 0, "fbs": 0, "thalach": 0, "exang": 0, "hasEKG": 0
                })
                risk_score = get_risk_assessment(patient_inputs, 'cancer')
                
                if risk_score is not None:
                    patient_for_history['risk_score'] = risk_score
                patient_history.append(patient_for_history)
            else:
                print(Fore.RED + "Cancer model is not available.")
        
        elif choice == '4':
            show_patient_history()
            
        elif choice == '5': 
            print(Fore.CYAN + "Exiting. Stay healthy!")
            break
            
        else:
            print(Fore.RED + "Invalid choice. Please enter a number from 1 to 5.")

