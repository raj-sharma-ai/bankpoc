#risk_model_raj.py
import pandas as pd
import numpy as np
import json
import hashlib
import pickle
from pathlib import Path

# =====================================================
# STEP 1: Generate ONLY Basic Synthetic Users
# =====================================================

def generate_basic_synthetic_users(num_users=50):
    """Generate ONLY basic synthetic user profiles (no risk, no features)"""
    print(f"Generating {num_users} basic synthetic users...")
    
    genders = ['M', 'F']
    occupations = ['Salaried', 'Self-employed', 'Business', 'Retired']
    city_tiers = [1, 2, 3]
    investment_types = ['None', 'Stocks', 'Mutual_Funds', 'Insurance', 'Real_Estate', 
                       'Stocks_Mutual_Funds', 'Stocks_Insurance']
    
    users = []
    
    for i in range(1, num_users + 1):
        # Base features ONLY
        user_id = f"USER_{i:04d}"
        age = np.random.randint(22, 65)
        gender = np.random.choice(genders)
        citytier = np.random.choice(city_tiers)
        
        # Financial features
        if citytier == 1:
            annualincome = np.random.randint(400000, 2500000)
        elif citytier == 2:
            annualincome = np.random.randint(300000, 1200000)
        else:
            annualincome = np.random.randint(200000, 800000)
        
        occupation = np.random.choice(occupations)
        creditscore = np.random.randint(550, 850)
        
        # Spending based on income
        avgmonthlyspend = int(annualincome / 12 * np.random.uniform(0.5, 0.85))
        
        # Savings rate
        savingsrate = round(np.random.uniform(0.05, 0.40), 2)
        
        # Investment
        investmentamountlastyear = int(annualincome * np.random.uniform(0.0, 0.15))
        pastinvestments = np.random.choice(investment_types)
        
        # Create ONLY base user (NO DERIVED FEATURES)
        user_base = {
            'customer_id': user_id,
            'age': age,
            'gender': gender,
            'citytier': citytier,
            'annualincome': annualincome,
            'occupation': occupation,
            'creditscore': creditscore,
            'avgmonthlyspend': avgmonthlyspend,
            'savingsrate': savingsrate,
            'investmentamountlastyear': investmentamountlastyear,
            'pastinvestments': pastinvestments
        }
        
        users.append(user_base)
    
    df = pd.DataFrame(users)
    print(f"✓ Generated {len(df)} basic synthetic users")
    return df


def save_basic_users_to_csv(df, filename='synthetic_users_basic.csv'):
    """Save basic users to CSV"""
    df.to_csv(filename, index=False)
    print(f"✓ Saved {len(df)} users to {filename}")
    print(f"\n📋 Available User IDs: {', '.join(df['customer_id'].tolist()[:5])}...")
    return filename


# =====================================================
# STEP 2: Process SINGLE User - Calculate Everything
# =====================================================

def deterministic_random(seed_str, low, high):
    """Generate deterministic pseudo-random number based on string seed"""
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (10**8)
    np.random.seed(seed)
    return np.random.uniform(low, high)


def calculate_derived_features(user_input):
    """Calculate derived features for a user"""
    seed_str = f"{user_input['age']}_{user_input['gender']}_{user_input['occupation']}_{user_input['creditscore']}_{user_input['pastinvestments']}"
    
    # 1. Credit Utilization Ratio
    if user_input['creditscore'] >= 750:
        creditutilizationratio = deterministic_random(seed_str+"1", 0.05, 0.20)
    elif user_input['creditscore'] >= 650:
        creditutilizationratio = deterministic_random(seed_str+"2", 0.20, 0.40)
    else:
        creditutilizationratio = deterministic_random(seed_str+"3", 0.40, 0.70)
    
    # 2. Debt to Income Ratio
    monthly_income = user_input['annualincome'] / 12
    monthly_surplus = monthly_income - user_input['avgmonthlyspend']
    estimated_emi = max(0, monthly_surplus * 0.25)
    debttoincomeratio = estimated_emi / monthly_income
    
    # 3. Transaction Volatility
    if user_input['occupation'] == 'Salaried' and user_input['creditscore'] >= 700:
        transactionvolatility = deterministic_random(seed_str+"4", 0.08, 0.15)
    elif user_input['occupation'] == 'Self-employed':
        transactionvolatility = deterministic_random(seed_str+"5", 0.20, 0.35)
    else:
        transactionvolatility = deterministic_random(seed_str+"6", 0.15, 0.25)
    
    # 4. Spending Stability Index
    if user_input['savingsrate'] >= 0.30:
        spendingstabilityindex = deterministic_random(seed_str+"7", 0.70, 0.85)
    elif user_input['savingsrate'] >= 0.15:
        spendingstabilityindex = deterministic_random(seed_str+"8", 0.55, 0.70)
    else:
        spendingstabilityindex = deterministic_random(seed_str+"9", 0.40, 0.55)
    
    # 5. Missed Payment Count
    if user_input['creditscore'] >= 750:
        missedpaymentcount = 0
    elif user_input['creditscore'] >= 650:
        missedpaymentcount = int(deterministic_random(seed_str+"10", 0, 1.99))
    else:
        missedpaymentcount = int(deterministic_random(seed_str+"11", 1, 3.99))
    
    # 6. Digital Activity Score
    if user_input['age'] < 30:
        digitalactivityscore = deterministic_random(seed_str+"12", 70, 85)
    elif user_input['age'] < 45:
        digitalactivityscore = deterministic_random(seed_str+"13", 55, 75)
    else:
        digitalactivityscore = deterministic_random(seed_str+"14", 40, 60)
    
    if user_input['citytier'] == 1:
        digitalactivityscore += 10
    if user_input['creditscore'] < 650:
        digitalactivityscore -= 10
    digitalactivityscore = max(0, min(100, digitalactivityscore))
    
    # 7. Portfolio Diversity Score
    investment_types = str(user_input['pastinvestments'])  # Convert to string
    if investment_types == 'None' or investment_types == 'nan':
        portfoliodiversityscore = 0
    elif '_' in investment_types or ',' in investment_types:
        portfoliodiversityscore = deterministic_random(seed_str+"15", 60, 80)
    else:
        portfoliodiversityscore = deterministic_random(seed_str+"16", 30, 50)
    
    return {
        'transactionvolatility': transactionvolatility,
        'spendingstabilityindex': spendingstabilityindex,
        'creditutilizationratio': creditutilizationratio,
        'debttoincomeratio': debttoincomeratio,
        'missedpaymentcount': missedpaymentcount,
        'digitalactivityscore': digitalactivityscore,
        'portfoliodiversityscore': portfoliodiversityscore
    }


def load_risk_model(model_path='risk_model.pkl'):
    """Load pre-trained risk model"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"✓ Risk model loaded from {model_path}")
        return model_data
    except FileNotFoundError:
        print(f"⚠️  Model file not found: {model_path}")
        print("Using fallback risk calculation...")
        return None


def predict_user_risk(user_data, model_data):
    """Predict risk for a single user using loaded model"""
    if model_data is None:
        # Fallback risk calculation
        if user_data['creditscore'] >= 750:
            risk_label = 'low'
            risk_score = 0.3
        elif user_data['creditscore'] >= 650:
            risk_label = 'medium'
            risk_score = 0.6
        else:
            risk_label = 'high'
            risk_score = 0.9
        
        return {
            'risk_label': risk_label,
            'risk_score': risk_score,
            'confidence_high': 0.33,
            'confidence_medium': 0.33,
            'confidence_low': 0.33
        }
    
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    
    # Create DataFrame
    user_df = pd.DataFrame([user_data])
    
    # Encode categorical variables
    for col in ['gender', 'occupation', 'pastinvestments']:
        if col in user_df.columns:
            col_values = user_df[col].astype(str)
            known_classes = set(label_encoders[col].classes_)
            col_values = col_values.apply(lambda x: x if x in known_classes else 'Unknown')
            user_df[f'{col}_encoded'] = label_encoders[col].transform(col_values)
    
    # Prepare features
    X = user_df[feature_columns].copy()
    X = X.fillna(X.median())
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    class_order = label_encoders['target'].classes_
    prob_dict = dict(zip(class_order, probabilities))
    risk_label = label_encoders['target'].inverse_transform([prediction])[0]
    
    # Calculate risk score (0-1 scale)
    risk_score_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
    risk_score = risk_score_map.get(risk_label, 0.5)
    
    return {
        'risk_label': risk_label,
        'risk_score': risk_score,
        'confidence_high': prob_dict.get('high', 0),
        'confidence_medium': prob_dict.get('medium', 0),
        'confidence_low': prob_dict.get('low', 0)
    }


def robust_normalize(value, min_val, max_val):
    """Normalize single value to [0, 1] range"""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)


def normalize_age_bucket(age):
    """Convert age to normalized bucket"""
    if age <= 30:
        return 0.3
    elif age <= 50:
        return 0.6
    else:
        return 0.9


def engineer_single_user_vector(user_complete):
    """Engineer 7-dimensional vector for SINGLE user"""
    
    # Normalize metrics (using reasonable ranges)
    normalized_risk_score = np.clip(user_complete['risk_score'], 0, 1)
    normalized_income = robust_normalize(user_complete['annualincome'], 200000, 2500000)
    normalized_savings_rate = np.clip(user_complete['savingsrate'], 0, 1)
    normalized_debt_to_income = robust_normalize(user_complete['debttoincomeratio'], 0, 0.5)
    normalized_digital_activity = robust_normalize(user_complete['digitalactivityscore'], 0, 100)
    normalized_portfolio_diversity = robust_normalize(user_complete['portfoliodiversityscore'], 0, 100)
    
    # Credit score normalization
    credit_min = 300
    credit_max = 850
    normalized_credit_score = (user_complete['creditscore'] - credit_min) / (credit_max - credit_min)
    normalized_credit_score = np.clip(normalized_credit_score, 0, 1)
    
    # Age bucket
    normalized_age_bucket = normalize_age_bucket(user_complete['age'])
    
    # 7-Dimensional Vector
    risk_preference = normalized_risk_score
    
    return_expectation = (
        0.40 * normalized_income +
        0.35 * normalized_savings_rate +
        0.25 * normalized_digital_activity
    )
    
    stability_preference = 1 - normalized_risk_score
    
    volatility_tolerance = (
        0.70 * normalized_risk_score +
        0.30 * normalized_digital_activity
    )
    
    market_cap_preference = 1 - normalized_debt_to_income
    
    dividend_preference = normalized_portfolio_diversity
    
    momentum_preference = (
        0.60 * normalized_digital_activity +
        0.40 * normalized_credit_score
    )
    
    # Clip all preferences
    vector = [
        np.clip(risk_preference, 0, 1),
        np.clip(return_expectation, 0, 1),
        np.clip(stability_preference, 0, 1),
        np.clip(volatility_tolerance, 0, 1),
        np.clip(market_cap_preference, 0, 1),
        np.clip(dividend_preference, 0, 1),
        np.clip(momentum_preference, 0, 1)
    ]
    
    return vector


def process_single_user(user_id, csv_file='synthetic_users_basic.csv', model_path='risk_model.pkl'):
    """Process a SINGLE user: calculate risk, features, and create JSON"""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING USER: {user_id}")
    print(f"{'='*60}")
    
    # Load CSV
    print(f"\n[1] Loading user data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Find user
    user_row = df[df['customer_id'] == user_id]
    
    if user_row.empty:
        print(f"❌ User {user_id} not found!")
        print(f"Available users: {', '.join(df['customer_id'].tolist()[:10])}...")
        return None
    
    user_base = user_row.iloc[0].to_dict()
    print(f"✓ Found user: {user_id}")
    print(f"   Age: {user_base['age']}, Gender: {user_base['gender']}")
    print(f"   Occupation: {user_base['occupation']}, Credit: {user_base['creditscore']}")
    
    # Calculate derived features
    print(f"\n[2] Calculating derived features...")
    derived_features = calculate_derived_features(user_base)
    print(f"✓ Calculated {len(derived_features)} derived features")
    
    # Merge
    user_complete = {**user_base, **derived_features}
    
    # Load model and predict risk
    print(f"\n[3] Predicting risk...")
    model_data = load_risk_model(model_path)
    risk_pred = predict_user_risk(user_complete, model_data)
    
    user_complete['risk_label'] = risk_pred['risk_label']
    user_complete['risk_score'] = risk_pred['risk_score']
    
    print(f"✓ Risk Label: {risk_pred['risk_label'].upper()}")
    print(f"✓ Risk Score: {risk_pred['risk_score']}")
    
    # Engineer vector
    print(f"\n[4] Engineering 7D preference vector...")
    vector = engineer_single_user_vector(user_complete)
    print(f"✓ Vector: {[round(v, 4) for v in vector]}")
    
    # Create JSON
    print(f"\n[5] Creating JSON output...")
    user_json = {
        "user_id": str(user_complete['customer_id']),
        "engineered_vector": [round(float(v), 6) for v in vector],
        "metadata": {
            "age": int(user_complete['age']),
            "gender": str(user_complete['gender']),
            "occupation": str(user_complete['occupation']),
            "risk_label": str(user_complete['risk_label']),
            "risk_score": round(float(user_complete['risk_score']), 2),
            "income": round(float(user_complete['annualincome']), 2),
            "credit_score": int(user_complete['creditscore']),
            "savings_rate": round(float(user_complete['savingsrate']), 4),
            "debt_to_income": round(float(user_complete['debttoincomeratio']), 4),
            "digital_activity": round(float(user_complete['digitalactivityscore']), 2),
            "portfolio_diversity": round(float(user_complete['portfoliodiversityscore']), 2)
        },
        "derived_features": {
            "transaction_volatility": round(float(user_complete['transactionvolatility']), 4),
            "spending_stability": round(float(user_complete['spendingstabilityindex']), 4),
            "credit_utilization": round(float(user_complete['creditutilizationratio']), 4),
            "missed_payments": int(user_complete['missedpaymentcount'])
        },
        "notes": "User vector aligned with stock & mutual fund embeddings"
    }
    
    # Save JSON
    output_file = f"user_{user_id}_profile.json"
    with open(output_file, 'w') as f:
        json.dump(user_json, f, indent=2)
    
    print(f"✓ Saved to {output_file}")
    
    print(f"\n{'='*60}")
    print(f"✓ USER PROCESSING COMPLETED!")
    print(f"{'='*60}")
    
    return user_json


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    
    print("="*60)
    print("TWO-STEP USER DATA PIPELINE")
    print("="*60)
    
    # ============ STEP 1: Generate Basic Users ============
    print("\n[STEP 1] Generate Basic Synthetic Users")
    print("-" * 60)
    
    NUM_USERS = 50
    df_basic = generate_basic_synthetic_users(NUM_USERS)
    csv_file = save_basic_users_to_csv(df_basic)
    
    print("\n✓ Step 1 Complete!")
    print(f"✓ {NUM_USERS} users saved to {csv_file}")
    print("\n📋 Sample User IDs:")
    for i in range(min(10, len(df_basic))):
        user = df_basic.iloc[i]
        print(f"   {i+1}. {user['customer_id']}: Age {user['age']}, {user['occupation']}, Credit {user['creditscore']}")
    
    # ============ STEP 2: User Selection ============
    print("\n\n[STEP 2] Select User to Process")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n👤 Enter User ID (e.g., USER_0001) or 'exit' to quit: ").strip()
            
            if user_input.lower() == 'exit':
                print("\n👋 Exiting pipeline. Goodbye!")
                break
            
            # Check if user exists
            if user_input not in df_basic['customer_id'].values:
                print(f"❌ User '{user_input}' not found! Please choose from the list above.")
                continue
            
            # Process selected user
            print(f"\n>>> Processing User: {user_input}")
            
            user_json = process_single_user(
                user_id=user_input,
                csv_file=csv_file,
                model_path='risk_model.pkl'
            )
            
            if user_json:
                print("\n✅ User processing successful!")
                print(f"📄 JSON saved: user_{user_input}_profile.json")
            
            # Ask if they want to process another user
            another = input("\n🔄 Process another user? (y/n): ").strip().lower()
            if another != 'y':
                print("\n👋 Pipeline completed. Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Pipeline interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")
    
    print("\n" + "="*60)
    print("PIPELINE USAGE SUMMARY:")
    print("="*60)
    print("✓ synthetic_users_basic.csv - Contains all basic user data")
    print("✓ user_[ID]_profile.json - Individual user risk & features")
    print("\nTo process more users, run this script again!")
    print("="*60)