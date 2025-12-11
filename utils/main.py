# integrated_pipeline.py

import pandas as pd
import numpy as np
import json
import hashlib
import pickle
from pathlib import Path
import os
import glob

# =====================================================
# PART 1: USER GENERATION & RISK MODELING
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
        user_id = f"USER_{i:04d}"
        age = np.random.randint(22, 65)
        gender = np.random.choice(genders)
        citytier = np.random.choice(city_tiers)
        
        if citytier == 1:
            annualincome = np.random.randint(400000, 2500000)
        elif citytier == 2:
            annualincome = np.random.randint(300000, 1200000)
        else:
            annualincome = np.random.randint(200000, 800000)
        
        occupation = np.random.choice(occupations)
        creditscore = np.random.randint(550, 850)
        avgmonthlyspend = int(annualincome / 12 * np.random.uniform(0.5, 0.85))
        savingsrate = round(np.random.uniform(0.05, 0.40), 2)
        investmentamountlastyear = int(annualincome * np.random.uniform(0.0, 0.15))
        pastinvestments = np.random.choice(investment_types)
        
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
    return filename


def deterministic_random(seed_str, low, high):
    """Generate deterministic pseudo-random number based on string seed"""
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (10**8)
    np.random.seed(seed)
    return np.random.uniform(low, high)


def calculate_derived_features(user_input):
    """Calculate derived features for a user"""
    seed_str = f"{user_input['age']}_{user_input['gender']}_{user_input['occupation']}_{user_input['creditscore']}_{user_input['pastinvestments']}"
    
    if user_input['creditscore'] >= 750:
        creditutilizationratio = deterministic_random(seed_str+"1", 0.05, 0.20)
    elif user_input['creditscore'] >= 650:
        creditutilizationratio = deterministic_random(seed_str+"2", 0.20, 0.40)
    else:
        creditutilizationratio = deterministic_random(seed_str+"3", 0.40, 0.70)
    
    monthly_income = user_input['annualincome'] / 12
    monthly_surplus = monthly_income - user_input['avgmonthlyspend']
    estimated_emi = max(0, monthly_surplus * 0.25)
    debttoincomeratio = estimated_emi / monthly_income
    
    if user_input['occupation'] == 'Salaried' and user_input['creditscore'] >= 700:
        transactionvolatility = deterministic_random(seed_str+"4", 0.08, 0.15)
    elif user_input['occupation'] == 'Self-employed':
        transactionvolatility = deterministic_random(seed_str+"5", 0.20, 0.35)
    else:
        transactionvolatility = deterministic_random(seed_str+"6", 0.15, 0.25)
    
    if user_input['savingsrate'] >= 0.30:
        spendingstabilityindex = deterministic_random(seed_str+"7", 0.70, 0.85)
    elif user_input['savingsrate'] >= 0.15:
        spendingstabilityindex = deterministic_random(seed_str+"8", 0.55, 0.70)
    else:
        spendingstabilityindex = deterministic_random(seed_str+"9", 0.40, 0.55)
    
    if user_input['creditscore'] >= 750:
        missedpaymentcount = 0
    elif user_input['creditscore'] >= 650:
        missedpaymentcount = int(deterministic_random(seed_str+"10", 0, 1.99))
    else:
        missedpaymentcount = int(deterministic_random(seed_str+"11", 1, 3.99))
    
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
    
    investment_types = str(user_input['pastinvestments'])
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
    
    user_df = pd.DataFrame([user_data])
    
    for col in ['gender', 'occupation', 'pastinvestments']:
        if col in user_df.columns:
            col_values = user_df[col].astype(str)
            known_classes = set(label_encoders[col].classes_)
            col_values = col_values.apply(lambda x: x if x in known_classes else 'Unknown')
            user_df[f'{col}_encoded'] = label_encoders[col].transform(col_values)
    
    X = user_df[feature_columns].copy()
    X = X.fillna(X.median())
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    class_order = label_encoders['target'].classes_
    prob_dict = dict(zip(class_order, probabilities))
    risk_label = label_encoders['target'].inverse_transform([prediction])[0]
    
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
    
    normalized_risk_score = np.clip(user_complete['risk_score'], 0, 1)
    normalized_income = robust_normalize(user_complete['annualincome'], 200000, 2500000)
    normalized_savings_rate = np.clip(user_complete['savingsrate'], 0, 1)
    normalized_debt_to_income = robust_normalize(user_complete['debttoincomeratio'], 0, 0.5)
    normalized_digital_activity = robust_normalize(user_complete['digitalactivityscore'], 0, 100)
    normalized_portfolio_diversity = robust_normalize(user_complete['portfoliodiversityscore'], 0, 100)
    
    credit_min = 300
    credit_max = 850
    normalized_credit_score = (user_complete['creditscore'] - credit_min) / (credit_max - credit_min)
    normalized_credit_score = np.clip(normalized_credit_score, 0, 1)
    
    normalized_age_bucket = normalize_age_bucket(user_complete['age'])
    
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
    
    print(f"\n[1] Loading user data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    user_row = df[df['customer_id'] == user_id]
    
    if user_row.empty:
        print(f"❌ User {user_id} not found!")
        return None
    
    user_base = user_row.iloc[0].to_dict()
    print(f"✓ Found user: {user_id}")
    print(f"   Age: {user_base['age']}, Gender: {user_base['gender']}")
    print(f"   Occupation: {user_base['occupation']}, Credit: {user_base['creditscore']}")
    
    print(f"\n[2] Calculating derived features...")
    derived_features = calculate_derived_features(user_base)
    print(f"✓ Calculated {len(derived_features)} derived features")
    
    user_complete = {**user_base, **derived_features}
    
    print(f"\n[3] Predicting risk...")
    model_data = load_risk_model(model_path)
    risk_pred = predict_user_risk(user_complete, model_data)
    
    user_complete['risk_label'] = risk_pred['risk_label']
    user_complete['risk_score'] = risk_pred['risk_score']
    
    print(f"✓ Risk Label: {risk_pred['risk_label'].upper()}")
    print(f"✓ Risk Score: {risk_pred['risk_score']}")
    
    print(f"\n[4] Engineering 7D preference vector...")
    vector = engineer_single_user_vector(user_complete)
    print(f"✓ Vector: {[round(v, 4) for v in vector]}")
    
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
    
    output_file = f"user_{user_id}_profile.json"
    with open(output_file, 'w') as f:
        json.dump(user_json, f, indent=2)
    
    print(f"✓ Saved to {output_file}")
    print(f"\n{'='*60}")
    print(f"✓ USER PROCESSING COMPLETED!")
    print(f"{'='*60}")
    
    return user_json


# =====================================================
# PART 2: RECOMMENDATION ENGINE
# =====================================================

def load_json(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    u = np.array(vec1)
    v = np.array(vec2)
    
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0
    
    similarity = dot_product / (norm_u * norm_v)
    return float(similarity)


def get_stock_recommendations(user_vector, stocks, top_k=10):
    """Calculate cosine similarity between user and all stocks"""
    recommendations = []
    
    for stock in stocks:
        stock_vector = stock['engineered_vector']
        similarity = cosine_similarity(user_vector, stock_vector)
        
        recommendations.append({
            'symbol': stock['symbol'],
            'company_name': stock.get('company_name', ''),
            'similarity_score': round(similarity, 6),
            'metadata': stock.get('metadata', {})
        })
    
    recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
    return recommendations[:top_k]


def get_mutual_fund_recommendations(user_vector, mutual_funds, top_k=10):
    """Calculate cosine similarity between user and all mutual funds"""
    recommendations = []
    
    for fund in mutual_funds:
        fund_vector = fund['engineered_vector']
        similarity = cosine_similarity(user_vector, fund_vector)
        
        recommendations.append({
            'fund_name': fund['fund_name'],
            'fund_link': fund.get('fund_link', ''),
            'similarity_score': round(similarity, 6),
            'metadata': fund.get('metadata', {})
        })
    
    recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
    return recommendations[:top_k]


def generate_recommendations_for_user(user_json, stocks_json_path, mutual_funds_json_path, top_k=10):
    """Generate recommendations for a specific user JSON"""
    
    print(f"\n{'='*60}")
    print(f"GENERATING RECOMMENDATIONS FOR {user_json['user_id']}")
    print(f"{'='*60}")
    
    try:
        user_vector = user_json['engineered_vector']
        
        print(f"\n[1] Loading stocks from: {stocks_json_path}")
        stocks = load_json(stocks_json_path)
        print(f"   ✓ Loaded {len(stocks)} stocks")
        
        print(f"\n[2] Loading mutual funds from: {mutual_funds_json_path}")
        mutual_funds = load_json(mutual_funds_json_path)
        print(f"   ✓ Loaded {len(mutual_funds)} mutual funds")
        
        print(f"\n[3] Calculating stock recommendations...")
        stock_recommendations = get_stock_recommendations(user_vector, stocks, top_k)
        print(f"   ✓ Found top {len(stock_recommendations)} stock matches")
        
        print(f"\n[4] Calculating mutual fund recommendations...")
        fund_recommendations = get_mutual_fund_recommendations(user_vector, mutual_funds, top_k)
        print(f"   ✓ Found top {len(fund_recommendations)} mutual fund matches")
        
        result = {
            "user_id": user_json['user_id'],
            "user_metadata": user_json.get('metadata', {}),
            "top_stock_recommendations": stock_recommendations,
            "top_mutual_fund_recommendations": fund_recommendations
        }
        
        output_path = f"recommendations_{user_json['user_id']}.json"
        print(f"\n[5] Saving recommendations to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"   ✓ Recommendations saved successfully")
        
        # Print summary
        print("\n" + "="*60)
        print("RECOMMENDATION SUMMARY")
        print("="*60)
        print(f"User ID: {result['user_id']}")
        print(f"Risk Level: {result['user_metadata'].get('risk_label', 'N/A').upper()}")
        print(f"Income: ₹{result['user_metadata'].get('income', 0):,.0f}")
        
        print(f"\n📈 TOP {min(5, len(stock_recommendations))} STOCK RECOMMENDATIONS:")
        for i, stock in enumerate(stock_recommendations[:5], 1):
            print(f"{i}. {stock['symbol']} - {stock['company_name']}")
            print(f"   Similarity: {stock['similarity_score']:.4f}")
            print(f"   Sector: {stock['metadata'].get('sector', 'N/A')}")
        
        print(f"\n📊 TOP {min(5, len(fund_recommendations))} MUTUAL FUND RECOMMENDATIONS:")
        for i, fund in enumerate(fund_recommendations[:5], 1):
            print(f"{i}. {fund['fund_name'][:50]}")
            print(f"   Similarity: {fund['similarity_score']:.4f}")
        
        print("\n" + "="*60)
        print("✓ RECOMMENDATIONS GENERATED SUCCESSFULLY!")
        print("="*60)
        
        return result
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =====================================================
# MAIN INTEGRATED PIPELINE
# =====================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   INTEGRATED FINANCIAL RECOMMENDATION PIPELINE")
    print("="*70)
    
    # Check if stocks and funds JSON exist
    stocks_json = "engineered_stocks.json"
    funds_json = "engineered_funds.json"
    
    if not os.path.exists(stocks_json):
        print(f"\n❌ ERROR: {stocks_json} not found!")
        print("Please ensure the stocks JSON file exists.")
        exit(1)
    
    if not os.path.exists(funds_json):
        print(f"\n❌ ERROR: {funds_json} not found!")
        print("Please ensure the mutual funds JSON file exists.")
        exit(1)
    
    # ============ STEP 1: Generate Basic Users ============
    print("\n[STEP 1] Generate Basic Synthetic Users")
    print("-" * 70)
    
    NUM_USERS = 50
    csv_file = 'synthetic_users_basic.csv'
    
    # Check if CSV already exists
    if os.path.exists(csv_file):
        print(f"✓ Found existing {csv_file}")
        df_basic = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(df_basic)} users")
    else:
        df_basic = generate_basic_synthetic_users(NUM_USERS)
        save_basic_users_to_csv(df_basic, csv_file)
    
    print("\n📋 Available Users (first 10):")
    for i in range(min(10, len(df_basic))):
        user = df_basic.iloc[i]
        print(f"   {i+1}. {user['customer_id']}: Age {user['age']}, {user['occupation']}, Credit {user['creditscore']}")
    
    # ============ STEP 2: User Processing & Recommendations ============
    print("\n\n[STEP 2] Process User & Generate Recommendations")
    print("-" * 70)
    
    while True:
        try:
            user_input = input("\n👤 Enter User ID (e.g., USER_0001) or 'exit' to quit: ").strip()
            
            if user_input.lower() == 'exit':
                print("\n👋 Exiting pipeline. Goodbye!")
                break
            
            if user_input not in df_basic['customer_id'].values:
                print(f"❌ User '{user_input}' not found! Please choose from the list above.")
                continue
            
            # STEP 2A: Process User (Risk Model)
            print(f"\n{'='*70}")
            print(f"STEP 2A: RISK ANALYSIS FOR {user_input}")
            print(f"{'='*70}")
            
            user_json = process_single_user(
                user_id=user_input,
                csv_file=csv_file,
                model_path='risk_model.pkl'
            )
            
            if not user_json:
                print(f"❌ Failed to process user {user_input}")
                continue
            
            # STEP 2B: Generate Recommendations
            print(f"\n{'='*70}")
            print(f"STEP 2B: RECOMMENDATIONS FOR {user_input}")
            print(f"{'='*70}")
            
            recommendations = generate_recommendations_for_user(
                user_json=user_json,
                stocks_json_path=stocks_json,
                mutual_funds_json_path=funds_json,
                top_k=10
            )
            
            if recommendations:
                print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY FOR {user_input}!")
                print(f"\n📁 Generated Files:")
                print(f"   1. user_{user_input}_profile.json (Risk Profile)")
                print(f"   2. recommendations_{user_input}.json (Recommendations)")
            
            # Ask to process another
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
    
    print("\n" + "="*70)
    print("PIPELINE SUMMARY:")
    print("="*70)
    print("✓ synthetic_users_basic.csv - All basic user data")
    print("✓ user_[ID]_profile.json - Individual user risk profiles")
    print("✓ recommendations_[ID].json - Personalized recommendations")
    print("\nThank you for using the Financial Recommendation Pipeline!")
    print("="*70)