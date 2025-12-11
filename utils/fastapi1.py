# main.py - FastAPI Backend

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import json
import hashlib
import pickle
from datetime import datetime
import os

app = FastAPI(title="Financial Recommendation API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# PYDANTIC MODELS
# =====================================================

class UserProfile(BaseModel):
    customer_id: str
    age: int
    gender: str
    citytier: int
    annualincome: float
    occupation: str
    creditscore: int
    avgmonthlyspend: float
    savingsrate: float
    investmentamountlastyear: float
    pastinvestments: str

class UserResponse(BaseModel):
    user_id: str
    engineered_vector: List[float]
    metadata: Dict
    derived_features: Dict
    notes: str

class RecommendationResponse(BaseModel):
    user_id: str
    user_metadata: Dict
    top_stock_recommendations: List[Dict]
    top_mutual_fund_recommendations: List[Dict]

class AdminLog(BaseModel):
    timestamp: str
    action: str
    status: str
    details: str

# =====================================================
# GLOBAL VARIABLES & CACHE
# =====================================================

users_df = None
model_data = None
stocks_data = None
funds_data = None
admin_logs = []

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def deterministic_random(seed_str: str, low: float, high: float) -> float:
    """Generate deterministic pseudo-random number"""
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (10**8)
    np.random.seed(seed)
    return np.random.uniform(low, high)

def calculate_derived_features(user_input: Dict) -> Dict:
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

def predict_user_risk(user_data: Dict, model_data) -> Dict:
    """Predict risk for a single user"""
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
            'risk_score': risk_score
        }
    
    # Model prediction logic here
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
    risk_label = label_encoders['target'].inverse_transform([prediction])[0]
    
    risk_score_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
    risk_score = risk_score_map.get(risk_label, 0.5)
    
    return {
        'risk_label': risk_label,
        'risk_score': risk_score
    }

def robust_normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize single value to [0, 1] range"""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)

def engineer_single_user_vector(user_complete: Dict) -> List[float]:
    """Engineer 7-dimensional vector for user"""
    
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
    
    risk_preference = normalized_risk_score
    return_expectation = 0.40 * normalized_income + 0.35 * normalized_savings_rate + 0.25 * normalized_digital_activity
    stability_preference = 1 - normalized_risk_score
    volatility_tolerance = 0.70 * normalized_risk_score + 0.30 * normalized_digital_activity
    market_cap_preference = 1 - normalized_debt_to_income
    dividend_preference = normalized_portfolio_diversity
    momentum_preference = 0.60 * normalized_digital_activity + 0.40 * normalized_credit_score
    
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

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity"""
    u = np.array(vec1)
    v = np.array(vec2)
    
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0
    
    return float(dot_product / (norm_u * norm_v))

def add_log(action: str, status: str, details: str):
    """Add admin log"""
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "status": status,
        "details": details
    }
    admin_logs.append(log)
    if len(admin_logs) > 100:  # Keep only last 100 logs
        admin_logs.pop(0)

# =====================================================
# STARTUP EVENT
# =====================================================

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    global users_df, model_data, stocks_data, funds_data
    
    try:
        # Load users CSV
        if os.path.exists('reco_dummy_gpt.csv'):
            users_df = pd.read_csv('reco_dummy_gpt.csv')
            add_log("STARTUP", "SUCCESS", f"Loaded {len(users_df)} users from CSV")
        else:
            add_log("STARTUP", "WARNING", "Users CSV not found")
        
        # Load model
        if os.path.exists('risk_model.pkl'):
            with open('risk_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            add_log("STARTUP", "SUCCESS", "Risk model loaded")
        else:
            add_log("STARTUP", "WARNING", "Risk model not found, using fallback")
        
        # Load stocks
        if os.path.exists('engineered_stocks.json'):
            with open('engineered_stocks.json', 'r') as f:
                stocks_data = json.load(f)
            add_log("STARTUP", "SUCCESS", f"Loaded {len(stocks_data)} stocks")
        else:
            add_log("STARTUP", "WARNING", "Stocks JSON not found")
        
        # Load mutual funds
        if os.path.exists('engineered_funds.json'):
            with open('engineered_funds.json', 'r') as f:
                funds_data = json.load(f)
            add_log("STARTUP", "SUCCESS", f"Loaded {len(funds_data)} funds")
        else:
            add_log("STARTUP", "WARNING", "Funds JSON not found")
        
        add_log("STARTUP", "SUCCESS", "API initialized successfully")
        
    except Exception as e:
        add_log("STARTUP", "ERROR", str(e))

# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "users": "/api/users",
            "user_profile": "/api/user/{user_id}",
            "recommendations": "/api/recommendations/{user_id}",
            "admin_logs": "/api/admin/logs"
        }
    }

# @app.get("/api/users")
# async def get_all_users():
#     """Get list of all users"""
#     if users_df is None:
#         raise HTTPException(status_code=404, detail="Users data not loaded")
    
#     users_list = users_df.to_dict('records')
#     add_log("GET_USERS", "SUCCESS", f"Retrieved {len(users_list)} users")
    
#     return {
#         "total_users": len(users_list),
#         "users": users_list
#     }


@app.get("/api/users")
async def get_all_users():
    """Get list of all users"""
    if users_df is None:
        raise HTTPException(status_code=404, detail="Users data not loaded")
    
    # Replace NaN with None for JSON serialization
    users_list = users_df.replace({np.nan: None}).to_dict('records')
    add_log("GET_USERS", "SUCCESS", f"Retrieved {len(users_list)} users")
    
    return {
        "total_users": len(users_list),
        "users": users_list
    }

@app.get("/api/user/{user_id}")
async def get_user_profile(user_id: int):
    """Get complete user profile with risk analysis"""
    if users_df is None:
        raise HTTPException(status_code=404, detail="Users data not loaded")
    
    user_row = users_df[users_df['customer_id'] == user_id]
    
    if user_row.empty:
        add_log("GET_USER", "ERROR", f"User {user_id} not found")
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    user_base = user_row.iloc[0].to_dict()
    
    # Calculate derived features
    derived_features = calculate_derived_features(user_base)
    user_complete = {**user_base, **derived_features}
    
    # Predict risk
    risk_pred = predict_user_risk(user_complete, model_data)
    user_complete['risk_label'] = risk_pred['risk_label']
    user_complete['risk_score'] = risk_pred['risk_score']
    
    # Engineer vector
    vector = engineer_single_user_vector(user_complete)
    
    result = {
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
    
    add_log("GET_USER", "SUCCESS", f"Retrieved profile for {user_id}")
    return result

@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: int, top_k: int = 10):
    """Get stock and mutual fund recommendations for a user"""
    
    if stocks_data is None or funds_data is None:
        raise HTTPException(status_code=404, detail="Stocks or Funds data not loaded")
    
    # Get user profile first
    user_profile = await get_user_profile(user_id)
    user_vector = user_profile['engineered_vector']
    
    # Calculate stock recommendations
    stock_recommendations = []
    for stock in stocks_data:
        stock_vector = stock['engineered_vector']
        similarity = cosine_similarity(user_vector, stock_vector)
        
        stock_recommendations.append({
            'symbol': stock['symbol'],
            'company_name': stock.get('company_name', ''),
            'similarity_score': round(similarity, 6),
            'metadata': stock.get('metadata', {})
        })
    
    stock_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
    stock_recommendations = stock_recommendations[:top_k]
    
    # Calculate mutual fund recommendations
    fund_recommendations = []
    for fund in funds_data:
        fund_vector = fund['engineered_vector']
        similarity = cosine_similarity(user_vector, fund_vector)
        
        fund_recommendations.append({
            'fund_name': fund['fund_name'],
            'fund_link': fund.get('fund_link', ''),
            'similarity_score': round(similarity, 6),
            'metadata': fund.get('metadata', {})
        })
    
    fund_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
    fund_recommendations = fund_recommendations[:top_k]
    
    result = {
        "user_id": user_profile['user_id'],
        "user_metadata": user_profile['metadata'],
        "top_stock_recommendations": stock_recommendations,
        "top_mutual_fund_recommendations": fund_recommendations
    }
    
    add_log("GET_RECOMMENDATIONS", "SUCCESS", f"Generated recommendations for {user_id}")
    return result

@app.get("/api/admin/logs")
async def get_admin_logs():
    """Get admin logs"""
    return {
        "total_logs": len(admin_logs),
        "logs": admin_logs
    }

@app.post("/api/admin/refresh-data")
async def refresh_data(background_tasks: BackgroundTasks):
    """Refresh data from files"""
    try:
        global users_df, model_data, stocks_data, funds_data
        
        # Reload users
        if os.path.exists('reco_dummy_gpt.csv'):
            users_df = pd.read_csv('reco_dummy_gpt.csv')
            add_log("REFRESH", "SUCCESS", f"Reloaded {len(users_df)} users")
        
        # Reload stocks
        if os.path.exists('engineered_stocks.json'):
            with open('engineered_stocks.json', 'r') as f:
                stocks_data = json.load(f)
            add_log("REFRESH", "SUCCESS", f"Reloaded {len(stocks_data)} stocks")
        
        # Reload funds
        if os.path.exists('engineered_funds.json'):
            with open('engineered_funds.json', 'r') as f:
                funds_data = json.load(f)
            add_log("REFRESH", "SUCCESS", f"Reloaded {len(funds_data)} funds")
        
        return {"status": "success", "message": "Data refreshed successfully"}
        
    except Exception as e:
        add_log("REFRESH", "ERROR", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_statistics():
    """Get system statistics"""
    return {
        "users_loaded": len(users_df) if users_df is not None else 0,
        "stocks_loaded": len(stocks_data) if stocks_data is not None else 0,
        "funds_loaded": len(funds_data) if funds_data is not None else 0,
        "model_loaded": model_data is not None,
        "total_logs": len(admin_logs)
    }

# =====================================================
# RUN SERVER
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)