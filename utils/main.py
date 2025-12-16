# main.py - FastAPI Backend with LLM Integration

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json
import hashlib
import pickle
from datetime import datetime
import os
import aiohttp
import asyncio
import hashlib
import schedule
import threading
import subprocess
import sys
from pathlib import Path
import time

app = FastAPI(title="Financial Recommendation API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class SchedulerConfig:
    BASE_DIR = Path(__file__).parent.absolute()
    
    # Script paths
    STOCK_DATA_SCRIPT = str(BASE_DIR / "data_injection" / "stock_data_gathering.py")
    STOCK_VECTOR_SCRIPT = str(BASE_DIR / "data_injection" / "stock_vector.py")

    
    FUND_DATA_SCRIPT = str(BASE_DIR / "data_injection"/ "mutual_fund_data_gathering.py")
    FUND_VECTOR_SCRIPT = str(BASE_DIR /"data_injection"/ "mutualfund_vector.py")

    # ⏰ STOCK SCHEDULING (Daily)
    STOCK_SCHEDULE_TIME = "02:00"   # daily 2 AM

    # 📅 FUND SCHEDULING (Weekly)
    FUND_SCHEDULE_DAY = "sunday"    # once a week
    FUND_SCHEDULE_TIME = "03:00"    # sunday 3 AM

    ENABLED = True

    # Timeout
    GATHERING_TIMEOUT = 7200
    VECTOR_TIMEOUT = 1800



scheduler_status = {
    "enabled": SchedulerConfig.ENABLED,
    "last_run": None,
    "next_run": None,
    "is_running": False,
    "last_result": None
}    

# =====================================================
# OLLAMA LLM CONFIGURATION
# =====================================================

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"  # Change to your preferred model

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

class ExplanationRequest(BaseModel):
    user_profile: Dict
    top_stocks: List[Dict]
    top_mutual_funds: List[Dict]

class IndividualExplanationRequest(BaseModel):
    user_profile: Dict
    item_type: str  # "stock" or "mutual_fund"
    item_data: Dict

class ExplanationResponse(BaseModel):
    explanation: str
    status: str
    metadata: Dict

# =====================================================
# GLOBAL VARIABLES & CACHE
# =====================================================

users_df = None
model_data = None
stocks_data = None
funds_data = None
admin_logs = []







#sechuler.py


def run_script_background(script_path: str, timeout: int, script_name: str) -> tuple:
    """Run a Python script in background"""
    try:
        if not Path(script_path).exists():
            return False, f"Script not found: {script_path}"
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            add_log("SCHEDULER", "SUCCESS", f"{script_name} completed")
            return True, result.stdout
        else:
            add_log("SCHEDULER", "ERROR", f"{script_name} failed: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        add_log("SCHEDULER", "ERROR", f"{script_name} timed out")
        return False, f"Script timed out after {timeout}s"
    except Exception as e:
        add_log("SCHEDULER", "ERROR", f"{script_name} error: {str(e)}")
        return False, str(e)


def run_pipeline_background(data_type: str, gathering_script: str, vector_script: str) -> bool:
    """Run a complete data pipeline"""
    add_log("SCHEDULER", "INFO", f"Starting {data_type} pipeline...")
    
    # Data gathering
    success, output = run_script_background(
        gathering_script,
        SchedulerConfig.GATHERING_TIMEOUT,
        f"{data_type} Data Gathering"
    )
    
    if not success:
        return False
    
    # Feature engineering
    success, output = run_script_background(
        vector_script,
        SchedulerConfig.VECTOR_TIMEOUT,
        f"{data_type} Feature Engineering"
    )
    
    return success


def scheduled_data_refresh():
    """Main scheduled job - runs all pipelines and refreshes data"""
    if scheduler_status["is_running"]:
        add_log("SCHEDULER", "WARNING", "Pipeline already running, skipping...")
        return
    
    scheduler_status["is_running"] = True
    scheduler_status["last_run"] = datetime.now().isoformat()
    
    add_log("SCHEDULER", "INFO", "Starting scheduled data refresh...")
    
    results = {
        "stocks": False,
        "funds": False,
        "insurance": False
    }
    
    try:
        # Run stock pipeline
        if Path(SchedulerConfig.STOCK_DATA_SCRIPT).exists():
            results["stocks"] = run_pipeline_background(
                "stocks",
                SchedulerConfig.STOCK_DATA_SCRIPT,
                SchedulerConfig.STOCK_VECTOR_SCRIPT
            )
        
        # Run mutual fund pipeline
        if Path(SchedulerConfig.FUND_DATA_SCRIPT).exists():
            results["funds"] = run_pipeline_background(
                "funds",
                SchedulerConfig.FUND_DATA_SCRIPT,
                SchedulerConfig.FUND_VECTOR_SCRIPT
            )
        
        # Run insurance pipeline
        # if Path(SchedulerConfig.INSURANCE_DATA_SCRIPT).exists():
        #     results["insurance"] = run_pipeline_background(
        #         "insurance",
        #         SchedulerConfig.INSURANCE_DATA_SCRIPT,
        #         SchedulerConfig.INSURANCE_VECTOR_SCRIPT
        #     )
        
        # Reload data into memory
        if any(results.values()):
            load_all_data() 
            add_log("SCHEDULER", "SUCCESS", "Data refreshed successfully")
        
        scheduler_status["last_result"] = results
        
    except Exception as e:
        add_log("SCHEDULER", "ERROR", f"Scheduler error: {str(e)}")
        scheduler_status["last_result"] = {"error": str(e)}
    
    finally:
        scheduler_status["is_running"] = False


# def run_scheduler_thread():
#     """Run scheduler in background thread"""
#     schedule.every().day.at(SchedulerConfig.SCHEDULE_TIME).do(scheduled_data_refresh)
    
#     add_log("SCHEDULER", "INFO", f"Scheduler started - runs daily at {SchedulerConfig.SCHEDULE_TIME}")
    
#     while True:
#         schedule.run_pending()
#         scheduler_status["next_run"] = str(schedule.next_run()) if schedule.jobs else None
#         asyncio.run(asyncio.sleep(60))  # Check every 



def run_scheduler_thread():
    """Run scheduler in background thread"""
    # Stock pipeline - Daily
    schedule.every().day.at(SchedulerConfig.STOCK_SCHEDULE_TIME).do(
        lambda: run_pipeline_background(
            "Stock",
            SchedulerConfig.STOCK_DATA_SCRIPT,
            SchedulerConfig.STOCK_VECTOR_SCRIPT
        )
    )
    
    # Fund pipeline - Weekly
    getattr(schedule.every(), SchedulerConfig.FUND_SCHEDULE_DAY).at(
        SchedulerConfig.FUND_SCHEDULE_TIME
    ).do(
        lambda: run_pipeline_background(
            "Fund",
            SchedulerConfig.FUND_DATA_SCRIPT,
            SchedulerConfig.FUND_VECTOR_SCRIPT
        )
    )
    
    add_log("SCHEDULER", "INFO", f"Scheduler started - Stock: Daily {SchedulerConfig.STOCK_SCHEDULE_TIME}, Fund: Weekly {SchedulerConfig.FUND_SCHEDULE_DAY} {SchedulerConfig.FUND_SCHEDULE_TIME}")
    
    while True:
        schedule.run_pending()
        scheduler_status["next_run"] = str(schedule.next_run()) if schedule.jobs else None
        time.sleep(60)  # 🔥 YE CHANGE KARO - asyncio.sleep nahi, normal sleep!


def start_scheduler():
    """Start scheduler in daemon thread"""
    if not SchedulerConfig.ENABLED:
        add_log("SCHEDULER", "INFO", "Scheduler disabled in config")
        return
    
    scheduler_thread = threading.Thread(target=run_scheduler_thread, daemon=True)
    scheduler_thread.start()
    add_log("SCHEDULER", "SUCCESS", "Scheduler thread started")

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
    if len(admin_logs) > 100:
        admin_logs.pop(0)

# =====================================================
# LLM HELPER FUNCTIONS
# =====================================================

def generate_fallback_explanation(
    user_profile: Dict,
    top_stocks: List[Dict],
    top_mutual_funds: List[Dict]
) -> str:
    """Generate rule-based explanation when LLM is unavailable"""
    
    risk = user_profile['risk_label'].upper()
    income = user_profile['income']
    savings_rate = user_profile['savings_rate'] * 100
    
    explanation = f"""**Why These Recommendations Match Your Profile:**

• **Risk Alignment**: Your {risk} risk profile has been carefully matched with investments that suit your risk tolerance and financial goals.

• **Income-Based Selection**: With an annual income of ₹{income:,.0f} and a savings rate of {savings_rate:.1f}%, these options are sized appropriately for your financial capacity.

• **Diversification**: The recommendations span multiple sectors and categories to help balance your portfolio and reduce concentration risk.

• **Match Quality**: All recommendations have high similarity scores (80%+), indicating strong alignment with your complete financial profile.

⚠️ **Important**: Past performance doesn't guarantee future returns. Market investments carry risk - please consult a financial advisor before making investment decisions."""
    
    return explanation

async def generate_llm_explanation(
    user_profile: Dict,
    top_stocks: List[Dict],
    top_mutual_funds: List[Dict]
) -> str:
    """Generate explanation using Ollama LLM"""
    
    # Format stocks for prompt
    stocks_text = "\n".join([
        f"- {s['symbol']} ({s['company_name']}): Match Score {s['similarity_score']*100:.1f}%, "
        f"Sector: {s['metadata'].get('sector', 'N/A')}"
        for s in top_stocks[:5]
    ])
    
    # Format mutual funds for prompt
    funds_text = "\n".join([
        f"- {f['fund_name'][:60]}: Match Score {f['similarity_score']*100:.1f}%, "
        f"Category: {f['metadata'].get('category', 'N/A')}"
        for f in top_mutual_funds[:5]
    ])
    
    # Create the prompt for a which recommedation suit the user 
    prompt = f"""You are a financial insights assistant inside Raj's investment app.
Your task is to explain WHY the recommended stocks and mutual funds match the user's profile.

Guidelines:
- Use simple, human-friendly language.
- Explain in short bullet points (4-6 points maximum).
- Mention factors like risk level, volatility, goal alignment, sector stability, and past consistency.
- Never give financial advice or guaranteed returns.
- Keep the answer within 6-7 lines total.
- Add 1-2 generic caution notes about market risk at the end.

User Profile:
- Age: {user_profile['age']} years
- Occupation: {user_profile['occupation']}
- Risk Profile: {user_profile['risk_label'].upper()}
- Risk Score: {user_profile['risk_score']}
- Annual Income: ₹{user_profile['income']:,.0f}
- Credit Score: {user_profile['credit_score']}
- Savings Rate: {user_profile['savings_rate']*100:.1f}%
- Debt-to-Income Ratio: {user_profile['debt_to_income']*100:.1f}%
- Digital Activity Score: {user_profile['digital_activity']:.0f}/100
- Portfolio Diversity Score: {user_profile['portfolio_diversity']:.0f}/100

Top Stock Recommendations:
{stocks_text}

Top Mutual Fund Recommendations:
{funds_text}

Provide a clear, short explanation the user can understand. Focus on why these recommendations suit their profile."""

    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            async with session.post(OLLAMA_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    result = await response.json()
                    explanation = result.get('response', '').strip()
                    
                    if not explanation:
                        return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)
                    
                    return explanation
                else:
                    add_log("LLM_EXPLAIN", "ERROR", f"Ollama API error: {response.status}")
                    return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)
    
    except asyncio.TimeoutError:
        add_log("LLM_EXPLAIN", "ERROR", "Ollama API timeout")
        return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)
    
    except Exception as e:
        add_log("LLM_EXPLAIN", "ERROR", f"Ollama error: {str(e)}")
        return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)

async def generate_individual_llm_explanation(
    user_profile: Dict,
    item_type: str,
    item_data: Dict
) -> str:
    """Generate explanation for individual stock or mutual fund"""
    
    # Format item details based on type
    if item_type == "stock":
        item_name = f"{item_data.get('symbol', 'N/A')} ({item_data.get('company_name', 'N/A')})"
        item_details = f"""
Stock Details:
- Symbol: {item_data.get('symbol', 'N/A')}
- Company: {item_data.get('company_name', 'N/A')}
- Match Score: {item_data.get('similarity_score', 0)*100:.1f}%
- Sector: {item_data.get('metadata', {}).get('sector', 'N/A')}
- Market Cap: {item_data.get('metadata', {}).get('market_cap', 'N/A')}
"""
    else:  # mutual_fund
        item_name = item_data.get('fund_name', 'N/A')
        item_details = f"""
Mutual Fund Details:
- Fund Name: {item_data.get('fund_name', 'N/A')}
- Match Score: {item_data.get('similarity_score', 0)*100:.1f}%
- Category: {item_data.get('metadata', {}).get('category', 'N/A')}
- AUM: ₹{item_data.get('metadata', {}).get('aum', 'N/A')} Cr
"""
    
    prompt = f"""You are a financial insights assistant inside Raj's investment app.
Your task is to explain WHY this specific {item_type.replace('_', ' ')} is recommended for this user.

Guidelines:
- Use simple, human-friendly language.
- Explain in 3-4 short bullet points.
- Focus on why THIS specific investment matches the user's profile.
- Mention the match score and what makes it a good fit.
- Never give financial advice or guaranteed returns.
- Keep it brief and focused (4-5 lines total).
- Add 1 generic caution note about market risk at the end.

User Profile:
- Age: {user_profile['age']} years
- Occupation: {user_profile['occupation']}
- Risk Profile: {user_profile['risk_label'].upper()}
- Risk Score: {user_profile['risk_score']}
- Annual Income: ₹{user_profile['income']:,.0f}
- Credit Score: {user_profile['credit_score']}
- Savings Rate: {user_profile['savings_rate']*100:.1f}%
- Debt-to-Income Ratio: {user_profile['debt_to_income']*100:.1f}%
- Portfolio Diversity Score: {user_profile['portfolio_diversity']:.0f}/100

{item_details}

Explain specifically why {item_name} is recommended for this user. Focus on the match and alignment with their profile."""

    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 300
                }
            }
            
            async with session.post(OLLAMA_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    result = await response.json()
                    explanation = result.get('response', '').strip()
                    
                    if not explanation:
                        return generate_fallback_individual_explanation(user_profile, item_type, item_data)
                    
                    return explanation
                else:
                    add_log("LLM_INDIVIDUAL", "ERROR", f"Ollama API error: {response.status}")
                    return generate_fallback_individual_explanation(user_profile, item_type, item_data)
    
    except asyncio.TimeoutError:
        add_log("LLM_INDIVIDUAL", "ERROR", "Ollama API timeout")
        return generate_fallback_individual_explanation(user_profile, item_type, item_data)
    
    except Exception as e:
        add_log("LLM_INDIVIDUAL", "ERROR", f"Ollama error: {str(e)}")
        return generate_fallback_individual_explanation(user_profile, item_type, item_data)

def generate_fallback_individual_explanation(
    user_profile: Dict,
    item_type: str,
    item_data: Dict
) -> str:
    """Generate rule-based explanation for individual item when LLM is unavailable"""
    
    risk = user_profile['risk_label'].upper()
    match_score = item_data.get('similarity_score', 0) * 100
    
    if item_type == "stock":
        item_name = f"{item_data.get('symbol', 'N/A')} ({item_data.get('company_name', 'N/A')})"
        sector = item_data.get('metadata', {}).get('sector', 'N/A')
        
        explanation = f"""**Why {item_name} is Recommended:**

• **High Match Score**: With a {match_score:.1f}% match score, this stock aligns well with your {risk} risk profile and financial goals.

• **Sector Alignment**: The {sector} sector is suitable for your investment profile and provides appropriate exposure for your risk tolerance.

• **Profile Compatibility**: This stock's characteristics match your financial metrics including income level, savings rate, and investment experience.

⚠️ **Important**: Stock market investments carry risk. Past performance is not indicative of future results."""
    
    else:  # mutual_fund
        item_name = item_data.get('fund_name', 'N/A')[:60]
        category = item_data.get('metadata', {}).get('category', 'N/A')
        
        explanation = f"""**Why {item_name} is Recommended:**

• **High Match Score**: With a {match_score:.1f}% match score, this fund aligns well with your {risk} risk profile and investment objectives.

• **Category Fit**: The {category} category is appropriate for your financial situation and provides suitable diversification.

• **Profile Compatibility**: This fund's investment strategy matches your risk tolerance, income level, and long-term financial goals.

⚠️ **Important**: Mutual fund investments are subject to market risks. Please read the offer document carefully."""
    
    return explanation



@app.on_event("startup")
async def load_all_data():
    """Load data on startup"""
    global users_df, model_data, stocks_data, funds_data, insurance_data

    try:
        # ... existing data loading code ...
        
        add_log("STARTUP", "SUCCESS", "API initialized successfully")
        
        # 🔥 YE LINE ADD KARO BRO!
        start_scheduler()  # Start the background scheduler
        
    except Exception as e:
        add_log("STARTUP", "ERROR", str(e))
    
    try:
        if os.path.exists('reco_dummy_gpt.csv'):
            users_df = pd.read_csv('reco_dummy_gpt.csv')
            add_log("STARTUP", "SUCCESS", f"Loaded {len(users_df)} users from CSV")
        else:
            add_log("STARTUP", "WARNING", "Users CSV not found")
        
        if os.path.exists('risk_model.pkl'):
            with open('risk_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            add_log("STARTUP", "SUCCESS", "Risk model loaded")
        else:
            add_log("STARTUP", "WARNING", "Risk model not found, using fallback")
        
        if os.path.exists('engineered_stocks.json'):
            with open('engineered_stocks.json', 'r') as f:
                stocks_data = json.load(f)
            add_log("STARTUP", "SUCCESS", f"Loaded {len(stocks_data)} stocks")
        else:
            add_log("STARTUP", "WARNING", "Stocks JSON not found")
        
        if os.path.exists('engineered_funds.json'):
            with open('engineered_funds.json', 'r') as f:
                funds_data = json.load(f)
            add_log("STARTUP", "SUCCESS", f"Loaded {len(funds_data)} funds")
        else:
            add_log("STARTUP", "WARNING", "Funds JSON not found")
        
        # Load Insurance Data
        if os.path.exists('engineered_insurance.json'):
            with open('engineered_insurance.json', 'r') as f:
                insurance_data = json.load(f)
            add_log("STARTUP", "SUCCESS", f"Loaded {len(insurance_data)} insurance products")
        else:
            add_log("STARTUP", "WARNING", "Insurance JSON not found")
            insurance_data = []  # Initialize as empty list if file not found
        
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
        "scheduler": {
            "enabled": scheduler_status["enabled"],
            "next_run": scheduler_status["next_run"],
            "last_run": scheduler_status["last_run"]
        },
        "endpoints": {
            "users": "/api/users",
            "user_profile": "/api/user/{user_id}",
            "recommendations": "/api/recommendations/{user_id}",
            "explain": "/api/explain",
            "explain_individual": "/api/explain-individual",
            "llm_health": "/api/llm/health",
            "admin_logs": "/api/admin/logs"
        }
    }



@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status"""
    return scheduler_status

@app.post("/api/scheduler/trigger")
async def trigger_scheduler(background_tasks: BackgroundTasks):
    """Manually trigger data refresh"""
    if scheduler_status["is_running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")
    
    background_tasks.add_task(scheduled_data_refresh)
    return {"status": "triggered", "message": "Data refresh started in background"}

@app.get("/api/users")
async def get_all_users():
    """Get list of all users"""
    if users_df is None:
        raise HTTPException(status_code=404, detail="Users data not loaded")
    
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
    
    derived_features = calculate_derived_features(user_base)
    user_complete = {**user_base, **derived_features}
    
    risk_pred = predict_user_risk(user_complete, model_data)
    user_complete['risk_label'] = risk_pred['risk_label']
    user_complete['risk_score'] = risk_pred['risk_score']
    
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
        "notes": "User vector aligned with stock, mutual fund, and insurance embeddings"
    }
    
    add_log("GET_USER", "SUCCESS", f"Retrieved profile for {user_id}")
    return result

@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: int, top_k: int = 10):
    """Get stock, mutual fund, and insurance recommendations for a user"""
    
    if stocks_data is None or funds_data is None:
        raise HTTPException(status_code=404, detail="Stocks or Funds data not loaded")
    
    user_profile = await get_user_profile(user_id)
    user_vector = user_profile['engineered_vector']
    
    # Stock Recommendations
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
    
    # Mutual Fund Recommendations
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
    
    # Insurance Recommendations (with error handling)
    insurance_recommendations = []
    if insurance_data and len(insurance_data) > 0:
        try:
            for insurance in insurance_data:
                # Get vector safely
                insurance_vector = insurance.get('engineered_vector')
                if insurance_vector is None:
                    continue
                
                similarity = cosine_similarity(user_vector, insurance_vector)
                
                # Handle different possible key names flexibly
                insurance_name = (
                    insurance.get('policy_name') or 
                    insurance.get('insurance_name') or 
                    insurance.get('name') or 
                    insurance.get('product_name') or 
                    'Unknown Insurance'
                )
                
                insurer = insurance.get('insurer', 'N/A')
                
                # Determine insurance type based on covers and features
                insurance_type = 'Health Insurance'
                if insurance.get('critical_illness_cover'):
                    insurance_type = 'Health + Critical Illness'
                
                # Extract premium info
                premium_info = insurance.get('premium_amount_range', 'Varies')
                sum_insured = insurance.get('sum_insured_range', 'Check policy')
                
                # Build metadata
                metadata = {
                    'insurer': insurer,
                    'url': insurance.get('url', ''),
                    'premium_range': premium_info,
                    'sum_insured': sum_insured,
                    'waiting_period_general': insurance.get('waiting_period_general', 'N/A'),
                    'waiting_period_preexisting': insurance.get('waiting_period_preexisting', 'N/A'),
                    'covers_preexisting': insurance.get('covers_preexisting', False),
                    'maternity_cover': insurance.get('maternity_cover', False),
                    'critical_illness_cover': insurance.get('critical_illness_cover', False),
                    'opd_cover': insurance.get('opd_cover', False),
                    'network_hospitals': insurance.get('network_hospitals', 'N/A'),
                    'no_claim_bonus': insurance.get('no_claim_bonus', 'N/A'),
                    'features': insurance.get('features', {})
                }
                
                insurance_recommendations.append({
                    'insurance_name': insurance_name,
                    'insurance_type': insurance_type,
                    'similarity_score': round(similarity, 6),
                    'metadata': metadata
                })
            
            insurance_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            insurance_recommendations = insurance_recommendations[:top_k]
            
        except Exception as e:
            add_log("GET_RECOMMENDATIONS", "WARNING", f"Error processing insurance: {str(e)}")
            insurance_recommendations = []
    
    result = {
        "user_id": user_profile['user_id'],
        "user_metadata": user_profile['metadata'],
        "top_stock_recommendations": stock_recommendations,
        "top_mutual_fund_recommendations": fund_recommendations,
        "top_insurance_recommendations": insurance_recommendations
    }
    
    add_log("GET_RECOMMENDATIONS", "SUCCESS", f"Generated recommendations for {user_id}")
    return result

@app.post("/api/explain")
async def explain_recommendations(request: ExplanationRequest):
    """Generate LLM explanation for recommendations"""
    
    try:
        add_log("EXPLAIN", "PROCESSING", f"Generating explanation for user")
        
        explanation = await generate_llm_explanation(
            user_profile=request.user_profile,
            top_stocks=request.top_stocks,
            top_mutual_funds=request.top_mutual_funds
        )
        
        add_log("EXPLAIN", "SUCCESS", "Explanation generated successfully")
        
        return {
            "explanation": explanation,
            "status": "success",
            "metadata": {
                "model": OLLAMA_MODEL,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_risk": request.user_profile['risk_label']
            }
        }
    
    except Exception as e:
        add_log("EXPLAIN", "ERROR", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/explain-individual")
async def explain_individual_recommendation(request: IndividualExplanationRequest):
    """Generate LLM explanation for individual stock or mutual fund"""
    
    try:
        add_log("EXPLAIN_INDIVIDUAL", "PROCESSING", 
                f"Generating explanation for {request.item_type}")
        
        explanation = await generate_individual_llm_explanation(
            user_profile=request.user_profile,
            item_type=request.item_type,
            item_data=request.item_data
        )
        
        add_log("EXPLAIN_INDIVIDUAL", "SUCCESS", 
                f"Explanation generated for {request.item_type}")
        
        return {
            "explanation": explanation,
            "status": "success",
            "metadata": {
                "model": OLLAMA_MODEL,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "item_type": request.item_type,
                "user_risk": request.user_profile['risk_label']
            }
        }
    
    except Exception as e:
        add_log("EXPLAIN_INDIVIDUAL", "ERROR", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/llm/health")
async def check_llm_health():
    """Check if Ollama is running and accessible"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    return {
                        "status": "online",
                        "available_models": models,
                        "configured_model": OLLAMA_MODEL
                    }
                else:
                    return {"status": "error", "message": "Ollama API not responding"}
    except Exception as e:
        return {"status": "offline", "message": str(e)}

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
        
        if os.path.exists('reco_dummy_gpt.csv'):
            users_df = pd.read_csv('reco_dummy_gpt.csv')
            add_log("REFRESH", "SUCCESS", f"Reloaded {len(users_df)} users")
        
        if os.path.exists('engineered_stocks.json'):
            with open('engineered_stocks.json', 'r') as f:
                stocks_data = json.load(f)
            add_log("REFRESH", "SUCCESS", f"Reloaded {len(stocks_data)} stocks")
        
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