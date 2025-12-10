# import pandas as pd
# import numpy as np
# import pickle
# from scipy import stats
# from ctgan import CTGAN
# from sklearn.preprocessing import QuantileTransformer, StandardScaler
# import warnings
# warnings.filterwarnings('ignore')

# # Check if models exist - LOAD if available, TRAIN if not
# import os
# MODEL_EXISTS = os.path.exists('ctgan_model.pkl')
# BEHAVIOR_EXISTS = os.path.exists('ctgan_behavior_model.pkl')

# np.random.seed(42)
# n_customers = 10000

# # Dataset 1: Customer Profile
# if not MODEL_EXISTS:
#     print("🚀 Training NEW CTGAN model...")
    
#     customer_data = pd.DataFrame({
#         'customerid': range(1, n_customers + 1),
#         'age': np.clip(stats.truncnorm.rvs(-1, 2, loc=40, scale=12, size=n_customers), 18, 70).astype(int),
#         'gender': np.random.choice(['M', 'F', 'O'], n_customers, p=[0.55, 0.44, 0.01]),
#         'citytier': np.random.choice([1, 2, 3], n_customers, p=[0.3, 0.4, 0.3]),
#     })

#     income_base = np.random.lognormal(mean=0.4, sigma=0.6, size=n_customers) * 1000000
#     income_adjust = 1 + 0.3*(customer_data['age']/40 - 1) + 0.2*(customer_data['citytier']-2)
#     customer_data['annualincome'] = np.clip(income_base * income_adjust, 300000, 6000000).astype(int)

#     customer_data['occupation'] = np.random.choice(['Salaried', 'Self-employed', 'Business'], 
#                                                   n_customers, p=[0.6, 0.25, 0.15])
#     customer_data['creditscore'] = np.clip(np.random.normal(700, 100, n_customers), 300, 850).astype(int)
#     spend_ratio = np.random.beta(0.3, 1.2, n_customers) * 0.4
#     customer_data['avgmonthlyspend'] = np.clip(
#         (customer_data['annualincome']/12 * spend_ratio).astype(int), 10000, 200000
#     )
#     customer_data['savingsrate'] = np.clip(np.random.beta(2, 5, n_customers), 0.05, 0.5)
#     customer_data['investmentamountlastyear'] = np.clip(
#         np.random.exponential(500000, n_customers) * (customer_data['annualincome']/2000000), 
#         0, 2000000
#     ).astype(int)
#     customer_data['pastinvestments'] = np.random.choice(['MF', 'Stocks', 'Insurance', 'MF_Stocks', 'None'], n_customers)
#     customer_data['riskflags'] = np.random.poisson(1.5, n_customers).clip(0, 5)

#     customer_data.to_csv('customerinfo_seed.csv', index=False)

#     numerical_cols = ['age', 'annualincome', 'creditscore', 'avgmonthlyspend', 'savingsrate', 
#                      'investmentamountlastyear', 'riskflags']
    
#     scaler = QuantileTransformer(output_distribution='normal', random_state=42)
#     customer_data[numerical_cols] = scaler.fit_transform(customer_data[numerical_cols])

#     discrete_columns = ['gender', 'citytier', 'occupation', 'pastinvestments']
#     ctgan = CTGAN(epochs=1000, batch_size=1000, discriminator_lr=1e-4, generator_lr=1e-4, verbose=True)
#     ctgan.fit(customer_data, discrete_columns=discrete_columns)

#     with open('ctgan_model.pkl', 'wb') as f:
#         pickle.dump({'model': ctgan, 'scaler': scaler, 'numerical_cols': numerical_cols}, f)
#     print("✅ CTGAN Model 1 SAVED!")
# else:
#     print("📂 Loading EXISTING CTGAN model...")
#     with open('ctgan_model.pkl', 'rb') as f:
#         model_data = pickle.load(f)
#     ctgan = model_data['model']
#     scaler = model_data['scaler']
#     numerical_cols = model_data['numerical_cols']

# # Generate Dataset 1
# synthetic_customers = ctgan.sample(10000)
# synthetic_customers[numerical_cols] = scaler.inverse_transform(synthetic_customers[numerical_cols])
# synthetic_customers['customerid'] = range(1, 10001)
# synthetic_customers.to_csv('customerinfo.csv', index=False)
# print(f"✅ Customer Info: {len(synthetic_customers)} rows, {synthetic_customers['customerid'].nunique()} unique IDs")

# # Dataset 2: Financial Behavior - FIXED VERSION
# if not BEHAVIOR_EXISTS:
#     print("🚀 Training NEW CTGAN Behavior model...")
#     customer_info = pd.read_csv('customerinfo.csv')
    
#     # 🔧 FIX: Use ALL customer IDs (no random sampling with replacement)
#     behavior_seed = pd.DataFrame({'customerid': customer_info['customerid'].values})
    
#     # Map customer data for behavior generation
#     customer_map = customer_info.set_index('customerid')
#     income_level = customer_map.loc[behavior_seed['customerid'], 'annualincome'].values / 1000000
#     credit_level = customer_map.loc[behavior_seed['customerid'], 'creditscore'].values / 850
    
#     behavior_seed['transactionvolatility'] = np.random.beta(1, 3, 10000) * (2-income_level.clip(0,2))
#     behavior_seed['spendingstabilityindex'] = np.random.beta(3, 1, 10000) * credit_level
#     behavior_seed['creditutilizationratio'] = np.random.beta(2, 3, 10000) * (1-credit_level)
#     behavior_seed['debttoincomeratio'] = np.random.beta(1.5, 4, 10000) / income_level.clip(0.1, 3)
    
#     lam_vals = 0.8 * (1 - credit_level)
#     lam_vals = np.clip(lam_vals, a_min=0, a_max=None)
#     lam_vals = np.nan_to_num(lam_vals, nan=0)
#     behavior_seed['missedpaymentcount'] = np.random.poisson(lam_vals)
    
#     behavior_seed['digitalactivityscore'] = np.random.normal(75, 15, 10000).clip(0, 100).astype(int)
#     behavior_seed['portfoliodiversityscore'] = np.random.normal(60, 20, 10000).clip(0, 100).astype(int)
    
#     num_cols2 = ['transactionvolatility', 'spendingstabilityindex', 'creditutilizationratio', 
#                 'debttoincomeratio', 'digitalactivityscore', 'portfoliodiversityscore']
#     scaler2 = StandardScaler()
#     behavior_seed[num_cols2] = scaler2.fit_transform(behavior_seed[num_cols2])
    
#     discrete_cols2 = ['missedpaymentcount']
#     ctgan_behavior = CTGAN(epochs=300, batch_size=500, verbose=True)
#     ctgan_behavior.fit(behavior_seed, discrete_cols2)
    
#     with open('ctgan_behavior_model.pkl', 'wb') as f:
#         pickle.dump({'model': ctgan_behavior, 'scaler': scaler2, 'num_cols': num_cols2}, f)
#     print("✅ CTGAN Behavior Model SAVED!")
# else:
#     print("📂 Loading EXISTING CTGAN Behavior model...")
#     with open('ctgan_behavior_model.pkl', 'rb') as f:
#         model_data2 = pickle.load(f)
#     ctgan_behavior = model_data2['model']
#     scaler2 = model_data2['scaler']
#     num_cols2 = model_data2['num_cols']

# synthetic_behavior = ctgan_behavior.sample(10000)
# synthetic_behavior[num_cols2] = scaler2.inverse_transform(synthetic_behavior[num_cols2])

# # 🔧 FIX: Ensure customer IDs match exactly
# synthetic_behavior['customerid'] = range(1, 10001)

# # 🔧 FIX: Validate no duplicates before saving
# print(f"✅ Customer Behavior: {len(synthetic_behavior)} rows, {synthetic_behavior['customerid'].nunique()} unique IDs")
# assert synthetic_behavior['customerid'].nunique() == len(synthetic_behavior), "❌ ERROR: Duplicate customer IDs detected!"

# synthetic_behavior.to_csv('customerbehavior.csv', index=False)

# # Dataset 3: Products (unchanged)
# products = []
# for i in range(60):
#     risk_map = {'Low': 0.3, 'Medium': 0.5, 'High': 0.7}
#     risklevel = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.4, 0.3])
#     base_return = np.random.normal(12, 3, 1)[0]
#     products.append({
#         'productid': f'MF_{i+1:03d}',
#         'category': np.random.choice(['Equity', 'Debt', 'Balanced'], p=[0.4, 0.35, 0.25]),
#         'risklevel': risklevel,
#         'returnrate': round(base_return + risk_map[risklevel]*5, 2),
#         'mininvestment': np.random.choice([5000, 10000, 25000, 50000], p=[0.2, 0.4, 0.3, 0.1])
#     })

# sectors = ['IT', 'Pharma', 'Banking', 'Auto', 'FMCG', 'Energy']
# for i in range(60):
#     products.append({
#         'productid': f'ST_{i+1:03d}',
#         'sector': np.random.choice(sectors),
#         'volatility': round(np.random.uniform(0.15, 0.45, 1)[0], 3),
#         'avg1yrreturn': round(np.random.normal(15, 8, 1)[0], 2)
#     })

# ins_types = ['Term', 'Health', 'Life']
# for i in range(60):
#     products.append({
#         'productid': f'INS_{i+1:03d}',
#         'type': np.random.choice(ins_types, p=[0.4, 0.3, 0.3]),
#         'premium': round(np.random.uniform(12000, 80000, 1)[0], 0),
#         'sumassured': round(np.random.uniform(500000, 5000000, 1)[0], 0),
#         'targetage': np.random.choice([25, 30, 35, 45], p=[0.2, 0.3, 0.3, 0.2]),
#         'targetincome': np.random.choice([500000, 1000000, 2000000, 5000000], p=[0.3, 0.4, 0.2, 0.1])
#     })

# pd.DataFrame(products).to_csv('products.csv', index=False)

# print("\n" + "="*50)
# print("✅ ALL DATASETS GENERATED SUCCESSFULLY!")
# print("="*50)
# print(f"📊 Customer Info: {synthetic_customers.shape[0]} unique customers")
# print(f"📊 Customer Behavior: {synthetic_behavior.shape[0]} unique customers")
# print(f"📊 Products: {len(products)} products (60 MF + 60 Stocks + 60 Insurance)")
# print("\n✅ No duplicate customer IDs - Ready for model training!")




import pandas as pd
import numpy as np
import pickle
from scipy import stats
from ctgan import CTGAN
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import warnings
warnings.filterwarnings('ignore')

import os

# -----------------------------
# PROJECT-STRUCTURE PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project/
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH_1 = os.path.join(MODEL_DIR, "ctgan_model.pkl")
MODEL_PATH_2 = os.path.join(MODEL_DIR, "ctgan_behavior_model.pkl")
CUSTOMER_INFO_CSV = os.path.join(DATA_DIR, "customer_info.csv")
CUSTOMER_BEHAVIOR_CSV = os.path.join(DATA_DIR, "customer_behavior.csv")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")


# -----------------------------
# MODEL EXIST CHECK
# -----------------------------
MODEL_EXISTS = os.path.exists(MODEL_PATH_1)
BEHAVIOR_EXISTS = os.path.exists(MODEL_PATH_2)

np.random.seed(42)
n_customers = 10000

# ============================================
# DATASET 1: CUSTOMER PROFILE
# ============================================

if not MODEL_EXISTS:
    print("🚀 Training NEW CTGAN model...")

    customer_data = pd.DataFrame({
        'customerid': range(1, n_customers + 1),
        'age': np.clip(stats.truncnorm.rvs(-1, 2, loc=40, scale=12, size=n_customers), 18, 70).astype(int),
        'gender': np.random.choice(['M', 'F', 'O'], n_customers, p=[0.55, 0.44, 0.01]),
        'citytier': np.random.choice([1, 2, 3], n_customers, p=[0.3, 0.4, 0.3]),
    })

    income_base = np.random.lognormal(mean=0.4, sigma=0.6, size=n_customers) * 1000000
    income_adjust = 1 + 0.3*(customer_data['age']/40 - 1) + 0.2*(customer_data['citytier']-2)
    customer_data['annualincome'] = np.clip(income_base * income_adjust, 300000, 6000000).astype(int)

    customer_data['occupation'] = np.random.choice(['Salaried', 'Self-employed', 'Business'],
                                                   n_customers, p=[0.6, 0.25, 0.15])
    customer_data['creditscore'] = np.clip(np.random.normal(700, 100, n_customers), 300, 850).astype(int)

    spend_ratio = np.random.beta(0.3, 1.2, n_customers) * 0.4
    customer_data['avgmonthlyspend'] = np.clip(
        (customer_data['annualincome']/12 * spend_ratio).astype(int), 10000, 200000
    )

    customer_data['savingsrate'] = np.clip(np.random.beta(2, 5, n_customers), 0.05, 0.5)

    customer_data['investmentamountlastyear'] = np.clip(
        np.random.exponential(500000, n_customers) * (customer_data['annualincome']/2000000),
        0, 2000000
    ).astype(int)

    customer_data['pastinvestments'] = np.random.choice(
        ['MF', 'Stocks', 'Insurance', 'MF_Stocks', 'None'], n_customers
    )

    customer_data['riskflags'] = np.random.poisson(1.5, n_customers).clip(0, 5)

    # Save seed (optional)
    customer_data.to_csv(os.path.join(DATA_DIR, 'customerinfo_seed.csv'), index=False)

    numerical_cols = ['age', 'annualincome', 'creditscore', 'avgmonthlyspend',
                      'savingsrate', 'investmentamountlastyear', 'riskflags']

    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    customer_data[numerical_cols] = scaler.fit_transform(customer_data[numerical_cols])

    discrete_columns = ['gender', 'citytier', 'occupation', 'pastinvestments']
    ctgan = CTGAN(epochs=1000, batch_size=1000, discriminator_lr=1e-4, generator_lr=1e-4, verbose=True)
    ctgan.fit(customer_data, discrete_columns=discrete_columns)

    with open(MODEL_PATH_1, 'wb') as f:
        pickle.dump({'model': ctgan, 'scaler': scaler, 'numerical_cols': numerical_cols}, f)

    print("✅ CTGAN Model 1 SAVED!")

else:
    print("📂 Loading EXISTING CTGAN model...")
    with open(MODEL_PATH_1, 'rb') as f:
        model_data = pickle.load(f)
    ctgan = model_data['model']
    scaler = model_data['scaler']
    numerical_cols = model_data['numerical_cols']

# Generate Synthetic Customer Info
synthetic_customers = ctgan.sample(10000)
synthetic_customers[numerical_cols] = scaler.inverse_transform(synthetic_customers[numerical_cols])
synthetic_customers['customerid'] = range(1, 10001)

synthetic_customers.to_csv(CUSTOMER_INFO_CSV, index=False)
print(f"✅ Customer Info: {len(synthetic_customers)} rows, {synthetic_customers['customerid'].nunique()} unique IDs")


# ============================================
# DATASET 2: FINANCIAL BEHAVIOR
# ============================================

if not BEHAVIOR_EXISTS:
    print("🚀 Training NEW CTGAN Behavior model...")

    customer_info = pd.read_csv(CUSTOMER_INFO_CSV)

    behavior_seed = pd.DataFrame({'customerid': customer_info['customerid'].values})

    customer_map = customer_info.set_index('customerid')

    income_level = customer_map.loc[behavior_seed['customerid'], 'annualincome'].values / 1000000
    credit_level = customer_map.loc[behavior_seed['customerid'], 'creditscore'].values / 850

    behavior_seed['transactionvolatility'] = np.random.beta(1, 3, 10000) * (2 - income_level.clip(0, 2))
    behavior_seed['spendingstabilityindex'] = np.random.beta(3, 1, 10000) * credit_level
    behavior_seed['creditutilizationratio'] = np.random.beta(2, 3, 10000) * (1 - credit_level)
    behavior_seed['debttoincomeratio'] = np.random.beta(1.5, 4, 10000) / income_level.clip(0.1, 3)

    lam_vals = 0.8 * (1 - credit_level)
    lam_vals = np.clip(lam_vals, a_min=0, a_max=None)
    lam_vals = np.nan_to_num(lam_vals, nan=0)
    behavior_seed['missedpaymentcount'] = np.random.poisson(lam_vals)

    behavior_seed['digitalactivityscore'] = np.random.normal(75, 15, 10000).clip(0, 100).astype(int)
    behavior_seed['portfoliodiversityscore'] = np.random.normal(60, 20, 10000).clip(0, 100).astype(int)

    num_cols2 = ['transactionvolatility', 'spendingstabilityindex',
                 'creditutilizationratio', 'debttoincomeratio',
                 'digitalactivityscore', 'portfoliodiversityscore']

    scaler2 = StandardScaler()
    behavior_seed[num_cols2] = scaler2.fit_transform(behavior_seed[num_cols2])

    discrete_cols2 = ['missedpaymentcount']
    ctgan_behavior = CTGAN(epochs=300, batch_size=500, verbose=True)
    ctgan_behavior.fit(behavior_seed, discrete_cols2)

    with open(MODEL_PATH_2, 'wb') as f:
        pickle.dump({'model': ctgan_behavior, 'scaler': scaler2, 'num_cols': num_cols2}, f)

    print("✅ CTGAN Behavior Model SAVED!")

else:
    print("📂 Loading EXISTING CTGAN Behavior model...")
    with open(MODEL_PATH_2, 'rb') as f:
        model_data2 = pickle.load(f)
    ctgan_behavior = model_data2['model']
    scaler2 = model_data2['scaler']
    num_cols2 = model_data2['num_cols']


# Generate Synthetic Behavior
synthetic_behavior = ctgan_behavior.sample(10000)
synthetic_behavior[num_cols2] = scaler2.inverse_transform(synthetic_behavior[num_cols2])

synthetic_behavior['customerid'] = range(1, 10001)

# Duplicate check
print(f"✅ Customer Behavior: {len(synthetic_behavior)} rows, {synthetic_behavior['customerid'].nunique()} unique IDs")
assert synthetic_behavior['customerid'].nunique() == len(synthetic_behavior), "❌ ERROR: Duplicate customer IDs detected!"

synthetic_behavior.to_csv(CUSTOMER_BEHAVIOR_CSV, index=False)


# ============================================
# DATASET 3: PRODUCTS
# ============================================

products = []

# Mutual Funds
for i in range(60):
    risk_map = {'Low': 0.3, 'Medium': 0.5, 'High': 0.7}
    risklevel = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.4, 0.3])
    base_return = np.random.normal(12, 3, 1)[0]
    products.append({
        'productid': f'MF_{i+1:03d}',
        'category': np.random.choice(['Equity', 'Debt', 'Balanced'], p=[0.4, 0.35, 0.25]),
        'risklevel': risklevel,
        'returnrate': round(base_return + risk_map[risklevel] * 5, 2),
        'mininvestment': np.random.choice([5000, 10000, 25000, 50000], p=[0.2, 0.4, 0.3, 0.1])
    })

sectors = ['IT', 'Pharma', 'Banking', 'Auto', 'FMCG', 'Energy']
for i in range(60):
    products.append({
        'productid': f'ST_{i+1:03d}',
        'sector': np.random.choice(sectors),
        'volatility': round(np.random.uniform(0.15, 0.45, 1)[0], 3),
        'avg1yrreturn': round(np.random.normal(15, 8, 1)[0], 2)
    })

ins_types = ['Term', 'Health', 'Life']
for i in range(60):
    products.append({
        'productid': f'INS_{i+1:03d}',
        'type': np.random.choice(ins_types, p=[0.4, 0.3, 0.3]),
        'premium': round(np.random.uniform(12000, 80000, 1)[0], 0),
        'sumassured': round(np.random.uniform(500000, 5000000, 1)[0], 0),
        'targetage': np.random.choice([25, 30, 35, 45], p=[0.2, 0.3, 0.3, 0.2]),
        'targetincome': np.random.choice([500000, 1000000, 2000000, 5000000], p=[0.3, 0.4, 0.2, 0.1])
    })

pd.DataFrame(products).to_csv(PRODUCTS_CSV, index=False)

print("\n" + "="*60)
print("✅ ALL DATASETS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"📊 Customer Info: {synthetic_customers.shape[0]} unique customers")
print(f"📊 Customer Behavior: {synthetic_behavior.shape[0]} unique customers")
print(f"📊 Products: {len(products)} products")
print("\n🚀 Ready for model training!")
