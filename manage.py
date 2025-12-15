import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class InsuranceRecommendationSystem:
    def __init__(self):
        self.risk_model = None
        self.recommendation_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def clean_numeric_column(self, series):
        """Clean numeric columns that may have strings like '6+'"""
        def clean_value(val):
            if pd.isna(val):
                return 0
            val_str = str(val).strip()
            # Handle cases like '6+', '5+', etc.
            if '+' in val_str:
                return int(val_str.replace('+', ''))
            try:
                return int(float(val_str))
            except:
                return 0
        return series.apply(clean_value)
    
    def load_data(self, csv_path):
        """Load customer data from CSV"""
        print("Loading customer data...")
        df = pd.read_csv(csv_path)
        
        # Clean numeric columns
        numeric_cols = ['familysize', 'numchildren', 'numadults', 'numelders', 
                       'age', 'creditscore', 'annualincome', 'avgmonthlyspend', 
                       'savingsrate', 'investmentamountlastyear']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = self.clean_numeric_column(df[col])
        
        # Handle citytier
        if 'citytier' in df.columns:
            df['citytier'] = self.clean_numeric_column(df['citytier'])
        
        print(f"Loaded {len(df)} customer records")
        print(f"Columns: {list(df.columns)}")
        return df
    
    def calculate_risk_score(self, df):
        """Calculate risk score using XGBoost"""
        print("\n=== Step 1: Calculating Risk Scores with XGBoost ===")
        
        # Create comprehensive risk labels based on customer profile
        df['risk_score'] = 0
        
        # Credit score factor (highest weight - financial stability)
        df.loc[df['creditscore'] < 550, 'risk_score'] += 50
        df.loc[(df['creditscore'] >= 550) & (df['creditscore'] < 650), 'risk_score'] += 35
        df.loc[(df['creditscore'] >= 650) & (df['creditscore'] < 700), 'risk_score'] += 25
        df.loc[(df['creditscore'] >= 700) & (df['creditscore'] < 750), 'risk_score'] += 15
        df.loc[df['creditscore'] >= 750, 'risk_score'] += 5
        
        # Age-based health risk
        df.loc[df['age'] < 25, 'risk_score'] += 12
        df.loc[(df['age'] >= 25) & (df['age'] < 35), 'risk_score'] += 8
        df.loc[(df['age'] >= 35) & (df['age'] < 45), 'risk_score'] += 5
        df.loc[(df['age'] >= 45) & (df['age'] < 55), 'risk_score'] += 15
        df.loc[(df['age'] >= 55) & (df['age'] < 65), 'risk_score'] += 25
        df.loc[df['age'] >= 65, 'risk_score'] += 35
        
        # Income stability and affordability
        df.loc[df['annualincome'] < 250000, 'risk_score'] += 30
        df.loc[(df['annualincome'] >= 250000) & (df['annualincome'] < 500000), 'risk_score'] += 20
        df.loc[(df['annualincome'] >= 500000) & (df['annualincome'] < 800000), 'risk_score'] += 10
        df.loc[(df['annualincome'] >= 800000) & (df['annualincome'] < 1200000), 'risk_score'] += 5
        df.loc[df['annualincome'] >= 1200000, 'risk_score'] += 3
        
        # Savings rate (financial discipline)
        df.loc[df['savingsrate'] < 0.03, 'risk_score'] += 25
        df.loc[(df['savingsrate'] >= 0.03) & (df['savingsrate'] < 0.08), 'risk_score'] += 15
        df.loc[(df['savingsrate'] >= 0.08) & (df['savingsrate'] < 0.15), 'risk_score'] += 8
        df.loc[df['savingsrate'] >= 0.15, 'risk_score'] += 3
        
        # Family size and dependents (coverage needs)
        df.loc[df['familysize'] >= 6, 'risk_score'] += 15
        df.loc[(df['familysize'] >= 4) & (df['familysize'] < 6), 'risk_score'] += 10
        df.loc[df['familysize'] == 3, 'risk_score'] += 5
        
        # Elderly dependents (higher medical needs)
        df.loc[df['numelders'] >= 2, 'risk_score'] += 25
        df.loc[df['numelders'] == 1, 'risk_score'] += 15
        
        # Children (long-term coverage need)
        df.loc[df['numchildren'] >= 3, 'risk_score'] += 12
        df.loc[df['numchildren'] == 2, 'risk_score'] += 8
        df.loc[df['numchildren'] == 1, 'risk_score'] += 5
        
        # Spending pattern (financial stress indicator)
        df['spending_ratio'] = df['avgmonthlyspend'] / (df['annualincome'] / 12 + 1)
        df.loc[df['spending_ratio'] > 0.9, 'risk_score'] += 30
        df.loc[(df['spending_ratio'] > 0.7) & (df['spending_ratio'] <= 0.9), 'risk_score'] += 20
        df.loc[(df['spending_ratio'] > 0.5) & (df['spending_ratio'] <= 0.7), 'risk_score'] += 10
        
        # Investment history (financial awareness)
        df.loc[df['investmentamountlastyear'] == 0, 'risk_score'] += 15
        df.loc[(df['investmentamountlastyear'] > 0) & (df['investmentamountlastyear'] < 10000), 'risk_score'] += 10
        
        # City tier (access to healthcare)
        if 'citytier' in df.columns:
            df.loc[df['citytier'] == 3, 'risk_score'] += 10
            df.loc[df['citytier'] == 2, 'risk_score'] += 5
        
        # Occupation risk
        high_risk_occupations = ['Business', 'Self-employed']
        if 'occupation' in df.columns:
            df.loc[df['occupation'].isin(high_risk_occupations), 'risk_score'] += 12
        
        print(f"Risk Score Range: {df['risk_score'].min():.2f} - {df['risk_score'].max():.2f}")
        print(f"Average Risk Score: {df['risk_score'].mean():.2f}")
        
        # Categorize risk
        df['risk_category'] = pd.cut(df['risk_score'], 
                                     bins=[0, 40, 80, 120, 300],
                                     labels=['Low', 'Medium', 'High', 'Very High'])
        
        print("\nRisk Distribution:")
        print(df['risk_category'].value_counts())
        
        # Prepare features for XGBoost
        feature_cols = ['age', 'annualincome', 'creditscore', 'avgmonthlyspend', 
                       'savingsrate', 'investmentamountlastyear', 'familysize', 
                       'numchildren', 'numadults', 'numelders', 'spending_ratio']
        
        if 'citytier' in df.columns:
            feature_cols.append('citytier')
        
        # Encode categorical variables
        categorical_cols = ['gender', 'occupation', 'pastinvestments']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                feature_cols.append(f'{col}_encoded')
        
        X = df[feature_cols].fillna(0)
        y = df['risk_score']
        
        # Split data with stratification for better distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost model with better hyperparameters
        print("\nTraining XGBoost Risk Model...")
        self.risk_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            objective='reg:squarederror'
        )
        self.risk_model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred_train = self.risk_model.predict(X_train)
        y_pred_test = self.risk_model.predict(X_test)
        
        train_mse = np.mean((y_train - y_pred_train) ** 2)
        test_mse = np.mean((y_test - y_pred_test) ** 2)
        train_r2 = 1 - (train_mse / np.var(y_train))
        test_r2 = 1 - (test_mse / np.var(y_test))
        
        print(f"✓ Training MSE: {train_mse:.2f}, R²: {train_r2:.4f}")
        print(f"✓ Testing MSE: {test_mse:.2f}, R²: {test_r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.risk_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Important Features:")
        print(feature_importance.head())
        
        # Save model
        with open('risk_model_xgboost.pkl', 'wb') as f:
            pickle.dump(self.risk_model, f)
        print("\n✓ Risk model saved as 'risk_model_xgboost.pkl'")
        
        # Add predicted risk to dataframe
        df['predicted_risk'] = self.risk_model.predict(X)
        
        return df
    
    def match_insurance_products(self, df, insurance_json_path):
        """Match customers to insurance products and predict purchase probability"""
        print("\n=== Step 2: Matching Insurance Products with Logistic Regression ===")
        
        # Load insurance product data
        with open(insurance_json_path, 'r') as f:
            insurance_data_raw = json.load(f)
        
        # Handle both single product (dict) and multiple products (list)
        if isinstance(insurance_data_raw, list):
            insurance_data = insurance_data_raw[0]  # Use first product for now
            print(f"Loaded {len(insurance_data_raw)} insurance products. Using: {insurance_data['policy_name']}")
        else:
            insurance_data = insurance_data_raw
            print(f"Loaded insurance product: {insurance_data['policy_name']}")
        
        # Create realistic purchase behavior labels
        purchase_score = pd.Series(0, index=df.index)
        
        # Past insurance experience (strongest indicator)
        purchase_score += (df['pastinvestments'] == 'insurance').astype(int) * 35
        
        # Credit score (ability to pay)
        purchase_score += ((df['creditscore'] > 700).astype(int) * 25)
        purchase_score += ((df['creditscore'] > 650) & (df['creditscore'] <= 700)).astype(int) * 15
        
        # Family protection needs
        purchase_score += ((df['numchildren'] > 0) | (df['numelders'] > 0)).astype(int) * 20
        purchase_score += (df['familysize'] >= 4).astype(int) * 15
        
        # Income and affordability
        purchase_score += (df['annualincome'] > 600000).astype(int) * 20
        purchase_score += ((df['annualincome'] > 400000) & (df['annualincome'] <= 600000)).astype(int) * 12
        
        # Savings rate (financial discipline)
        purchase_score += (df['savingsrate'] > 0.1).astype(int) * 15
        purchase_score += ((df['savingsrate'] > 0.05) & (df['savingsrate'] <= 0.1)).astype(int) * 8
        
        # Age factor (awareness and need)
        purchase_score += ((df['age'] >= 30) & (df['age'] <= 50)).astype(int) * 15
        
        # Investment history (financially aware)
        purchase_score += (df['investmentamountlastyear'] > 10000).astype(int) * 10
        
        # City tier (awareness and access)
        if 'citytier' in df.columns:
            purchase_score += (df['citytier'] == 1).astype(int) * 10
        
        # Risk awareness (medium-high risk more likely to buy)
        purchase_score += ((df['predicted_risk'] > 50) & (df['predicted_risk'] < 100)).astype(int) * 12
        
        # Create labels with some randomness for realism
        np.random.seed(42)
        noise = np.random.randint(-10, 10, size=len(df))
        purchase_score += noise
        
        # Binary labels
        labels = (purchase_score > 65).astype(int)
        
        print(f"Purchase Distribution: {labels.sum()} likely buyers out of {len(labels)} ({labels.mean()*100:.1f}%)")
        
        # Prepare features for logistic regression
        feature_cols = ['age', 'annualincome', 'creditscore', 'predicted_risk',
                       'familysize', 'numchildren', 'numelders', 'savingsrate',
                       'investmentamountlastyear', 'avgmonthlyspend', 'spending_ratio']
        
        if 'citytier' in df.columns:
            feature_cols.append('citytier')
        
        # Add encoded features
        for col in ['gender', 'occupation', 'pastinvestments']:
            if f'{col}_encoded' in df.columns:
                feature_cols.append(f'{col}_encoded')
        
        X = df[feature_cols].fillna(0)
        y = labels
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Logistic Regression with better parameters
        print("\nTraining Logistic Regression Model...")
        self.recommendation_model = LogisticRegression(
            max_iter=2000,
            C=0.5,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        )
        self.recommendation_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.recommendation_model.predict(X_train)
        y_pred_test = self.recommendation_model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"✓ Training Accuracy: {train_accuracy:.2%}")
        print(f"✓ Testing Accuracy: {test_accuracy:.2%}")
        
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred_test, target_names=['Not Buy', 'Will Buy']))
        
        # Save models
        with open('recommendation_model_logistic.pkl', 'wb') as f:
            pickle.dump(self.recommendation_model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print("\n✓ Recommendation model saved as 'recommendation_model_logistic.pkl'")
        
        # Predict purchase probability for all customers
        purchase_probabilities = self.recommendation_model.predict_proba(X_scaled)[:, 1]
        df['purchase_probability'] = purchase_probabilities
        df['predicted_purchase'] = self.recommendation_model.predict(X_scaled)
        
        return df, insurance_data
    
    def generate_recommendations(self, df, insurance_data, top_n=10):
        """Generate top N recommendations"""
        print(f"\n=== Step 3: Generating Top {top_n} Recommendations ===\n")
        
        # Sort by purchase probability
        df_sorted = df.sort_values('purchase_probability', ascending=False)
        top_customers = df_sorted.head(top_n)
        
        recommendations = []
        
        print("=" * 100)
        for idx, customer in top_customers.iterrows():
            recommendation = {
                'customer_id': int(idx),
                'age': int(customer['age']),
                'gender': str(customer['gender']),
                'annual_income': int(customer['annualincome']),
                'credit_score': int(customer['creditscore']),
                'family_size': int(customer['familysize']),
                'num_children': int(customer['numchildren']),
                'num_elders': int(customer['numelders']),
                'savings_rate': float(customer['savingsrate']),
                'risk_score': float(customer['predicted_risk']),
                'risk_category': self.categorize_risk(customer['predicted_risk']),
                'recommended_insurance': insurance_data['policy_name'],
                'insurer': insurance_data['insurer'],
                'purchase_probability': float(customer['purchase_probability']),
                'predicted_purchase': 'Yes' if customer['predicted_purchase'] == 1 else 'No',
                'confidence': self.get_confidence(customer['purchase_probability']),
                'reason': self.generate_reason(customer, insurance_data)
            }
            recommendations.append(recommendation)
            
            # Print recommendation
            print(f"🎯 CUSTOMER #{recommendation['customer_id']}")
            print(f"   Profile: {recommendation['age']}Y {recommendation['gender']}, Family: {recommendation['family_size']} members")
            print(f"   Income: ₹{recommendation['annual_income']:,}/year, Credit: {recommendation['credit_score']}")
            print(f"   Children: {recommendation['num_children']}, Elders: {recommendation['num_elders']}, Savings: {recommendation['savings_rate']*100:.1f}%")
            print(f"   Risk: {recommendation['risk_score']:.1f} ({recommendation['risk_category']})")
            print(f"   ✓ RECOMMENDED: {recommendation['recommended_insurance']} by {recommendation['insurer']}")
            print(f"   ✓ PURCHASE PROBABILITY: {recommendation['purchase_probability']*100:.1f}%")
            print(f"   ✓ PREDICTION: {recommendation['predicted_purchase']} (Confidence: {recommendation['confidence']})")
            print(f"   📋 Reason: {recommendation['reason']}")
            print("-" * 100)
        
        # Save recommendations to JSON
        with open('recommendations_output.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        print("\n✓ Recommendations saved to 'recommendations_output.json'")
        
        # Summary statistics
        print("\n" + "=" * 100)
        print("📊 SUMMARY STATISTICS")
        print("=" * 100)
        print(f"Total Customers Analyzed: {len(df)}")
        print(f"High Probability Customers (>70%): {len(df[df['purchase_probability'] > 0.7])}")
        print(f"Medium Probability Customers (50-70%): {len(df[(df['purchase_probability'] > 0.5) & (df['purchase_probability'] <= 0.7)])}")
        print(f"Average Purchase Probability: {df['purchase_probability'].mean()*100:.1f}%")
        print(f"Average Risk Score: {df['predicted_risk'].mean():.1f}")
        
        return recommendations
    
    def categorize_risk(self, risk_score):
        """Categorize risk score"""
        if risk_score < 40:
            return 'Low'
        elif risk_score < 80:
            return 'Medium'
        elif risk_score < 120:
            return 'High'
        else:
            return 'Very High'
    
    def get_confidence(self, probability):
        """Get confidence level"""
        if probability > 0.8:
            return 'Very High'
        elif probability > 0.7:
            return 'High'
        elif probability > 0.5:
            return 'Medium'
        else:
            return 'Low'
    
    def generate_reason(self, customer, insurance_data):
        """Generate personalized reason for recommendation"""
        reasons = []
        
        if customer['numchildren'] > 0:
            reasons.append(f"{int(customer['numchildren'])} children need protection")
        
        if customer['numelders'] > 0:
            reasons.append(f"{int(customer['numelders'])} elderly dependents")
        
        if customer['pastinvestments'] == 'insurance':
            reasons.append("Previous insurance buyer")
        
        if customer['creditscore'] > 700:
            reasons.append("Excellent credit")
        elif customer['creditscore'] > 650:
            reasons.append("Good credit")
        
        if customer['annualincome'] > 800000:
            reasons.append("High income stability")
        elif customer['annualincome'] > 500000:
            reasons.append("Good income")
        
        if customer['savingsrate'] > 0.12:
            reasons.append("Strong savings habit")
        elif customer['savingsrate'] > 0.08:
            reasons.append("Good savings")
        
        if customer['familysize'] >= 4:
            reasons.append("Large family coverage needed")
        
        # Product-specific reasons
        if insurance_data.get('maternity_cover') and customer['age'] < 40 and customer['gender'] == 'F':
            reasons.append("Maternity benefits available")
        
        if insurance_data.get('critical_illness_cover') and customer['age'] > 40:
            reasons.append("Critical illness coverage")
        
        return "; ".join(reasons) if reasons else "Comprehensive health coverage recommended"
    
    def run_complete_pipeline(self, csv_path, insurance_json_path, top_n=10):
        """Run the complete recommendation pipeline"""
        print("\n" + "=" * 100)
        print(" " * 25 + "INSURANCE RECOMMENDATION SYSTEM")
        print(" " * 30 + "AI-Powered Risk & Purchase Prediction")
        print("=" * 100 + "\n")
        
        # Step 1: Load data and calculate risk
        df = self.load_data(csv_path)
        df = self.calculate_risk_score(df)
        
        # Step 2: Match products and predict purchase probability
        df, insurance_data = self.match_insurance_products(df, insurance_json_path)
        
        # Step 3: Generate recommendations
        recommendations = self.generate_recommendations(df, insurance_data, top_n)
        
        print("\n" + "=" * 100)
        print(" " * 35 + "✓ PIPELINE COMPLETED!")
        print("=" * 100)
        print("\n📦 Models Saved:")
        print("   • risk_model_xgboost.pkl")
        print("   • recommendation_model_logistic.pkl")
        print("   • scaler.pkl")
        print("   • label_encoders.pkl")
        print("\n📄 Output Files:")
        print("   • recommendations_output.json")
        print("\n" + "=" * 100)
        
        return recommendations


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    # Initialize the system
    system = InsuranceRecommendationSystem()
    
    # Run the complete pipeline
    recommendations = system.run_complete_pipeline(
        csv_path='customer_info.csv',
        insurance_json_path='data/health_policies.json',
        top_n=10
    )
    
    print("\n✅ All Done! Check 'recommendations_output.json' for complete results.")