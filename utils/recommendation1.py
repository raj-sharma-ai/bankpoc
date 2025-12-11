# import json
# import numpy as np
# from typing import List, Dict, Tuple

# def load_json(file_path):
#     """Load JSON data from file"""
#     with open(file_path, 'r') as f:
#         return json.load(f)

# def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
#     """
#     Calculate cosine similarity between two vectors
#     cosine = dot(u, v) / (||u|| * ||v||)
#     """
#     # Convert to numpy arrays
#     u = np.array(vec1)
#     v = np.array(vec2)
    
#     # Calculate dot product
#     dot_product = np.dot(u, v)
    
#     # Calculate magnitudes
#     norm_u = np.linalg.norm(u)
#     norm_v = np.linalg.norm(v)
    
#     # Handle zero vectors
#     if norm_u == 0 or norm_v == 0:
#         return 0.0
    
#     # Calculate cosine similarity
#     similarity = dot_product / (norm_u * norm_v)
    
#     return float(similarity)

# def get_stock_recommendations(user_vector: List[float], 
#                               stocks: List[Dict], 
#                               top_k: int = 10) -> List[Dict]:
#     """
#     Calculate cosine similarity between user and all stocks
#     Return top K recommendations
#     """
#     recommendations = []
    
#     for stock in stocks:
#         stock_vector = stock['engineered_vector']
#         similarity = cosine_similarity(user_vector, stock_vector)
        
#         recommendations.append({
#             'symbol': stock['symbol'],
#             'company_name': stock.get('company_name', ''),
#             'similarity_score': round(similarity, 6),
#             'metadata': stock.get('metadata', {})
#         })
    
#     # Sort by similarity score in descending order
#     recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
    
#     # Return top K
#     return recommendations[:top_k]

# def get_mutual_fund_recommendations(user_vector: List[float], 
#                                    mutual_funds: List[Dict], 
#                                    top_k: int = 10) -> List[Dict]:
#     """
#     Calculate cosine similarity between user and all mutual funds
#     Return top K recommendations
#     """
#     recommendations = []
    
#     for fund in mutual_funds:
#         fund_vector = fund['engineered_vector']
#         similarity = cosine_similarity(user_vector, fund_vector)
        
#         recommendations.append({
#             'fund_name': fund['fund_name'],
#             'fund_link': fund.get('fund_link', ''),
#             'similarity_score': round(similarity, 6),
#             'metadata': fund.get('metadata', {})
#         })
    
#     # Sort by similarity score in descending order
#     recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
    
#     # Return top K
#     return recommendations[:top_k]

# def generate_recommendations(user_json_path: str,
#                             stocks_json_path: str,
#                             mutual_funds_json_path: str,
#                             output_path: str = 'recommendations.json',
#                             top_k: int = 10) -> Dict:
#     """
#     Main recommendation engine
#     """
#     print("="*60)
#     print("FINANCIAL RECOMMENDATION ENGINE")
#     print("="*60)
    
#     try:
#         # Load user data
#         print(f"\n1. Loading user data from: {user_json_path}")
#         users_data = load_json(user_json_path)
        
#         # Handle single user or list of users
#         if isinstance(users_data, list):
#             if len(users_data) == 0:
#                 raise ValueError("No users found in user JSON")
#             user = users_data[0]  # Take first user
#             print(f"   ✓ Loaded user: {user['user_id']}")
#         else:
#             user = users_data
#             print(f"   ✓ Loaded user: {user['user_id']}")
        
#         # Extract user vector
#         user_vector = user['engineered_vector']
#         print(f"   User vector: {user_vector}")
        
#         # Load stocks
#         print(f"\n2. Loading stocks from: {stocks_json_path}")
#         stocks = load_json(stocks_json_path)
#         print(f"   ✓ Loaded {len(stocks)} stocks")
        
#         # Load mutual funds
#         print(f"\n3. Loading mutual funds from: {mutual_funds_json_path}")
#         mutual_funds = load_json(mutual_funds_json_path)
#         print(f"   ✓ Loaded {len(mutual_funds)} mutual funds")
        
#         # Calculate stock recommendations
#         print(f"\n4. Calculating stock recommendations...")
#         stock_recommendations = get_stock_recommendations(
#             user_vector, stocks, top_k
#         )
#         print(f"   ✓ Found top {len(stock_recommendations)} stock matches")
        
#         # Calculate mutual fund recommendations
#         print(f"\n5. Calculating mutual fund recommendations...")
#         fund_recommendations = get_mutual_fund_recommendations(
#             user_vector, mutual_funds, top_k
#         )
#         print(f"   ✓ Found top {len(fund_recommendations)} mutual fund matches")
        
#         # Create final output
#         result = {
#             "user_id": user['user_id'],
#             "user_metadata": user.get('metadata', {}),
#             "top_stock_recommendations": stock_recommendations,
#             "top_mutual_fund_recommendations": fund_recommendations
#         }
        
#         # Save to file
#         print(f"\n6. Saving recommendations to: {output_path}")
#         with open(output_path, 'w') as f:
#             json.dump(result, f, indent=2)
        
#         print(f"   ✓ Recommendations saved successfully")
        
#         # Print summary
#         print("\n" + "="*60)
#         print("RECOMMENDATION SUMMARY")
#         print("="*60)
#         print(f"User ID: {result['user_id']}")
        
#         print(f"\n📈 TOP {len(stock_recommendations)} STOCK RECOMMENDATIONS:")
#         for i, stock in enumerate(stock_recommendations[:5], 1):
#             print(f"{i}. {stock['symbol']} - {stock['company_name']}")
#             print(f"   Similarity: {stock['similarity_score']:.4f}")
#             print(f"   Sector: {stock['metadata'].get('sector', 'N/A')}")
        
#         if len(stock_recommendations) > 5:
#             print(f"   ... and {len(stock_recommendations) - 5} more")
        
#         print(f"\n📊 TOP {len(fund_recommendations)} MUTUAL FUND RECOMMENDATIONS:")
#         for i, fund in enumerate(fund_recommendations[:5], 1):
#             print(f"{i}. {fund['fund_name']}")
#             print(f"   Similarity: {fund['similarity_score']:.4f}")
#             print(f"   AUM: ₹{fund['metadata'].get('aum', 0):,.2f} Cr")
        
#         if len(fund_recommendations) > 5:
#             print(f"   ... and {len(fund_recommendations) - 5} more")
        
#         print("\n" + "="*60)
#         print("✓ RECOMMENDATION ENGINE COMPLETED SUCCESSFULLY!")
#         print("="*60)
        
#         return result
        
#     except FileNotFoundError as e:
#         print(f"\n✗ Error: File not found - {str(e)}")
#         return None
#     except KeyError as e:
#         print(f"\n✗ Error: Missing required field - {str(e)}")
#         return None
#     except Exception as e:
#         print(f"\n✗ Error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None

# def batch_recommendations(users_json_path: str,
#                          stocks_json_path: str,
#                          mutual_funds_json_path: str,
#                          output_path: str = 'batch_recommendations.json',
#                          top_k: int = 10) -> List[Dict]:
#     """
#     Generate recommendations for multiple users
#     """
#     print("="*60)
#     print("BATCH RECOMMENDATION ENGINE")
#     print("="*60)
    
#     try:
#         # Load data
#         print(f"\nLoading data...")
#         users = load_json(users_json_path)
#         stocks = load_json(stocks_json_path)
#         mutual_funds = load_json(mutual_funds_json_path)
        
#         if not isinstance(users, list):
#             users = [users]
        
#         print(f"✓ Users: {len(users)}")
#         print(f"✓ Stocks: {len(stocks)}")
#         print(f"✓ Mutual Funds: {len(mutual_funds)}")
        
#         # Generate recommendations for each user
#         all_recommendations = []
        
#         for i, user in enumerate(users, 1):
#             print(f"\nProcessing user {i}/{len(users)}: {user['user_id']}")
            
#             user_vector = user['engineered_vector']
            
#             stock_recs = get_stock_recommendations(user_vector, stocks, top_k)
#             fund_recs = get_mutual_fund_recommendations(user_vector, mutual_funds, top_k)
            
#             result = {
#                 "user_id": user['user_id'],
#                 "user_metadata": user.get('metadata', {}),
#                 "top_stock_recommendations": stock_recs,
#                 "top_mutual_fund_recommendations": fund_recs
#             }
            
#             all_recommendations.append(result)
        
#         # Save batch results
#         print(f"\nSaving batch recommendations to: {output_path}")
#         with open(output_path, 'w') as f:
#             json.dump(all_recommendations, f, indent=2)
        
#         print(f"✓ Batch recommendations saved successfully")
#         print(f"✓ Processed {len(all_recommendations)} users")
        
#         return all_recommendations
        
#     except Exception as e:
#         print(f"\n✗ Error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None

# # Main execution
# if __name__ == "__main__":
#     # File paths
#     user_json = "engineered_users.json"
#     stocks_json = "engineered_stocks.json"
#     funds_json = "engineered_funds.json"
#     output_json = "recommendations.json"
    
#     print("\n" + "="*60)
#     print("SELECT MODE:")
#     print("="*60)
#     print("1. Single User Recommendation")
#     print("2. Batch User Recommendations")
#     print("="*60)
    
#     # For single user recommendation
#     print("\nRunning Single User Recommendation Mode...\n")
#     result = generate_recommendations(
#         user_json_path=user_json,
#         stocks_json_path=stocks_json,
#         mutual_funds_json_path=funds_json,
#         output_path=output_json,
#         top_k=10
#     )
    
#     if result:
#         print(f"\n✓ Recommendations available in: {output_json}")
#         print("\nℹ️  To process multiple users, use batch_recommendations() function")
    
#     # Uncomment below for batch processing
#     # batch_result = batch_recommendations(
#     #     users_json_path=user_json,
#     #     stocks_json_path=stocks_json,
#     #     mutual_funds_json_path=funds_json,
#     #     output_path="batch_recommendations.json",
#     #     top_k=10
#     # )






















# recommendation1.py


import json
import numpy as np
from typing import List, Dict, Tuple
import os
import glob

def load_json(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    cosine = dot(u, v) / (||u|| * ||v||)
    """
    # Convert to numpy arrays
    u = np.array(vec1)
    v = np.array(vec2)
    
    # Calculate dot product
    dot_product = np.dot(u, v)
    
    # Calculate magnitudes
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    # Handle zero vectors
    if norm_u == 0 or norm_v == 0:
        return 0.0
    
    # Calculate cosine similarity
    similarity = dot_product / (norm_u * norm_v)
    
    return float(similarity)

def get_stock_recommendations(user_vector: List[float], 
                              stocks: List[Dict], 
                              top_k: int = 10) -> List[Dict]:
    """
    Calculate cosine similarity between user and all stocks
    Return top K recommendations
    """
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
    
    # Sort by similarity score in descending order
    recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Return top K
    return recommendations[:top_k]

def get_mutual_fund_recommendations(user_vector: List[float], 
                                   mutual_funds: List[Dict], 
                                   top_k: int = 10) -> List[Dict]:
    """
    Calculate cosine similarity between user and all mutual funds
    Return top K recommendations
    """
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
    
    # Sort by similarity score in descending order
    recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Return top K
    return recommendations[:top_k]

def find_user_json_file(user_id: str) -> str:
    """
    Find user JSON file by user_id
    Returns file path if found, None otherwise
    """
    # Expected filename pattern: user_USER_0001_profile.json
    expected_file = f"user_{user_id}_profile.json"
    
    if os.path.exists(expected_file):
        return expected_file
    
    # If not found, search in current directory
    pattern = f"user_{user_id}*.json"
    matches = glob.glob(pattern)
    
    if matches:
        return matches[0]
    
    return None

def list_available_users() -> List[str]:
    """
    List all available user profile JSON files
    """
    user_files = glob.glob("user_USER_*_profile.json")
    user_ids = []
    
    for file in user_files:
        # Extract user_id from filename: user_USER_0001_profile.json -> USER_0001
        try:
            user_id = file.replace("user_", "").replace("_profile.json", "")
            user_ids.append(user_id)
        except:
            continue
    
    return sorted(user_ids)

def generate_recommendations(user_id: str,
                            stocks_json_path: str,
                            mutual_funds_json_path: str,
                            top_k: int = 10) -> Dict:
    """
    Main recommendation engine - takes user_id instead of file path
    """
    print("="*60)
    print("FINANCIAL RECOMMENDATION ENGINE")
    print("="*60)
    
    try:
        # Find user JSON file
        print(f"\n1. Looking for user: {user_id}")
        user_json_path = find_user_json_file(user_id)
        
        if not user_json_path:
            print(f"   ✗ User JSON file not found for: {user_id}")
            print(f"   Expected: user_{user_id}_profile.json")
            
            # Show available users
            available = list_available_users()
            if available:
                print(f"\n   Available users:")
                for uid in available[:10]:
                    print(f"      - {uid}")
            return None
        
        print(f"   ✓ Found: {user_json_path}")
        
        # Load user data
        user_data = load_json(user_json_path)
        print(f"   ✓ Loaded user: {user_data['user_id']}")
        
        # Extract user vector
        user_vector = user_data['engineered_vector']
        print(f"   User vector: {[round(v, 3) for v in user_vector]}")
        
        # Load stocks
        print(f"\n2. Loading stocks from: {stocks_json_path}")
        stocks = load_json(stocks_json_path)
        print(f"   ✓ Loaded {len(stocks)} stocks")
        
        # Load mutual funds
        print(f"\n3. Loading mutual funds from: {mutual_funds_json_path}")
        mutual_funds = load_json(mutual_funds_json_path)
        print(f"   ✓ Loaded {len(mutual_funds)} mutual funds")
        
        # Calculate stock recommendations
        print(f"\n4. Calculating stock recommendations...")
        stock_recommendations = get_stock_recommendations(
            user_vector, stocks, top_k
        )
        print(f"   ✓ Found top {len(stock_recommendations)} stock matches")
        
        # Calculate mutual fund recommendations
        print(f"\n5. Calculating mutual fund recommendations...")
        fund_recommendations = get_mutual_fund_recommendations(
            user_vector, mutual_funds, top_k
        )
        print(f"   ✓ Found top {len(fund_recommendations)} mutual fund matches")
        
        # Create final output
        result = {
            "user_id": user_data['user_id'],
            "user_metadata": user_data.get('metadata', {}),
            "top_stock_recommendations": stock_recommendations,
            "top_mutual_fund_recommendations": fund_recommendations
        }
        
        # Save to file with user_id in name
        output_path = f"recommendations_{user_id}.json"
        print(f"\n6. Saving recommendations to: {output_path}")
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
        
        if len(stock_recommendations) > 5:
            print(f"   ... and {len(stock_recommendations) - 5} more")
        
        print(f"\n📊 TOP {min(5, len(fund_recommendations))} MUTUAL FUND RECOMMENDATIONS:")
        for i, fund in enumerate(fund_recommendations[:5], 1):
            print(f"{i}. {fund['fund_name'][:50]}")
            print(f"   Similarity: {fund['similarity_score']:.4f}")
            if 'aum' in fund['metadata']:
                print(f"   AUM: ₹{fund['metadata']['aum']:,.2f} Cr")
        
        if len(fund_recommendations) > 5:
            print(f"   ... and {len(fund_recommendations) - 5} more")
        
        print("\n" + "="*60)
        print("✓ RECOMMENDATION ENGINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return result
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found - {str(e)}")
        return None
    except KeyError as e:
        print(f"\n✗ Error: Missing required field - {str(e)}")
        return None
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def batch_recommendations(stocks_json_path: str,
                         mutual_funds_json_path: str,
                         output_path: str = 'batch_recommendations.json',
                         top_k: int = 10) -> List[Dict]:
    """
    Generate recommendations for ALL available users
    """
    print("="*60)
    print("BATCH RECOMMENDATION ENGINE")
    print("="*60)
    
    try:
        # Find all user JSON files
        print(f"\nSearching for user profile JSONs...")
        available_users = list_available_users()
        
        if not available_users:
            print("✗ No user profile JSON files found!")
            print("   Expected pattern: user_USER_XXXX_profile.json")
            return None
        
        print(f"✓ Found {len(available_users)} users")
        
        # Load stocks and funds
        print(f"\nLoading stocks from: {stocks_json_path}")
        stocks = load_json(stocks_json_path)
        print(f"✓ Loaded {len(stocks)} stocks")
        
        print(f"\nLoading mutual funds from: {mutual_funds_json_path}")
        mutual_funds = load_json(mutual_funds_json_path)
        print(f"✓ Loaded {len(mutual_funds)} mutual funds")
        
        # Generate recommendations for each user
        all_recommendations = []
        
        for i, user_id in enumerate(available_users, 1):
            print(f"\n[{i}/{len(available_users)}] Processing: {user_id}")
            
            user_json_path = find_user_json_file(user_id)
            user_data = load_json(user_json_path)
            user_vector = user_data['engineered_vector']
            
            stock_recs = get_stock_recommendations(user_vector, stocks, top_k)
            fund_recs = get_mutual_fund_recommendations(user_vector, mutual_funds, top_k)
            
            result = {
                "user_id": user_data['user_id'],
                "user_metadata": user_data.get('metadata', {}),
                "top_stock_recommendations": stock_recs,
                "top_mutual_fund_recommendations": fund_recs
            }
            
            all_recommendations.append(result)
            print(f"   ✓ Completed")
        
        # Save batch results
        print(f"\n{'='*60}")
        print(f"Saving batch recommendations to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(all_recommendations, f, indent=2)
        
        print(f"✓ Batch recommendations saved successfully")
        print(f"✓ Processed {len(all_recommendations)} users")
        print(f"{'='*60}")
        
        return all_recommendations
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Main execution
if __name__ == "__main__":
    # File paths for stocks and funds (these remain constant)
    stocks_json = "engineered_stocks.json"
    funds_json = "engineered_funds.json"
    
    print("\n" + "="*60)
    print("SELECT MODE:")
    print("="*60)
    print("1. Single User Recommendation")
    print("2. Batch Recommendations (All Users)")
    print("="*60)
    
    mode = input("\nEnter mode (1 or 2): ").strip()
    
    if mode == "1":
        # Single user mode
        print("\n📋 Available Users:")
        available = list_available_users()
        
        if not available:
            print("✗ No user profiles found!")
            print("   Please run the user generation script first.")
        else:
            for i, uid in enumerate(available, 1):
                print(f"   {i}. {uid}")
            
            user_id = input("\n👤 Enter User ID (e.g., USER_0001): ").strip()
            
            result = generate_recommendations(
                user_id=user_id,
                stocks_json_path=stocks_json,
                mutual_funds_json_path=funds_json,
                top_k=10
            )
            
            if result:
                print(f"\n✓ Recommendations saved: recommendations_{user_id}.json")
    
    elif mode == "2":
        # Batch mode
        print("\nRunning Batch Recommendation Mode...\n")
        batch_result = batch_recommendations(
            stocks_json_path=stocks_json,
            mutual_funds_json_path=funds_json,
            output_path="batch_recommendations.json",
            top_k=10
        )
        
        if batch_result:
            print(f"\n✓ Batch recommendations saved: batch_recommendations.json")
    
    else:
        print("\n✗ Invalid mode selected!")