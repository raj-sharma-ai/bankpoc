



# # app.py - Streamlit Frontend with Individual LLM Explanations

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime

# # =====================================================
# # CONFIGURATION
# # =====================================================

# API_BASE_URL = "http://localhost:8000"

# st.set_page_config(
#     page_title="Financial Recommendation System",
#     page_icon="💰",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 1.5rem;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         margin: 0.5rem 0;
#     }
#     .risk-low {
#         background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
#     }
#     .risk-medium {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#     }
#     .risk-high {
#         background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
#     }
#     .recommendation-card {
#         border: 2px solid #e0e0e0;
#         border-radius: 10px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         background: white;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
#     .stButton>button {
#         width: 100%;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         border-radius: 5px;
#         padding: 0.5rem 1rem;
#         font-weight: bold;
#     }
#     .llm-explanation {
#         background: ;
#         padding: 1rem;
#         border-radius: 8px;
#         border-left: 4px solid #2196F3;
#         margin-top: 1rem;
#         font-size: 0.95rem;
#         line-height: 1.6;
#     }
#     .explanation-title {
#         color: #1565C0;
#         font-weight: 600;
#         margin-bottom: 0.5rem;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # =====================================================
# # API HELPER FUNCTIONS
# # =====================================================

# def get_all_users():
#     """Fetch all users from API"""
#     try:
#         response = requests.get(f"{API_BASE_URL}/api/users")
#         if response.status_code == 200:
#             return response.json()
#         return None
#     except Exception as e:
#         st.error(f"Error fetching users: {str(e)}")
#         return None

# def get_user_profile(user_id):
#     """Fetch user profile from API"""
#     try:
#         response = requests.get(f"{API_BASE_URL}/api/user/{user_id}")
#         if response.status_code == 200:
#             return response.json()
#         return None
#     except Exception as e:
#         st.error(f"Error fetching user profile: {str(e)}")
#         return None

# def get_recommendations(user_id, top_k=3):
#     """Fetch recommendations from API"""
#     try:
#         response = requests.get(f"{API_BASE_URL}/api/recommendations/{user_id}?top_k={top_k}")
#         if response.status_code == 200:
#             return response.json()
#         return None
#     except Exception as e:
#         st.error(f"Error fetching recommendations: {str(e)}")
#         return None

# def get_individual_explanation(user_metadata, item_type, item_data):
#     """Fetch individual explanation for a stock or mutual fund"""
#     try:
#         payload = {
#             "user_profile": user_metadata,
#             "item_type": item_type,  # "stock" or "mutual_fund"
#             "item_data": item_data
#         }
        
#         response = requests.post(
#             f"{API_BASE_URL}/api/explain-individual",
#             json=payload,
#             timeout=30
#         )
        
#         if response.status_code == 200:
#             return response.json()
#         return None
#     except Exception as e:
#         return None

# def check_llm_health():
#     """Check if Ollama LLM is available"""
#     try:
#         response = requests.get(f"{API_BASE_URL}/api/llm/health", timeout=5)
#         if response.status_code == 200:
#             return response.json()
#         return None
#     except Exception as e:
#         return None

# def get_admin_logs():
#     """Fetch admin logs from API"""
#     try:
#         response = requests.get(f"{API_BASE_URL}/api/admin/logs")
#         if response.status_code == 200:
#             return response.json()
#         return None
#     except Exception as e:
#         st.error(f"Error fetching logs: {str(e)}")
#         return None

# def get_stats():
#     """Fetch system statistics"""
#     try:
#         response = requests.get(f"{API_BASE_URL}/api/stats")
#         if response.status_code == 200:
#             return response.json()
#         return None
#     except Exception as e:
#         st.error(f"Error fetching stats: {str(e)}")
#         return None

# def refresh_data():
#     """Trigger data refresh"""
#     try:
#         response = requests.post(f"{API_BASE_URL}/api/admin/refresh-data")
#         if response.status_code == 200:
#             return response.json()
#         return None
#     except Exception as e:
#         st.error(f"Error refreshing data: {str(e)}")
#         return None

# # =====================================================
# # SIDEBAR NAVIGATION
# # =====================================================

# st.sidebar.title("🏦 Navigation")
# page = st.sidebar.radio(
#     "Select Page",
#     ["🏠 Home", "👤 Customer 360 View", "📊 Recommendations", "⚙️ Admin Dashboard"]
# )

# st.sidebar.markdown("---")
# st.sidebar.info("""
#     **Financial Recommendation System**
    
#     Version 2.0.0
    
#     Powered by FastAPI & Streamlit
    
#     🤖 AI-Powered Explanations
# """)

# # =====================================================
# # PAGE 1: HOME
# # =====================================================

# if page == "🏠 Home":
#     st.markdown('<div class="main-header">💰 Financial Recommendation System</div>', unsafe_allow_html=True)
    
#     st.markdown("""
#     ### Welcome to the Financial Recommendation Platform
    
#     This intelligent system provides personalized investment recommendations based on:
#     - **Risk Profiling**: Advanced ML-based risk assessment
#     - **Financial Health Analysis**: Comprehensive financial metrics
#     - **Personalized Recommendations**: AI-powered stock and mutual fund suggestions
#     - **🤖 Individual AI Explanations**: Get AI insights for each recommendation
#     """)
    
#     stats = get_stats()
    
#     if stats:
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.markdown(f"""
#                 <div class="metric-card">
#                     <h2>{stats['users_loaded']}</h2>
#                     <p>Total Users</p>
#                 </div>
#             """, unsafe_allow_html=True)
        
#         with col2:
#             st.markdown(f"""
#                 <div class="metric-card">
#                     <h2>{stats['stocks_loaded']}</h2>
#                     <p>Stocks Available</p>
#                 </div>
#             """, unsafe_allow_html=True)
        
#         with col3:
#             st.markdown(f"""
#                 <div class="metric-card">
#                     <h2>{stats['funds_loaded']}</h2>
#                     <p>Mutual Funds</p>
#                 </div>
#             """, unsafe_allow_html=True)
        
#         with col4:
#             model_status = "✅ Loaded" if stats['model_loaded'] else "❌ Not Loaded"
#             st.markdown(f"""
#                 <div class="metric-card">
#                     <h2>{model_status}</h2>
#                     <p>ML Model</p>
#                 </div>
#             """, unsafe_allow_html=True)
    
#     st.markdown("---")
    
#     # LLM Status on Home Page
#     st.subheader("🤖 AI Assistant Status")
    
#     llm_health = check_llm_health()
    
#     if llm_health and llm_health.get('status') == 'online':
#         st.success(f"✅ AI Assistant is Online - Model: {llm_health.get('configured_model', 'N/A')}")
#         st.info(f"📚 Available Models: {', '.join(llm_health.get('available_models', []))}")
#     else:
#         st.warning("⚠️ AI Assistant is Offline - Fallback explanations will be used")
#         with st.expander("🔧 How to enable AI Assistant"):
#             st.markdown("""
#             **Setup Ollama:**
#             1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
#             2. Pull model: `ollama pull llama3.1`
#             3. Verify: `curl http://localhost:11434/api/tags`
#             """)
    
#     st.markdown("---")
    
#     st.markdown("""
#     ### 🚀 Quick Start Guide
    
#     1. **Customer 360 View**: Select a user to view their complete financial profile
#     2. **Recommendations**: Get personalized recommendations with automatic AI explanations
#     3. **Admin Dashboard**: Monitor system logs and LLM status
    
#     Navigate using the sidebar to get started!
#     """)

# # =====================================================
# # PAGE 2: CUSTOMER 360 VIEW
# # =====================================================

# elif page == "👤 Customer 360 View":
#     st.markdown('<div class="main-header">👤 Customer 360° View</div>', unsafe_allow_html=True)
    
#     users_data = get_all_users()
    
#     if users_data and users_data.get('users'):
#         users_list = users_data['users']
#         user_ids = [user['customer_id'] for user in users_list]
        
#         selected_user_id = st.selectbox("🔍 Select Customer ID", user_ids)
        
#         if st.button("🔄 Load Customer Profile", key="load_profile"):
#             with st.spinner("Loading customer profile..."):
#                 user_profile = get_user_profile(selected_user_id)
                
#                 if user_profile:
#                     st.success(f"✅ Profile loaded for {selected_user_id}")
#                     st.session_state.current_profile = user_profile
        
#         if 'current_profile' in st.session_state:
#             profile = st.session_state.current_profile
#             metadata = profile['metadata']
#             derived = profile['derived_features']
            
#             st.markdown("---")
            
#             st.subheader("📋 Basic Information")
            
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.metric("Age", f"{metadata['age']} years")
#             with col2:
#                 st.metric("Gender", metadata['gender'])
#             with col3:
#                 st.metric("Occupation", metadata['occupation'])
#             with col4:
#                 st.metric("Credit Score", metadata['credit_score'])
            
#             st.markdown("---")
            
#             st.subheader("⚠️ Risk Profile")
            
#             risk_label = metadata['risk_label'].upper()
#             risk_score = metadata['risk_score']
            
#             risk_class = f"risk-{metadata['risk_label'].lower()}"
            
#             col1, col2 = st.columns([1, 2])
            
#             with col1:
#                 st.markdown(f"""
#                     <div class="metric-card {risk_class}">
#                         <h1>{risk_label}</h1>
#                         <p>Risk Category</p>
#                         <h3>Score: {risk_score}</h3>
#                     </div>
#                 """, unsafe_allow_html=True)
            
#             with col2:
#                 fig = go.Figure(go.Indicator(
#                     mode="gauge+number",
#                     value=risk_score * 100,
#                     title={'text': "Risk Score (%)"},
#                     gauge={
#                         'axis': {'range': [0, 100]},
#                         'bar': {'color': "darkblue"},
#                         'steps': [
#                             {'range': [0, 33], 'color': "lightgreen"},
#                             {'range': [33, 66], 'color': "yellow"},
#                             {'range': [66, 100], 'color': "salmon"}
#                         ],
#                         'threshold': {
#                             'line': {'color': "red", 'width': 4},
#                             'thickness': 0.75,
#                             'value': risk_score * 100
#                         }
#                     }
#                 ))
#                 fig.update_layout(height=250)
#                 st.plotly_chart(fig, width='stretch')
            
#             st.markdown("---")
            
#             st.subheader("💰 Financial Health")
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 st.metric("Annual Income", f"₹{metadata['income']:,.0f}")
#                 st.metric("Savings Rate", f"{metadata['savings_rate']*100:.1f}%")
            
#             with col2:
#                 st.metric("Debt-to-Income", f"{metadata['debt_to_income']*100:.1f}%")
#                 st.metric("Credit Utilization", f"{derived['credit_utilization']*100:.1f}%")
            
#             with col3:
#                 st.metric("Digital Activity", f"{metadata['digital_activity']:.0f}/100")
#                 st.metric("Portfolio Diversity", f"{metadata['portfolio_diversity']:.0f}/100")
            
#             st.markdown("---")
#             st.subheader("📈 Financial Metrics Breakdown")
            
#             metrics_data = {
#                 'Metric': ['Savings Rate', 'Debt-to-Income', 'Credit Utilization', 
#                           'Spending Stability', 'Transaction Volatility'],
#                 'Value': [
#                     metadata['savings_rate'] * 100,
#                     metadata['debt_to_income'] * 100,
#                     derived['credit_utilization'] * 100,
#                     derived['spending_stability'] * 100,
#                     derived['transaction_volatility'] * 100
#                 ]
#             }
            
#             df_metrics = pd.DataFrame(metrics_data)
#             fig = px.bar(df_metrics, x='Metric', y='Value', 
#                         title='Financial Health Indicators (%)',
#                         color='Value',
#                         color_continuous_scale='RdYlGn_r')
#             fig.update_layout(height=400)
#             st.plotly_chart(fig, width='stretch')
            
#             st.markdown("---")
            
#             st.subheader("📊 Past Investment Profile")
            
#             base_user = next((u for u in users_list if u['customer_id'] == selected_user_id), None)
            
#             if base_user:
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.metric("Investment Amount (Last Year)", 
#                              f"₹{base_user['investmentamountlastyear']:,.0f}")
                
#                 with col2:
#                     st.metric("Past Investments", base_user['pastinvestments'])
                
#                 if base_user['investmentamountlastyear'] > 0:
#                     investment_type = base_user.get('pastinvestments', '')
#                     if investment_type:
#                         types = investment_type.replace('_', ', ').split(', ')
#                         st.info(f"**Investment Categories**: {', '.join(types)}")
    
#     else:
#         st.error("⚠️ Unable to load users. Please check if the API is running.")
#         st.info("Start the FastAPI server with: `python main.py`")

# #Recommendations Page


# elif page == "📊 Recommendations":
#     st.markdown('<div class="main-header">📊 Investment Recommendations</div>', unsafe_allow_html=True)
    
#     users_data = get_all_users()
    
#     if users_data and users_data.get('users'):
#         users_list = users_data['users']
#         user_ids = [user['customer_id'] for user in users_list]
        
#         col1, col2 = st.columns([3, 1])
        
#         with col1:
#             selected_user_id = st.selectbox("🔍 Select Customer ID", user_ids)
        
#         with col2:
#             top_k = st.number_input("Top K", min_value=5, max_value=20, value=10)
        
#         if st.button("🚀 Generate Recommendations", key="gen_rec"):
#             with st.spinner("🤖 Generating personalized recommendations with AI explanations..."):
#                 recommendations = get_recommendations(selected_user_id, top_k)
                
#                 if recommendations:
#                     st.success(f"✅ Recommendations generated for {selected_user_id}")
#                     st.session_state.recommendations = recommendations
#                     # Clear any previous explanations
#                     st.session_state.stock_explanations = {}
#                     st.session_state.fund_explanations = {}
#                     st.session_state.insurance_explanations = {}
#                     st.rerun()
        
#         if 'recommendations' in st.session_state:
#             recs = st.session_state.recommendations
#             user_meta = recs['user_metadata']
            
#             # Initialize explanation storage if not exists
#             if 'stock_explanations' not in st.session_state:
#                 st.session_state.stock_explanations = {}
#             if 'fund_explanations' not in st.session_state:
#                 st.session_state.fund_explanations = {}
#             if 'insurance_explanations' not in st.session_state:
#                 st.session_state.insurance_explanations = {}
            
#             st.markdown("---")
            
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.metric("Risk Profile", user_meta['risk_label'].upper())
#             with col2:
#                 st.metric("Annual Income", f"₹{user_meta['income']:,.0f}")
#             with col3:
#                 st.metric("Credit Score", user_meta['credit_score'])
#             with col4:
#                 st.metric("Savings Rate", f"{user_meta['savings_rate']*100:.1f}%")
            
#             st.markdown("---")
            
#             # STOCKS SECTION WITH INDIVIDUAL EXPLANATIONS
#             st.subheader("📈 Top Stock Recommendations")
            
#             stocks = recs['top_stock_recommendations']
            
#             for i, stock in enumerate(stocks[:5], 1):
#                 stock_key = f"stock_{i}"
                
#                 with st.container():
#                     st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                    
#                     col1, col2, col3 = st.columns([2, 2, 1])
                    
#                     with col1:
#                         st.markdown(f"### {i}. {stock['symbol']}")
#                         st.markdown(f"**{stock['company_name']}**")
                    
#                     with col2:
#                         sector = stock['metadata'].get('sector', 'N/A')
#                         st.markdown(f"**Sector**: {sector}")
                        
#                         market_cap = stock['metadata'].get('market_cap', 'N/A')
#                         st.markdown(f"**Market Cap**: {market_cap}")
                    
#                     with col3:
#                         similarity = stock['similarity_score']
#                         st.metric("Match Score", f"{similarity*100:.1f}%")
                    
#                     st.progress(similarity)
                    
#                     # Auto-generate explanation if not already generated
#                     if stock_key not in st.session_state.stock_explanations:
#                         with st.spinner(f"🤖 Generating AI insight for {stock['symbol']}..."):
#                             explanation = get_individual_explanation(
#                                 user_meta,
#                                 "stock",
#                                 stock
#                             )
                            
#                             if explanation and explanation.get('explanation'):
#                                 st.session_state.stock_explanations[stock_key] = explanation['explanation']
#                             else:
#                                 # Enhanced diverse fallback explanations
#                                 sector = stock['metadata'].get('sector', 'diversified')
#                                 symbol = stock['symbol']
#                                 company = stock['company_name']
#                                 risk_level = user_meta['risk_label']
#                                 score = similarity * 100
                                
#                                 # Different explanation templates based on rank
#                                 if i == 1:
#                                     fallback = f"""🏆 Top recommendation! {symbol} ({sector} sector) shows the highest compatibility at {score:.1f}% with your {risk_level} risk profile. 
#                                     Your strong credit score of {user_meta['credit_score']} and healthy savings rate of {user_meta['savings_rate']*100:.1f}% make this an ideal choice. 
#                                     This stock could serve as a core holding in your portfolio given your financial stability."""
#                                 elif i == 2:
#                                     fallback = f"""Strong diversification option! {symbol} from the {sector} sector complements your portfolio with {score:.1f}% match. 
#                                     With your annual income of ₹{user_meta['income']:,.0f}, you have sufficient capacity for this investment. 
#                                     Your low debt-to-income ratio of {user_meta['debt_to_income']*100:.1f}% supports adding this position."""
#                                 elif i == 3:
#                                     fallback = f"""Balanced growth potential! {symbol} aligns with your {risk_level} tolerance ({score:.1f}% compatibility). 
#                                     This {sector} stock adds sector diversification to your holdings. 
#                                     Your financial metrics suggest you can comfortably allocate funds here while maintaining portfolio balance."""
#                                 elif i == 4:
#                                     fallback = f"""Strategic addition! {symbol} offers {score:.1f}% alignment with your investment profile. 
#                                     Given your savings rate of {user_meta['savings_rate']*100:.1f}%, this {sector} sector exposure could enhance returns. 
#                                     Your stable financial position supports this moderately sized allocation."""
#                                 else:
#                                     fallback = f"""Solid opportunity! {symbol} from {sector} sector shows {score:.1f}% compatibility. 
#                                     This investment fits well within your {risk_level} risk framework and income level of ₹{user_meta['income']:,.0f}. 
#                                     Consider this for portfolio diversification and potential growth."""
                                
#                                 st.session_state.stock_explanations[stock_key] = fallback
                    
#                     # Display explanation
#                     if stock_key in st.session_state.stock_explanations:
#                         st.markdown(f"""
#                         <div class="llm-explanation">
#                             <div class="explanation-title">💡 Why this recommendation?</div>
#                             {st.session_state.stock_explanations[stock_key]}
#                         </div>
#                         """, unsafe_allow_html=True)
                    
#                     st.markdown('</div>', unsafe_allow_html=True)
#                     st.markdown("")  # spacing
            
#             st.markdown("---")
            
#             # MUTUAL FUNDS SECTION WITH INDIVIDUAL EXPLANATIONS
#             st.subheader("💼 Top Mutual Fund Recommendations")
            
#             funds = recs['top_mutual_fund_recommendations']
            
#             for i, fund in enumerate(funds[:5], 1):
#                 fund_key = f"fund_{i}"
                
#                 with st.container():
#                     st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                    
#                     col1, col2 = st.columns([3, 1])
                    
#                     with col1:
#                         st.markdown(f"### {i}. {fund['fund_name']}")
                        
#                         category = fund['metadata'].get('category', 'N/A')
#                         st.markdown(f"**Category**: {category}")
                        
#                         if 'aum' in fund['metadata']:
#                             st.markdown(f"**AUM**: ₹{fund['metadata']['aum']:,.2f} Cr")
                    
#                     with col2:
#                         similarity = fund['similarity_score']
#                         st.metric("Match Score", f"{similarity*100:.1f}%")
                    
#                     st.progress(similarity)
                    
#                     if fund.get('fund_link'):
#                         st.markdown(f"[🔗 More Info]({fund['fund_link']})")
                    
#                     # Auto-generate explanation if not already generated
#                     if fund_key not in st.session_state.fund_explanations:
#                         with st.spinner(f"🤖 Generating AI insight for this fund..."):
#                             explanation = get_individual_explanation(
#                                 user_meta,
#                                 "mutual_fund",
#                                 fund
#                             )
                            
#                             if explanation and explanation.get('explanation'):
#                                 st.session_state.fund_explanations[fund_key] = explanation['explanation']
#                             else:
#                                 # Enhanced diverse fallback explanations for mutual funds
#                                 category = fund['metadata'].get('category', 'diversified')
#                                 fund_name_short = fund['fund_name'][:40] + "..." if len(fund['fund_name']) > 40 else fund['fund_name']
#                                 risk_level = user_meta['risk_label']
#                                 score = similarity * 100
#                                 aum = fund['metadata'].get('aum', 0)
                                
#                                 # Different explanation templates based on rank
#                                 if i == 1:
#                                     fallback = f"""🏆 Best match! This {category} fund leads with {score:.1f}% compatibility for your {risk_level} risk profile. 
#                                     Your income of ₹{user_meta['income']:,.0f} and low debt burden ({user_meta['debt_to_income']*100:.1f}% DTI) make this an excellent core investment. 
#                                     The fund's track record and category alignment suggest strong potential for your portfolio."""
#                                 elif i == 2:
#                                     fallback = f"""Excellent diversifier! This {category} fund offers {score:.1f}% alignment with your investment goals. 
#                                     With your credit score of {user_meta['credit_score']}, you can leverage this fund for balanced growth. 
#                                     Your savings rate of {user_meta['savings_rate']*100:.1f}% supports systematic investment in this category."""
#                                 elif i == 3:
#                                     fallback = f"""Smart allocation option! This {category} fund shows {score:.1f}% compatibility with your profile. 
#                                     Given your financial stability, this fund can provide sector-specific exposure while managing risk. 
#                                     Your {risk_level} tolerance aligns well with the fund's investment strategy."""
#                                 elif i == 4:
#                                     fallback = f"""Strategic complement! At {score:.1f}% match, this {category} fund enhances portfolio diversity. 
#                                     Your debt-to-income ratio of {user_meta['debt_to_income']*100:.1f}% indicates capacity for this allocation. 
#                                     This fund can balance your existing investments while targeting growth."""
#                                 else:
#                                     fallback = f"""Valuable addition! This {category} fund offers {score:.1f}% alignment with your {risk_level} approach. 
#                                     Your annual income level supports this investment for long-term wealth building. 
#                                     The fund provides diversification benefits and professional management suited to your profile."""
                                
#                                 st.session_state.fund_explanations[fund_key] = fallback
                    
#                     # Display explanation
#                     if fund_key in st.session_state.fund_explanations:
#                         st.markdown(f"""
#                         <div class="llm-explanation">
#                             <div class="explanation-title">💡 Why this recommendation?</div>
#                             {st.session_state.fund_explanations[fund_key]}
#                         </div>
#                         """, unsafe_allow_html=True)
                    
#                     st.markdown('</div>', unsafe_allow_html=True)
#                     st.markdown("")  # spacing
            
#             st.markdown("---")
            
#             # INSURANCE SECTION WITH INDIVIDUAL EXPLANATIONS
#             st.subheader("🛡️ Top Insurance Recommendations")
            
#             insurances = recs.get('top_insurance_recommendations', [])
            
#             if insurances:
#                 for i, insurance in enumerate(insurances[:5], 1):
#                     insurance_key = f"insurance_{i}"
                    
#                     with st.container():
#                         st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                        
#                         col1, col2, col3 = st.columns([2, 2, 1])
                        
#                         with col1:
#                             st.markdown(f"### {i}. {insurance['insurance_name']}")
#                             insurance_type = insurance.get('insurance_type', 'N/A')
#                             insurer = insurance['metadata'].get('insurer', 'N/A')
#                             st.markdown(f"**Type**: {insurance_type}")
#                             st.markdown(f"**Insurer**: {insurer}")
                        
#                         with col2:
#                             premium_range = insurance['metadata'].get('premium_range', 'N/A')
#                             sum_insured = insurance['metadata'].get('sum_insured', 'N/A')
                            
#                             st.markdown(f"**Premium**: {premium_range}")
#                             st.markdown(f"**Sum Insured**: {sum_insured}")
                            
#                             # Show key features
#                             if insurance['metadata'].get('opd_cover'):
#                                 st.markdown("✅ OPD Cover")
#                             if insurance['metadata'].get('maternity_cover'):
#                                 st.markdown("✅ Maternity Cover")
                        
#                         with col3:
#                             similarity = insurance['similarity_score']
#                             st.metric("Match Score", f"{similarity*100:.1f}%")
                        
#                         st.progress(similarity)
                        
#                         # Display policy URL if available
#                         policy_url = insurance['metadata'].get('url', '')
#                         if policy_url:
#                             st.markdown(f"[🔗 View Policy Details]({policy_url})")
                        
#                         # Auto-generate explanation if not already generated
#                         if insurance_key not in st.session_state.insurance_explanations:
#                             with st.spinner(f"🤖 Generating AI insight for this insurance..."):
#                                 explanation = get_individual_explanation(
#                                     user_meta,
#                                     "insurance",
#                                     insurance
#                                 )
                                
#                                 if explanation and explanation.get('explanation'):
#                                     st.session_state.insurance_explanations[insurance_key] = explanation['explanation']
#                                 else:
#                                     # Enhanced diverse fallback explanations for insurance
#                                     insurance_type = insurance.get('insurance_type', 'comprehensive')
#                                     insurance_name = insurance['insurance_name']
#                                     insurer = insurance['metadata'].get('insurer', 'leading insurer')
#                                     risk_level = user_meta['risk_label']
#                                     score = similarity * 100
                                    
#                                     # Get key features
#                                     has_opd = insurance['metadata'].get('opd_cover', False)
#                                     has_maternity = insurance['metadata'].get('maternity_cover', False)
#                                     has_critical = insurance['metadata'].get('critical_illness_cover', False)
#                                     covers_preexisting = insurance['metadata'].get('covers_preexisting', False)
                                    
#                                     # Different explanation templates based on rank
#                                     if i == 1:
#                                         fallback = f"""🏆 Perfect protection match! **{insurance_name}** by {insurer} shows {score:.1f}% compatibility with your financial profile. 
#                                         Your annual income of ₹{user_meta['income']:,.0f} comfortably supports the premium, ensuring long-term affordability. 
#                                         With your credit score of {user_meta['credit_score']}, you qualify for favorable terms. This comprehensive policy includes {'OPD cover, ' if has_opd else ''}{'maternity benefits, ' if has_maternity else ''}{'critical illness protection, ' if has_critical else ''} providing complete health security."""
#                                     elif i == 2:
#                                         fallback = f"""Essential coverage! **{insurance_name}** by {insurer} aligns {score:.1f}% with your {risk_level} risk profile. 
#                                         At your income level and with {user_meta['debt_to_income']*100:.1f}% debt-to-income ratio, the premium is manageable within your budget. 
#                                         {'This policy covers pre-existing conditions, ' if covers_preexisting else ''}making it an excellent choice for comprehensive health protection for your family."""
#                                     elif i == 3:
#                                         fallback = f"""Smart protection strategy! **{insurance_name}** from {insurer} offers {score:.1f}% match with your needs. 
#                                         Your savings rate of {user_meta['savings_rate']*100:.1f}% indicates you can allocate funds for this important protection. 
#                                         The policy's {'OPD and ' if has_opd else ''}{'maternity ' if has_maternity else ''}coverage complements your investment strategy by providing security against medical emergencies."""
#                                     elif i == 4:
#                                         fallback = f"""Valuable safety net! **{insurance_name}** by {insurer} shows {score:.1f}% compatibility with your financial situation. 
#                                         Given your stable income and low debt burden, this insurance enhances your overall financial resilience. 
#                                         With features like {'critical illness cover' if has_critical else 'comprehensive health benefits'}, it's well-suited to protect your dependents and assets."""
#                                     else:
#                                         fallback = f"""Comprehensive protection! **{insurance_name}** from {insurer} provides {score:.1f}% alignment with your profile. 
#                                         Your financial stability supports this premium commitment while maintaining your investment capacity. 
#                                         This policy addresses specific protection needs identified based on your {risk_level} risk profile and life stage."""
                                    
#                                     st.session_state.insurance_explanations[insurance_key] = fallback
                        
#                         # Display explanation
#                         if insurance_key in st.session_state.insurance_explanations:
#                             st.markdown(f"""
#                             <div class="llm-explanation">
#                                 <div class="explanation-title">💡 Why this recommendation?</div>
#                                 {st.session_state.insurance_explanations[insurance_key]}
#                             </div>
#                             """, unsafe_allow_html=True)
                        
#                         st.markdown('</div>', unsafe_allow_html=True)
#                         st.markdown("")  # spacing
#             else:
#                 st.info("ℹ️ No insurance recommendations available for this user.")
    
#     else:
#         st.error("⚠️ Unable to load users. Please check if the API is running.")

# # =====================================================
# # PAGE 4: ADMIN DASHBOARD
# # =====================================================

# elif page == "⚙️ Admin Dashboard":
#     st.markdown('<div class="main-header">⚙️ Admin Dashboard</div>', unsafe_allow_html=True)
    
#     st.subheader("📊 System Statistics")
    
#     stats = get_stats()
    
#     if stats:
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Total Users", stats['users_loaded'])
#         with col2:
#             st.metric("Stocks Available", stats['stocks_loaded'])
#         with col3:
#             st.metric("Mutual Funds", stats['funds_loaded'])
#         with col4:
#             model_status = "Loaded ✅" if stats['model_loaded'] else "Not Loaded ❌"
#             st.metric("ML Model", model_status)
    
#     st.markdown("---")
    
#     # LLM HEALTH STATUS
#     st.subheader("🤖 LLM Assistant Status")
    
#     llm_health = check_llm_health()
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if llm_health:
#             status = llm_health.get('status', 'unknown')
#             if status == 'online':
#                 st.success("🟢 LLM Online")
#             elif status == 'offline':
#                 st.error("🔴 LLM Offline")
#             else:
#                 st.warning("🟡 LLM Unknown")
#         else:
#             st.error("🔴 Cannot Connect")
    
#     with col2:
#         if llm_health and llm_health.get('status') == 'online':
#             model = llm_health.get('configured_model', 'N/A')
#             st.metric("Active Model", model)
#         else:
#             st.metric("Active Model", "N/A")
    
#     with col3:
#         if llm_health and llm_health.get('status') == 'online':
#             models = llm_health.get('available_models', [])
#             st.metric("Available Models", len(models))
#         else:
#             st.metric("Available Models", "0")
    
#     if llm_health and llm_health.get('status') == 'online':
#         with st.expander("📋 Available Ollama Models"):
#             models = llm_health.get('available_models', [])
#             if models:
#                 for model in models:
#                     st.text(f"• {model}")
#             else:
#                 st.warning("No models found")
    
#     with st.expander("⚙️ LLM Setup Instructions"):
#         st.markdown("""
#         **To enable AI explanations, ensure Ollama is running:**
        
#         1. Install Ollama: https://ollama.ai/
#         2. Pull the model:
#            ```bash
#            ollama pull llama3.1
#            ```
#         3. Start Ollama (it runs automatically after installation)
#         4. Verify it's running:
#            ```bash
#            curl http://localhost:11434/api/tags
#            ```
        
#         **Troubleshooting:**
#         - If LLM is offline, restart Ollama service
#         - Check if port 11434 is available
#         - Ensure you have sufficient RAM (8GB+ recommended)
#         """)
    
#     st.markdown("---")
    
#     st.subheader("🔄 Data Refresh")
    
#     col1, col2 = st.columns([3, 1])
    
#     with col1:
#         st.info("Refresh system data from files (users, stocks, mutual funds)")
    
#     with col2:
#         if st.button("🔄 Refresh Data", key="refresh"):
#             with st.spinner("Refreshing data..."):
#                 result = refresh_data()
#                 if result:
#                     st.success("✅ Data refreshed successfully!")
#                 else:
#                     st.error("❌ Failed to refresh data")
    
#     st.markdown("---")
    
#     st.subheader("📋 System Logs")
    
#     logs_data = get_admin_logs()
    
#     if logs_data and logs_data.get('logs'):
#         logs = logs_data['logs']
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             action_filter = st.multiselect(
#                 "Filter by Action",
#                 options=list(set([log['action'] for log in logs])),
#                 default=None
#             )
        
#         with col2:
#             status_filter = st.multiselect(
#                 "Filter by Status",
#                 options=list(set([log['status'] for log in logs])),
#                 default=None
#             )
        
#         with col3:
#             limit = st.number_input("Show last N logs", min_value=10, max_value=100, value=20)
        
#         filtered_logs = logs[-limit:]
        
#         if action_filter:
#             filtered_logs = [log for log in filtered_logs if log['action'] in action_filter]
        
#         if status_filter:
#             filtered_logs = [log for log in filtered_logs if log['status'] in status_filter]
        
#         logs_df = pd.DataFrame(filtered_logs)
        
#         if not logs_df.empty:
#             st.dataframe(
#                 logs_df[['timestamp', 'action', 'status', 'details']],
#                 width='stretch',
#                 height=400
#             )
            
#             st.markdown("---")
#             st.subheader("📈 Log Statistics")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 status_counts = logs_df['status'].value_counts()
#                 fig = px.pie(
#                     values=status_counts.values,
#                     names=status_counts.index,
#                     title='Status Distribution'
#                 )
#                 st.plotly_chart(fig, width='stretch')
            
#             with col2:
#                 action_counts = logs_df['action'].value_counts()
#                 fig = px.bar(
#                     x=action_counts.index,
#                     y=action_counts.values,
#                     title='Actions Performed',
#                     labels={'x': 'Action', 'y': 'Count'}
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.warning("No logs match the selected filters")
    
#     else:
#         st.warning("⚠️ No logs available")

# # =====================================================
# # FOOTER
# # =====================================================

# # st.markdown("---")
# # st

# st.markdown("---")
# st.markdown("""
#     <div style="text-align: center; color: gray; padding: 1rem;">
#         <p>Financial Recommendation System v1.0.0</p>
#         <p>Built with ❤️ using FastAPI & Streamlit</p>
#         <p>🤖 Powered by Ollama LLM</p>
#     </div>
# """, unsafe_allow_html=True)































# app.py - Streamlit Frontend with Scheduler Monitoring

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =====================================================
# CONFIGURATION
# =====================================================

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Financial Recommendation System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .recommendation-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .llm-explanation {
       
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin-top: 1rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .explanation-title {
        color: #1565C0;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .scheduler-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .scheduler-running {
        background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
    }
    .scheduler-idle {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
    }
    .scheduler-disabled {
        background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);
    }
    </style>
""", unsafe_allow_html=True)










# =====================================================
# API HELPER FUNCTIONS
# =====================================================

def get_all_users():
    """Fetch all users from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/users")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching users: {str(e)}")
        return None

def get_user_profile(user_id):
    """Fetch user profile from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/user/{user_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching user profile: {str(e)}")
        return None

def get_recommendations(user_id, top_k=3):
    """Fetch recommendations from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/recommendations/{user_id}?top_k={top_k}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching recommendations: {str(e)}")
        return None

def get_individual_explanation(user_metadata, item_type, item_data):
    """Fetch individual explanation for a stock or mutual fund"""
    try:
        payload = {
            "user_profile": user_metadata,
            "item_type": item_type,
            "item_data": item_data
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/explain-individual",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def check_llm_health():
    """Check if Ollama LLM is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/llm/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def get_admin_logs():
    """Fetch admin logs from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/admin/logs")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching logs: {str(e)}")
        return None

def get_stats():
    """Fetch system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching stats: {str(e)}")
        return None

def refresh_data():
    """Trigger data refresh"""
    try:
        response = requests.post(f"{API_BASE_URL}/api/admin/refresh-data")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error refreshing data: {str(e)}")
        return None

# =====================================================
# NEW: SCHEDULER API FUNCTIONS
# =====================================================

def get_scheduler_status():
    """Fetch scheduler status from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/scheduler/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching scheduler status: {str(e)}")
        return None

def trigger_scheduler():
    """Manually trigger scheduler data refresh"""
    try:
        response = requests.post(f"{API_BASE_URL}/api/scheduler/trigger", timeout=5)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 409:
            return {"status": "error", "message": "Pipeline already running"}
        return None
    except Exception as e:
        st.error(f"Error triggering scheduler: {str(e)}")
        return None

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🏦 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "👤 Customer 360 View", "📊 Recommendations", "⚙️ Admin Dashboard"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **Financial Recommendation System**
    
    Version 2.0.0
    
    Powered by FastAPI & Streamlit
    
    🤖 AI-Powered Explanations
    ⏰ Automated Scheduler
""")

# =====================================================
# PAGE 1: HOME
# =====================================================

if page == "🏠 Home":
    st.markdown('<div class="main-header">💰 Financial Recommendation System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Financial Recommendation Platform
    
    This intelligent system provides personalized investment recommendations based on:
    - **Risk Profiling**: Advanced ML-based risk assessment
    - **Financial Health Analysis**: Comprehensive financial metrics
    - **Personalized Recommendations**: AI-powered stock and mutual fund suggestions
    - **🤖 Individual AI Explanations**: Get AI insights for each recommendation
    - **⏰ Automated Data Refresh**: Scheduled updates for stocks and mutual funds
    """)
    
    stats = get_stats()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h2>{stats['users_loaded']}</h2>
                    <p>Total Users</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h2>{stats['stocks_loaded']}</h2>
                    <p>Stocks Available</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h2>{stats['funds_loaded']}</h2>
                    <p>Mutual Funds</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            model_status = "✅ Loaded" if stats['model_loaded'] else "❌ Not Loaded"
            st.markdown(f"""
                <div class="metric-card">
                    <h2>{model_status}</h2>
                    <p>ML Model</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Scheduler Status on Home Page
    st.subheader("⏰ Automated Scheduler Status")
    
    scheduler_status = get_scheduler_status()
    
    if scheduler_status:
        enabled = scheduler_status.get('enabled', False)
        is_running = scheduler_status.get('is_running', False)
        last_run = scheduler_status.get('last_run')
        next_run = scheduler_status.get('next_run')
        
        if not enabled:
            status_class = "scheduler-disabled"
            status_icon = "🔴"
            status_text = "DISABLED"
        elif is_running:
            status_class = "scheduler-running"
            status_icon = "🟡"
            status_text = "RUNNING"
        else:
            status_class = "scheduler-idle"
            status_icon = "🟢"
            status_text = "IDLE"
        
        st.markdown(f"""
            <div class="scheduler-card {status_class}">
                <h2>{status_icon} Scheduler: {status_text}</h2>
                <p><strong>Last Run:</strong> {last_run if last_run else 'Never'}</p>
                <p><strong>Next Run:</strong> {next_run if next_run else 'Not scheduled'}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Unable to fetch scheduler status")
    
    st.markdown("---")
    
    # LLM Status on Home Page
    st.subheader("🤖 AI Assistant Status")
    
    llm_health = check_llm_health()
    
    if llm_health and llm_health.get('status') == 'online':
        st.success(f"✅ AI Assistant is Online - Model: {llm_health.get('configured_model', 'N/A')}")
        st.info(f"📚 Available Models: {', '.join(llm_health.get('available_models', []))}")
    else:
        st.warning("⚠️ AI Assistant is Offline - Fallback explanations will be used")
        with st.expander("🔧 How to enable AI Assistant"):
            st.markdown("""
            **Setup Ollama:**
            1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
            2. Pull model: `ollama pull llama3.1`
            3. Verify: `curl http://localhost:11434/api/tags`
            """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 Quick Start Guide
    
    1. **Customer 360 View**: Select a user to view their complete financial profile
    2. **Recommendations**: Get personalized recommendations with automatic AI explanations
    3. **Admin Dashboard**: Monitor system logs, LLM status, and scheduler
    
    Navigate using the sidebar to get started!
    """)

# =====================================================
# PAGE 2: CUSTOMER 360 VIEW
# =====================================================

elif page == "👤 Customer 360 View":
    st.markdown('<div class="main-header">👤 Customer 360° View</div>', unsafe_allow_html=True)
    
    users_data = get_all_users()
    
    if users_data and users_data.get('users'):
        users_list = users_data['users']
        user_ids = [user['customer_id'] for user in users_list]
        
        selected_user_id = st.selectbox("🔍 Select Customer ID", user_ids)
        
        if st.button("🔄 Load Customer Profile", key="load_profile"):
            with st.spinner("Loading customer profile..."):
                user_profile = get_user_profile(selected_user_id)
                
                if user_profile:
                    st.success(f"✅ Profile loaded for {selected_user_id}")
                    st.session_state.current_profile = user_profile
        
        if 'current_profile' in st.session_state:
            profile = st.session_state.current_profile
            metadata = profile['metadata']
            derived = profile['derived_features']
            
            st.markdown("---")
            
            st.subheader("📋 Basic Information")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Age", f"{metadata['age']} years")
            with col2:
                st.metric("Gender", metadata['gender'])
            with col3:
                st.metric("Occupation", metadata['occupation'])
            with col4:
                st.metric("Credit Score", metadata['credit_score'])
            
            st.markdown("---")
            
            st.subheader("⚠️ Risk Profile")
            
            risk_label = metadata['risk_label'].upper()
            risk_score = metadata['risk_score']
            
            risk_class = f"risk-{metadata['risk_label'].lower()}"
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h1>{risk_label}</h1>
                        <p>Risk Category</p>
                        <h3>Score: {risk_score}</h3>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    title={'text': "Risk Score (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "yellow"},
                            {'range': [66, 100], 'color': "salmon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_score * 100
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("💰 Financial Health")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Annual Income", f"₹{metadata['income']:,.0f}")
                st.metric("Savings Rate", f"{metadata['savings_rate']*100:.1f}%")
            
            with col2:
                st.metric("Debt-to-Income", f"{metadata['debt_to_income']*100:.1f}%")
                st.metric("Credit Utilization", f"{derived['credit_utilization']*100:.1f}%")
            
            with col3:
                st.metric("Digital Activity", f"{metadata['digital_activity']:.0f}/100")
                st.metric("Portfolio Diversity", f"{metadata['portfolio_diversity']:.0f}/100")
            
            st.markdown("---")
            st.subheader("📈 Financial Metrics Breakdown")
            
            metrics_data = {
                'Metric': ['Savings Rate', 'Debt-to-Income', 'Credit Utilization', 
                          'Spending Stability', 'Transaction Volatility'],
                'Value': [
                    metadata['savings_rate'] * 100,
                    metadata['debt_to_income'] * 100,
                    derived['credit_utilization'] * 100,
                    derived['spending_stability'] * 100,
                    derived['transaction_volatility'] * 100
                ]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            fig = px.bar(df_metrics, x='Metric', y='Value', 
                        title='Financial Health Indicators (%)',
                        color='Value',
                        color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("📊 Past Investment Profile")
            
            base_user = next((u for u in users_list if u['customer_id'] == selected_user_id), None)
            
            if base_user:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Investment Amount (Last Year)", 
                             f"₹{base_user['investmentamountlastyear']:,.0f}")
                
                with col2:
                    st.metric("Past Investments", base_user['pastinvestments'])
                
                if base_user['investmentamountlastyear'] > 0:
                    investment_type = base_user.get('pastinvestments', '')
                    if investment_type:
                        types = investment_type.replace('_', ', ').split(', ')
                        st.info(f"**Investment Categories**: {', '.join(types)}")
    
    else:
        st.error("⚠️ Unable to load users. Please check if the API is running.")
        st.info("Start the FastAPI server with: `python main.py`")

# =====================================================
# PAGE 3: RECOMMENDATIONS (keeping existing code)
# =====================================================

elif page == "📊 Recommendations":
    st.markdown('<div class="main-header">📊 Investment Recommendations</div>', unsafe_allow_html=True)
    
    users_data = get_all_users()
    
    if users_data and users_data.get('users'):
        users_list = users_data['users']
        user_ids = [user['customer_id'] for user in users_list]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_user_id = st.selectbox("🔍 Select Customer ID", user_ids)
        
        with col2:
            top_k = st.number_input("Top K", min_value=5, max_value=20, value=10)
        
        if st.button("🚀 Generate Recommendations", key="gen_rec"):
            with st.spinner("🤖 Generating personalized recommendations with AI explanations..."):
                recommendations = get_recommendations(selected_user_id, top_k)
                
                if recommendations:
                    st.success(f"✅ Recommendations generated for {selected_user_id}")
                    st.session_state.recommendations = recommendations
                    st.session_state.stock_explanations = {}
                    st.session_state.fund_explanations = {}
                    st.session_state.insurance_explanations = {}
                    st.rerun()
        
        if 'recommendations' in st.session_state:
            recs = st.session_state.recommendations
            user_meta = recs['user_metadata']
            
            if 'stock_explanations' not in st.session_state:
                st.session_state.stock_explanations = {}
            if 'fund_explanations' not in st.session_state:
                st.session_state.fund_explanations = {}
            if 'insurance_explanations' not in st.session_state:
                st.session_state.insurance_explanations = {}
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Risk Profile", user_meta['risk_label'].upper())
            with col2:
                st.metric("Annual Income", f"₹{user_meta['income']:,.0f}")
            with col3:
                st.metric("Credit Score", user_meta['credit_score'])
            with col4:
                st.metric("Savings Rate", f"{user_meta['savings_rate']*100:.1f}%")
            
            st.markdown("---")
            
            # STOCKS SECTION
            st.subheader("📈 Top Stock Recommendations")
            
            stocks = recs['top_stock_recommendations']
            
            for i, stock in enumerate(stocks[:5], 1):
                stock_key = f"stock_{i}"
                
                with st.container():
                    st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown(f"### {i}. {stock['symbol']}")
                        st.markdown(f"**{stock['company_name']}**")
                    
                    with col2:
                        sector = stock['metadata'].get('sector', 'N/A')
                        st.markdown(f"**Sector**: {sector}")
                        
                        market_cap = stock['metadata'].get('market_cap', 'N/A')
                        st.markdown(f"**Market Cap**: {market_cap}")
                    
                    with col3:
                        similarity = stock['similarity_score']
                        st.metric("Match Score", f"{similarity*100:.1f}%")
                    
                    st.progress(similarity)
                    
                    if stock_key not in st.session_state.stock_explanations:
                        with st.spinner(f"🤖 Generating AI insight for {stock['symbol']}..."):
                            explanation = get_individual_explanation(user_meta, "stock", stock)
                            
                            if explanation and explanation.get('explanation'):
                                st.session_state.stock_explanations[stock_key] = explanation['explanation']
                            else:
                                sector = stock['metadata'].get('sector', 'diversified')
                                symbol = stock['symbol']
                                risk_level = user_meta['risk_label']
                                score = similarity * 100
                                
                                if i == 1:
                                    fallback = f"""🏆 Top recommendation! {symbol} ({sector} sector) shows the highest compatibility at {score:.1f}% with your {risk_level} risk profile."""
                                else:
                                    fallback = f"""Strong match! {symbol} from {sector} sector shows {score:.1f}% compatibility with your investment profile."""
                                
                                st.session_state.stock_explanations[stock_key] = fallback
                    
                    if stock_key in st.session_state.stock_explanations:
                        st.markdown(f"""
                        <div class="llm-explanation">
                            <div class="explanation-title">💡 Why this recommendation?</div>
                            {st.session_state.stock_explanations[stock_key]}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("")
            
            st.markdown("---")
            
            # MUTUAL FUNDS SECTION
            st.subheader("💼 Top Mutual Fund Recommendations")
            
            funds = recs['top_mutual_fund_recommendations']
            
            for i, fund in enumerate(funds[:5], 1):
                fund_key = f"fund_{i}"
                
                with st.container():
                    st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"### {i}. {fund['fund_name']}")
                        
                        category = fund['metadata'].get('category', 'N/A')
                        st.markdown(f"**Category**: {category}")
                        
                        if 'aum' in fund['metadata']:
                            st.markdown(f"**AUM**: ₹{fund['metadata']['aum']:,.2f} Cr")
                    
                    with col2:
                        similarity = fund['similarity_score']
                        st.metric("Match Score", f"{similarity*100:.1f}%")
                    
                    st.progress(similarity)
                    
                    if fund.get('fund_link'):
                        st.markdown(f"[🔗 More Info]({fund['fund_link']})")
                    
                    if fund_key not in st.session_state.fund_explanations:
                        with st.spinner(f"🤖 Generating AI insight..."):
                            explanation = get_individual_explanation(user_meta, "mutual_fund", fund)
                            
                            if explanation and explanation.get('explanation'):
                                st.session_state.fund_explanations[fund_key] = explanation['explanation']
                            else:
                                category = fund['metadata'].get('category', 'diversified')
                                risk_level = user_meta['risk_label']
                                score = similarity * 100
                                
                                if i == 1:
                                    fallback = f"""🏆 Best match! This {category} fund leads with {score:.1f}% compatibility."""
                                else:
                                    fallback = f"""Excellent option! This {category} fund offers {score:.1f}% alignment."""
                                
                                st.session_state.fund_explanations[fund_key] = fallback
                    
                    if fund_key in st.session_state.fund_explanations:
                        st.markdown(f"""
                        <div class="llm-explanation">
                            <div class="explanation-title">💡 Why this recommendation?</div>
                            {st.session_state.fund_explanations[fund_key]}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("")
            
            st.markdown("---")
            
            #  INSURANCE SECTION WITH INDIVIDUAL EXPLANATIONS
            st.subheader("🛡️ Top Insurance Recommendations")
            
            insurances = recs.get('top_insurance_recommendations', [])
            
            if insurances:
                for i, insurance in enumerate(insurances[:5], 1):
                    insurance_key = f"insurance_{i}"
                    
                    with st.container():
                        st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.markdown(f"### {i}. {insurance['insurance_name']}")
                            insurance_type = insurance.get('insurance_type', 'N/A')
                            insurer = insurance['metadata'].get('insurer', 'N/A')
                            st.markdown(f"**Type**: {insurance_type}")
                            st.markdown(f"**Insurer**: {insurer}")
                        
                        with col2:
                            premium_range = insurance['metadata'].get('premium_range', 'N/A')
                            sum_insured = insurance['metadata'].get('sum_insured', 'N/A')
                            
                            st.markdown(f"**Premium**: {premium_range}")
                            st.markdown(f"**Sum Insured**: {sum_insured}")
                            
                            # Show key features
                            if insurance['metadata'].get('opd_cover'):
                                st.markdown("✅ OPD Cover")
                            if insurance['metadata'].get('maternity_cover'):
                                st.markdown("✅ Maternity Cover")
                        
                        with col3:
                            similarity = insurance['similarity_score']
                            st.metric("Match Score", f"{similarity*100:.1f}%")
                        
                        st.progress(similarity)
                        
                        # Display policy URL if available
                        policy_url = insurance['metadata'].get('url', '')
                        if policy_url:
                            st.markdown(f"[🔗 View Policy Details]({policy_url})")
                        
                        # Auto-generate explanation if not already generated
                        if insurance_key not in st.session_state.insurance_explanations:
                            with st.spinner(f"🤖 Generating AI insight for this insurance..."):
                                explanation = get_individual_explanation(
                                    user_meta,
                                    "insurance",
                                    insurance
                                )
                                
                                if explanation and explanation.get('explanation'):
                                    st.session_state.insurance_explanations[insurance_key] = explanation['explanation']
                                else:
                                    # Enhanced diverse fallback explanations for insurance
                                    insurance_type = insurance.get('insurance_type', 'comprehensive')
                                    insurance_name = insurance['insurance_name']
                                    insurer = insurance['metadata'].get('insurer', 'leading insurer')
                                    risk_level = user_meta['risk_label']
                                    score = similarity * 100
                                    
                                    # Get key features
                                    has_opd = insurance['metadata'].get('opd_cover', False)
                                    has_maternity = insurance['metadata'].get('maternity_cover', False)
                                    has_critical = insurance['metadata'].get('critical_illness_cover', False)
                                    covers_preexisting = insurance['metadata'].get('covers_preexisting', False)
                                    
                                    # Different explanation templates based on rank
                                    if i == 1:
                                        fallback = f"""🏆 Perfect protection match! **{insurance_name}** by {insurer} shows {score:.1f}% compatibility with your financial profile. 
                                        Your annual income of ₹{user_meta['income']:,.0f} comfortably supports the premium, ensuring long-term affordability. 
                                        With your credit score of {user_meta['credit_score']}, you qualify for favorable terms. This comprehensive policy includes {'OPD cover, ' if has_opd else ''}{'maternity benefits, ' if has_maternity else ''}{'critical illness protection, ' if has_critical else ''} providing complete health security."""
                                    elif i == 2:
                                        fallback = f"""Essential coverage! **{insurance_name}** by {insurer} aligns {score:.1f}% with your {risk_level} risk profile. 
                                        At your income level and with {user_meta['debt_to_income']*100:.1f}% debt-to-income ratio, the premium is manageable within your budget. 
                                        {'This policy covers pre-existing conditions, ' if covers_preexisting else ''}making it an excellent choice for comprehensive health protection for your family."""
                                    elif i == 3:
                                        fallback = f"""Smart protection strategy! **{insurance_name}** from {insurer} offers {score:.1f}% match with your needs. 
                                        Your savings rate of {user_meta['savings_rate']*100:.1f}% indicates you can allocate funds for this important protection. 
                                        The policy's {'OPD and ' if has_opd else ''}{'maternity ' if has_maternity else ''}coverage complements your investment strategy by providing security against medical emergencies."""
                                    elif i == 4:
                                        fallback = f"""Valuable safety net! **{insurance_name}** by {insurer} shows {score:.1f}% compatibility with your financial situation. 
                                        Given your stable income and low debt burden, this insurance enhances your overall financial resilience. 
                                        With features like {'critical illness cover' if has_critical else 'comprehensive health benefits'}, it's well-suited to protect your dependents and assets."""
                                    else:
                                        fallback = f"""Comprehensive protection! **{insurance_name}** from {insurer} provides {score:.1f}% alignment with your profile. 
                                        Your financial stability supports this premium commitment while maintaining your investment capacity. 
                                        This policy addresses specific protection needs identified based on your {risk_level} risk profile and life stage."""
                                    
                                    st.session_state.insurance_explanations[insurance_key] = fallback
                        
                        # Display explanation
                        if insurance_key in st.session_state.insurance_explanations:
                            st.markdown(f"""
                            <div class="llm-explanation">
                                <div class="explanation-title">💡 Why this recommendation?</div>
                                {st.session_state.insurance_explanations[insurance_key]}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("")  # spacing
            else:
                st.info("ℹ️ No insurance recommendations available for this user.")
    
    else:
        st.error("⚠️ Unable to load users. Please check if the API is running.")

# =====================================================
# PAGE 4: ADMIN DASHBOARD
# =====================================================

elif page == "⚙️ Admin Dashboard":
    st.markdown('<div class="main-header">⚙️ Admin Dashboard</div>', unsafe_allow_html=True)
    
    st.subheader("📊 System Statistics")
    
    stats = get_stats()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", stats['users_loaded'])
        with col2:
            st.metric("Stocks Available", stats['stocks_loaded'])
        with col3:
            st.metric("Mutual Funds", stats['funds_loaded'])
        with col4:
            model_status = "Loaded ✅" if stats['model_loaded'] else "Not Loaded ❌"
            st.metric("ML Model", model_status)
    
    st.markdown("---")
    
    # LLM HEALTH STATUS
    st.subheader("🤖 LLM Assistant Status")
    
    llm_health = check_llm_health()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if llm_health:
            status = llm_health.get('status', 'unknown')
            if status == 'online':
                st.success("🟢 LLM Online")
            elif status == 'offline':
                st.error("🔴 LLM Offline")
            else:
                st.warning("🟡 LLM Unknown")
        else:
            st.error("🔴 Cannot Connect")
    
    with col2:
        if llm_health and llm_health.get('status') == 'online':
            model = llm_health.get('configured_model', 'N/A')
            st.metric("Active Model", model)
        else:
            st.metric("Active Model", "N/A")
    
    with col3:
        if llm_health and llm_health.get('status') == 'online':
            models = llm_health.get('available_models', [])
            st.metric("Available Models", len(models))
        else:
            st.metric("Available Models", "0")
    
    if llm_health and llm_health.get('status') == 'online':
        with st.expander("📋 Available Ollama Models"):
            models = llm_health.get('available_models', [])
            if models:
                for model in models:
                    st.text(f"• {model}")
            else:
                st.warning("No models found")
    
    with st.expander("⚙️ LLM Setup Instructions"):
        st.markdown("""
        **To enable AI explanations, ensure Ollama is running:**
        
        1. Install Ollama: https://ollama.ai/
        2. Pull the model:
           ```bash
           ollama pull llama3.1
           ```
        3. Start Ollama (it runs automatically after installation)
        4. Verify it's running:
           ```bash
           curl http://localhost:11434/api/tags
           ```
        
        **Troubleshooting:**
        - If LLM is offline, restart Ollama service
        - Check if port 11434 is available
        - Ensure you have sufficient RAM (8GB+ recommended)
        """)
    
    st.markdown("---")
    
    st.subheader("🔄 Data Refresh")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("Refresh system data from files (users, stocks, mutual funds)")
    
    with col2:
        if st.button("🔄 Refresh Data", key="refresh"):
            with st.spinner("Refreshing data..."):
                result = refresh_data()
                if result:
                    st.success("✅ Data refreshed successfully!")
                else:
                    st.error("❌ Failed to refresh data")
    
    st.markdown("---")
    
    st.subheader("📋 System Logs")
    
    logs_data = get_admin_logs()
    
    if logs_data and logs_data.get('logs'):
        logs = logs_data['logs']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            action_filter = st.multiselect(
                "Filter by Action",
                options=list(set([log['action'] for log in logs])),
                default=None
            )
        
        with col2:
            status_filter = st.multiselect(
                "Filter by Status",
                options=list(set([log['status'] for log in logs])),
                default=None
            )
        
        with col3:
            limit = st.number_input("Show last N logs", min_value=10, max_value=100, value=20)
        
        filtered_logs = logs[-limit:]
        
        if action_filter:
            filtered_logs = [log for log in filtered_logs if log['action'] in action_filter]
        
        if status_filter:
            filtered_logs = [log for log in filtered_logs if log['status'] in status_filter]
        
        logs_df = pd.DataFrame(filtered_logs)
        
        if not logs_df.empty:
            st.dataframe(
                logs_df[['timestamp', 'action', 'status', 'details']],
                width='stretch',
                height=400
            )
            
            st.markdown("---")
            st.subheader("📈 Log Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                status_counts = logs_df['status'].value_counts()
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title='Status Distribution'
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                action_counts = logs_df['action'].value_counts()
                fig = px.bar(
                    x=action_counts.index,
                    y=action_counts.values,
                    title='Actions Performed',
                    labels={'x': 'Action', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No logs match the selected filters")
    
    else:
        st.warning("⚠️ No logs available")

# =====================================================
# FOOTER
# =====================================================

# st.markdown("---")
# st

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; padding: 1rem;">
        <p>Financial Recommendation System v1.0.0</p>
        <p>Built with ❤️ using FastAPI & Streamlit</p>
        <p>🤖 Powered by Ollama LLM</p>
    </div>
""", unsafe_allow_html=True)
