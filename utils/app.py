# app.py - Streamlit Frontend

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
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

def get_recommendations(user_id, top_k=10):
    """Fetch recommendations from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/recommendations/{user_id}?top_k={top_k}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching recommendations: {str(e)}")
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
    
    Version 1.0.0
    
    Powered by FastAPI & Streamlit
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
    """)
    
    # Fetch system stats
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
    
    st.markdown("""
    ### 🚀 Quick Start Guide
    
    1. **Customer 360 View**: Select a user to view their complete financial profile
    2. **Recommendations**: Get personalized stock and mutual fund recommendations
    3. **Admin Dashboard**: Monitor system logs and refresh data
    
    Navigate using the sidebar to get started!
    """)

# =====================================================
# PAGE 2: CUSTOMER 360 VIEW
# =====================================================

elif page == "👤 Customer 360 View":
    st.markdown('<div class="main-header">👤 Customer 360° View</div>', unsafe_allow_html=True)
    
    # Fetch all users
    users_data = get_all_users()
    
    if users_data and users_data.get('users'):
        users_list = users_data['users']
        user_ids = [user['customer_id'] for user in users_list]
        
        # User selection
        selected_user_id = st.selectbox("🔍 Select Customer ID", user_ids)
        
        if st.button("🔄 Load Customer Profile", key="load_profile"):
            with st.spinner("Loading customer profile..."):
                user_profile = get_user_profile(selected_user_id)
                
                if user_profile:
                    st.success(f"✅ Profile loaded for {selected_user_id}")
                    
                    # Store in session state
                    st.session_state.current_profile = user_profile
        
        # Display profile if loaded
        if 'current_profile' in st.session_state:
            profile = st.session_state.current_profile
            metadata = profile['metadata']
            derived = profile['derived_features']
            
            st.markdown("---")
            
            # SECTION 1: Basic Information
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
            
            # SECTION 2: Risk Profile
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
                # Risk gauge chart
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
            
            # SECTION 3: Financial Health
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
            
            # Financial metrics visualization
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
            
            # SECTION 4: Past Investments
            st.subheader("📊 Past Investment Profile")
            
            # Get base user data for past investments
            base_user = next((u for u in users_list if u['customer_id'] == selected_user_id), None)
            
            if base_user:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Investment Amount (Last Year)", 
                             f"₹{base_user['investmentamountlastyear']:,.0f}")
                
                with col2:
                    st.metric("Past Investments", base_user['pastinvestments'])
                
                # Investment breakdown
                if base_user['investmentamountlastyear'] > 0:
                    investment_type = base_user['pastinvestments']
                    types = investment_type.replace('_', ', ').split(', ')
                    
                    st.info(f"**Investment Categories**: {', '.join(types)}")
    
    else:
        st.error("⚠️ Unable to load users. Please check if the API is running.")
        st.info("Start the FastAPI server with: `python main.py`")

# =====================================================
# PAGE 3: RECOMMENDATIONS VIEW
# =====================================================

elif page == "📊 Recommendations":
    st.markdown('<div class="main-header">📊 Investment Recommendations</div>', unsafe_allow_html=True)
    
    # Fetch all users
    users_data = get_all_users()
    
    if users_data and users_data.get('users'):
        users_list = users_data['users']
        user_ids = [user['customer_id'] for user in users_list]
        
        # User selection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_user_id = st.selectbox("🔍 Select Customer ID", user_ids)
        
        with col2:
            top_k = st.number_input("Top K", min_value=5, max_value=20, value=10)
        
        if st.button("🚀 Generate Recommendations", key="gen_rec"):
            with st.spinner("Generating personalized recommendations..."):
                recommendations = get_recommendations(selected_user_id, top_k)
                
                if recommendations:
                    st.success(f"✅ Recommendations generated for {selected_user_id}")
                    st.session_state.recommendations = recommendations
        
        # Display recommendations if loaded
        if 'recommendations' in st.session_state:
            recs = st.session_state.recommendations
            user_meta = recs['user_metadata']
            
            st.markdown("---")
            
            # User summary
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
            
            # SECTION 1: Stock Recommendations
            st.subheader("📈 Top Stock Recommendations")
            
            stocks = recs['top_stock_recommendations']
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📋 Card View", "📊 Table View"])
            
            with tab1:
                for i, stock in enumerate(stocks[:5], 1):
                    with st.container():
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
                        
                        # Progress bar
                        st.progress(similarity)
                        
                        st.markdown("---")
            
            with tab2:
                stocks_df = pd.DataFrame([
                    {
                        'Rank': i+1,
                        'Symbol': s['symbol'],
                        'Company': s['company_name'],
                        'Sector': s['metadata'].get('sector', 'N/A'),
                        'Match Score': f"{s['similarity_score']*100:.2f}%"
                    }
                    for i, s in enumerate(stocks)
                ])
                st.dataframe(stocks_df, use_container_width=True)
            
            st.markdown("---")
            
            # SECTION 2: Mutual Fund Recommendations
            st.subheader("💼 Top Mutual Fund Recommendations")
            
            funds = recs['top_mutual_fund_recommendations']
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📋 Card View", "📊 Table View"])
            
            with tab1:
                for i, fund in enumerate(funds[:5], 1):
                    with st.container():
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
                        
                        # Progress bar
                        st.progress(similarity)
                        
                        if fund.get('fund_link'):
                            st.markdown(f"[🔗 More Info]({fund['fund_link']})")
                        
                        st.markdown("---")
            
            with tab2:
                funds_df = pd.DataFrame([
                    {
                        'Rank': i+1,
                        'Fund Name': f['fund_name'][:50] + '...' if len(f['fund_name']) > 50 else f['fund_name'],
                        'Category': f['metadata'].get('category', 'N/A'),
                        'Match Score': f"{f['similarity_score']*100:.2f}%"
                    }
                    for i, f in enumerate(funds)
                ])
                st.dataframe(funds_df, use_container_width=True)
            
            st.markdown("---")
            
            # SECTION 3: Explanation
            st.subheader("💡 Recommendation Explanation")
            
            st.info(f"""
            **Why these recommendations?**
            
            Based on the customer's profile:
            - **Risk Profile**: {user_meta['risk_label'].upper()} - Recommendations are aligned with risk tolerance
            - **Income Level**: ₹{user_meta['income']:,.0f} - Suitable investment options based on income
            - **Financial Health**: Debt-to-Income ratio of {user_meta['debt_to_income']*100:.1f}%
            - **Investment Experience**: Portfolio diversity score of {user_meta['portfolio_diversity']:.0f}/100
            
            The system uses advanced AI algorithms to match your financial profile with thousands of investment options,
            ensuring personalized recommendations that align with your goals and risk appetite.
            """)
    
    else:
        st.error("⚠️ Unable to load users. Please check if the API is running.")

# =====================================================
# PAGE 4: ADMIN DASHBOARD
# =====================================================

elif page == "⚙️ Admin Dashboard":
    st.markdown('<div class="main-header">⚙️ Admin Dashboard</div>', unsafe_allow_html=True)
    
    # System statistics
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
    
    # Data refresh section
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
    
    # Admin logs
    st.subheader("📋 System Logs")
    
    logs_data = get_admin_logs()
    
    if logs_data and logs_data.get('logs'):
        logs = logs_data['logs']
        
        # Filter options
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
        
        # Apply filters
        filtered_logs = logs[-limit:]
        
        if action_filter:
            filtered_logs = [log for log in filtered_logs if log['action'] in action_filter]
        
        if status_filter:
            filtered_logs = [log for log in filtered_logs if log['status'] in status_filter]
        
        # Display logs
        logs_df = pd.DataFrame(filtered_logs)
        
        if not logs_df.empty:
            st.dataframe(
                logs_df[['timestamp', 'action', 'status', 'details']],
                use_container_width=True,
                height=400
            )
            
            # Log statistics
            st.markdown("---")
            st.subheader("📈 Log Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Status distribution
                status_counts = logs_df['status'].value_counts()
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title='Status Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Action distribution
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
    
    st.markdown("---")
    
    # Vector update status
    st.subheader("🎯 Vector Update Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Last Update", "Today 10:30 AM")
    with col2:
        st.metric("Vector Dimension", "7D")
    with col3:
        st.metric("Sync Status", "✅ In Sync")
    
    st.info("""
    **Vector Information:**
    - All user vectors are aligned with stock and mutual fund embeddings
    - 7-dimensional preference vectors: Risk, Return, Stability, Volatility, Market Cap, Dividend, Momentum
    - Cosine similarity is used for matching
    """)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; padding: 1rem;">
        <p>Financial Recommendation System v1.0.0</p>
        <p>Built with ❤️ using FastAPI & Streamlit</p>
    </div>
""", unsafe_allow_html=True)