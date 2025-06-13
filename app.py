import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai',
          'Kolkata', 'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai',
          'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
          'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
          'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi',
          'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah',
          'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))

# Enhanced Title with styling
st.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h1 style='color: #2E86AB; font-size: 3rem; margin-bottom: 0;'>🏏 IPL Win Predictor</h1>
    <p style='color: #6c757d; font-size: 1.1rem; margin-top: 0;'>Real-time cricket match outcome prediction</p>
</div>
""", unsafe_allow_html=True)

st.subheader("⚡ Team Selection")
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('🏏 Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('🎯 Bowling Team', sorted(teams))

st.subheader("🏟️ Match Setup")
col1, col2 = st.columns(2)
with col1:
    selected_city = st.selectbox('📍 Host City', sorted(cities))
with col2:
    target = st.number_input('🎯 Target Score', min_value=1, value=150)

st.subheader("📊 Current State")
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('⚡ Score', min_value=0, value=75)
with col4:
    overs = st.number_input('⏱️ Overs', min_value=0.0, max_value=19.9, value=10.0, step=0.1)
with col5:
    wickets = st.number_input('💥 Wickets', min_value=0, max_value=10, value=3)

if st.button('🔮 Predict Match Outcome', type="primary"):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'crr':[crr], 'rrr': [rrr]})

    # Hide the data table in an expander (optional to view)
    with st.expander("📊 View Match Data", expanded=False):
        st.dataframe(input_df, use_container_width=True)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    
    # Enhanced Results Display
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label=f"🏆 {batting_team}",
            value=f"{round(win*100)}%",
            delta=f"{round(win*100 - 50)}% vs 50-50"
        )
    
    with col2:
        st.metric(
            label=f"🛡️ {bowling_team}",
            value=f"{round(loss*100)}%",
            delta=f"{round(loss*100 - 50)}% vs 50-50"
        )
    
    # Visual Progress Indicators
    st.subheader("Match Situation")
    st.progress(win, text=f"{batting_team} Win Probability")
    st.progress(loss, text=f"{bowling_team} Win Probability")
    
    # Quick Match Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Runs Needed:** {runs_left}")
    with col2:
        st.info(f"**Required Rate:** {rrr:.1f}")
    with col3:
        st.info(f"**Wickets Left:** {wickets}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0 1rem 0; color: #6c757d;'>
    <p style='margin-bottom: 1rem; font-size: 0.9rem;'>Built with ❤️ by <strong>Shivam Dali</strong></p>
    <div style='display: flex; justify-content: center; gap: 2rem;'>
        <a href='https://github.com/svdexe' target='_blank' style='text-decoration: none; color: #333; font-weight: 500;'>
            🔗 GitHub
        </a>
        <a href='https://www.linkedin.com/in/shivam-dali-86b0a1201/' target='_blank' style='text-decoration: none; color: #0077b5; font-weight: 500;'>
            💼 LinkedIn
        </a>
    </div>
    <p style='margin-top: 1rem; font-size: 0.8rem; color: #95a5a6;'>
        © 2024 IPL Win Predictor | Machine Learning Powered
    </p>
</div>
""", unsafe_allow_html=True)