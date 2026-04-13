import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

st.set_page_config(page_title="Movie Investment DSS", page_icon="🎬", layout="wide")

@st.cache_resource(show_spinner="Loading AI Model...")
def load_ai_model():
    model = xgb.XGBRegressor()
    try:
        model.load_model("models/xgboost_box_office_model_without_popularity_and_log.json")
    except Exception as e:
        st.error(f"⚠️ Could not load the model. Ensure 'xgboost_box_office_model_without_popularity.json' exists. Error: {e}")
    return model

@st.cache_data(show_spinner="Loading Historical Data...")
def load_historical_data():
    columns_we_need = ['revenue', 'directors', 'cast']
    try:
        df = pd.read_csv('dataset/TMDB_IMDB_Movies_Dataset.csv', usecols=columns_we_need, low_memory=False)
        df = df.dropna(subset=['revenue'])
        df['primary_director'] = df['directors'].fillna('Unknown').astype(str).str.split(',').str[0].str.strip()
        df['lead_actor'] = df['cast'].fillna('Unknown').astype(str).str.split(',').str[0].str.strip()
        return df
    except FileNotFoundError:
        st.error("⚠️ ERROR: Could not find 'dataset/TMDB_IMDB_Movies_Dataset.csv'. Please ensure it is in the correct folder.")
        return pd.DataFrame()

model = load_ai_model()
db = load_historical_data()

unique_directors = sorted([str(d) for d in db['primary_director'].unique() if str(d) != 'Unknown']) if not db.empty else []
unique_actors = sorted([str(a) for a in db['lead_actor'].unique() if str(a) != 'Unknown']) if not db.empty else []

director_options = ["(Debut / Unknown)"] + unique_directors
actor_options = ["(Debut / Unknown)"] + unique_actors

if not db.empty:
    valid_revs = db[db['revenue'] > 0]['revenue']
    GLOBAL_MEDIAN_REV = valid_revs.median() if not valid_revs.empty else 20000000
else:
    GLOBAL_MEDIAN_REV = 20000000

st.title("🎬 Greenlight: AI Box Office Predictor")
st.markdown("Enter the proposed movie details. The AI will query the database to evaluate the talent and automatically calculate expected revenue.")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Financial & Metadata")
    budget = st.number_input("Production Budget ($)", min_value=1000000, max_value=999999999, value=50000000, step=1000000, format="%d")
    runtime = st.slider("Target Runtime (Minutes)", min_value=80, max_value=200, value=110)
    release_month = st.selectbox("Release Month", list(range(1, 13)), index=6)
    
    all_genres = [
        "Drama", "Comedy", "Action", "Thriller", "Romance", "Adventure", 
        "Crime", "Horror", "Family", "Science Fiction", "Fantasy", "Mystery"
    ]
    # CHANGED: Now uses st.multiselect to allow multiple genres
    genres_selected = st.multiselect("Genres (Select all that apply)", options=all_genres, default=["Action", "Adventure"])

with col2:
    st.subheader("Talent Attachments")
    st.caption("Type names exactly as they appear in the TMDB database (e.g., Christopher Nolan, Tom Holland).")
    
    director_input = st.selectbox("Primary Director Name", options=director_options)
    actor_input = st.selectbox("Lead Actor Name", options=actor_options)

    dir_hist_rev, is_debut_director = GLOBAL_MEDIAN_REV, 1
    actor_hist_rev, is_debut_actor = GLOBAL_MEDIAN_REV, 1

    if director_input != "(Debut / Unknown)" and not db.empty:
        dir_movies = db[db['primary_director'].str.lower() == director_input.lower()]
        valid_dir_movies = dir_movies[dir_movies['revenue'] > 0]
        
        if not valid_dir_movies.empty:
            raw_avg = valid_dir_movies['revenue'].mean()
            if pd.isna(raw_avg) or raw_avg <= 10000:
                dir_hist_rev = GLOBAL_MEDIAN_REV
                is_debut_director = 1  
            else:
                dir_hist_rev = raw_avg
                is_debut_director = 0

    if actor_input != "(Debut / Unknown)" and not db.empty:
        actor_movies = db[db['lead_actor'].str.lower() == actor_input.lower()]
        valid_actor_movies = actor_movies[actor_movies['revenue'] > 0]
        
        if not valid_actor_movies.empty:
            raw_avg = valid_actor_movies['revenue'].mean()
            if pd.isna(raw_avg) or raw_avg <= 10000:
                actor_hist_rev = GLOBAL_MEDIAN_REV
                is_debut_actor = 1
            else:
                actor_hist_rev = raw_avg
                is_debut_actor = 0

    with st.expander("🔍 View AI Internal Talent Assessment", expanded=True):
        st.write(f"**Director Track Record:** {'Debut/Unknown' if is_debut_director else f'${dir_hist_rev:,.0f} Avg'}")
        st.write(f"**Actor Track Record:** {'Debut/Unknown' if is_debut_actor else f'${actor_hist_rev:,.0f} Avg'}")

input_data = {
    'runtime': [runtime],
    'budget': [budget],
    'release_month': [release_month],
    'director_hist_rev': [dir_hist_rev],
    'actor_hist_rev': [actor_hist_rev],
    'is_debut_director': [is_debut_director],
    'is_debut_actor': [is_debut_actor],
    'genre_Drama': [0], 'genre_Comedy': [0], 'genre_Action': [0], 'genre_Thriller': [0],
    'genre_Romance': [0], 'genre_Adventure': [0], 'genre_Crime': [0], 'genre_Horror': [0],
    'genre_Family': [0], 'genre_Science Fiction': [0], 'genre_Fantasy': [0], 'genre_Mystery': [0]
}

# CHANGED: Loop through the list of selected genres and encode each one
for genre in genres_selected:
    genre_key = f"genre_{genre}"
    if genre_key in input_data:
        input_data[genre_key] = [1]

expected_columns = [
    'runtime', 'budget', 'release_month', 'director_hist_rev',
    'actor_hist_rev', 'is_debut_director', 'is_debut_actor', 'genre_Drama',
    'genre_Comedy', 'genre_Action', 'genre_Thriller', 'genre_Romance',
    'genre_Adventure', 'genre_Crime', 'genre_Horror', 'genre_Family',
    'genre_Science Fiction', 'genre_Fantasy', 'genre_Mystery'
]
X_predict = pd.DataFrame(input_data)[expected_columns]

st.divider()

if st.button("🚀 RUN PREDICTIVE ANALYSIS & RISK SIMULATION", type="primary", use_container_width=True):
    with st.status("Initializing AI Systems...", expanded=True) as status:
        try:
            st.write("Running baseline XGBoost prediction...")
            log_prediction = model.predict(X_predict)[0]
            prediction = np.expm1(log_prediction)
            base_revenue = max(prediction, 0) 
            base_roi = ((base_revenue - budget) / budget) * 100
            
            st.write("Executing 1,000 Monte Carlo Simulations...")
            n_simulations = 1000
            X_sim = pd.concat([X_predict] * n_simulations, ignore_index=True)
            
            simulated_budgets = np.random.normal(loc=budget, scale=budget * 0.10, size=n_simulations)
            X_sim['budget'] = np.clip(simulated_budgets, a_min=100000, a_max=None)
            
            log_simulated_revenues = model.predict(X_sim)
            simulated_revenues = np.expm1(log_simulated_revenues)
            simulated_revenues = np.maximum(simulated_revenues, 0)
            
            simulated_rois = ((simulated_revenues - X_sim['budget']) / X_sim['budget']) * 100
            
            st.write("Calculating Risk Metrics...")
            prob_success = (simulated_rois > 0).mean() * 100 
            worst_case_rev = np.percentile(simulated_revenues, 5)
            best_case_rev = np.percentile(simulated_revenues, 95)
            
            status.update(label="Analysis & Simulation Complete!", state="complete", expanded=False)
            st.toast("Simulation finished successfully!", icon="✅")
            
            st.subheader("📊 1. The Baseline Forecast")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Predicted Global Box Office", f"${base_revenue:,.0f}")
            res_col2.metric("Projected Baseline ROI", f"{base_roi:.1f}%")
            
            if prob_success >= 60:
                res_col3.success("🟢 RECOMMENDATION: GREENLIGHT")
            elif 40 <= prob_success < 60:
                res_col3.warning("🟡 RECOMMENDATION: REVISE BUDGET")
            else:
                res_col3.error("🔴 RECOMMENDATION: PASS")
                
            st.divider()
            
            st.subheader("🎲 2. Monte Carlo Risk Analysis (1,000 Simulations)")
            st.markdown("We simulated this movie's release 1,000 times, injecting random real-world budget overruns to find your true probability of profitability.")
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            risk_col1.metric("Worst Case Scenario (Bottom 5%)", f"${worst_case_rev:,.0f}")
            risk_col2.metric("Best Case Scenario (Top 5%)", f"${best_case_rev:,.0f}")
            risk_col3.metric("Probability of Profitability", f"{prob_success:.1f}%")
            
            st.markdown("### Box Office Probability Curve")
            
            counts, bins = np.histogram(simulated_revenues, bins=30)
            
            chart_data = pd.DataFrame({
                "Revenue Scenario ($)": bins[:-1],
                "Frequency": counts
            })
            
            st.bar_chart(chart_data, x="Revenue Scenario ($)", y="Frequency", color="#1f77b4")
                
        except Exception as e:
            status.update(label="Prediction Error", state="error", expanded=True)
            st.error(f"Error Details: {e}")
