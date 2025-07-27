import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
df = pd.read_csv("players_stats.csv")
df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

# Clean percentage columns
df['Headshot %'] = df['Headshot %'].str.replace('%', '').astype(float)
df['Clutch Success %'] = df['Clutch Success %'].str.replace('%', '').astype(float)
df['Kill, Assist, Trade, Survive %'] = df['Kill, Assist, Trade, Survive %'].str.replace('%', '').astype(float)


# Define feature columns
feature_cols = [
    'Average Combat Score',
    'Kills:Deaths',
    'Kill, Assist, Trade, Survive %',
    'Average Damage Per Round',
    'Kills Per Round',
    'Assists Per Round',
    'First Kills',
    'First Deaths',
    'Headshot %',
    
]

# Drop rows with missing values in feature columns
df.dropna(subset=feature_cols, inplace=True)


# Streamlit UI
st.title("🎯 Valorant Pro Match Finder")
st.markdown("Enter your stats below and find the pro players most similar to you.")

with st.sidebar:
    st.header("Your Player Stats")
    acs = st.slider("Average Combat Score", 100, 400, 200)
    kd = st.slider('K/D',0,2,1)
    kast = st.slider('KAST%',0,100,50)
    adr = st.slider('Damage/round',0,300,100)
    kpr = st.slider("Kills/Round", 0.3, 2.0, 0.7)
    apr = st.slider("Assists/Round", 0.0, 2.0, 0.25)
    fk = st.slider("FK", 0, 100, 10)
    fd = st.slider("FD", 0, 100, 2)
    hs_percent = st.slider("Headshot %", 5, 50, 25)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

user_input = [[
     acs, kd, kast, adr, kpr, apr, fk, fd, hs_percent,
]]
user_scaled = scaler.transform(user_input)

# Compute similarity
df["Similarity"] = cosine_similarity(user_scaled, X_scaled)[0]



top_matches = df.sort_values(by="Similarity", ascending=False).head(3)
st.subheader("🎖️ Top 3 Pro Player Matches")
st.table(top_matches[["Player", "Similarity", "Rounds Played"]].round(3))
