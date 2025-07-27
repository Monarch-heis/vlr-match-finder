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



# Define feature columns
feature_cols = [
    'Rating',
    'Average Combat Score',
    'Kills Per Round',
    'Assists Per Round',
    'First Kills Per Round',
    'First Deaths Per Round',
    'Headshot %',
    'Clutch Success %'
]

# Drop rows with missing values in feature columns
df.dropna(subset=feature_cols, inplace=True)

st.write("DataFrame columns:", df.columns.tolist())
# Streamlit UI
st.title("🎯 Valorant Pro Match Finder")
st.markdown("Enter your stats below and find the pro players most similar to you.")

with st.sidebar:
    st.header("Your Player Stats")
    rating = st.slider("Rating", 0.5, 2.0, 1.0, 0.01)
    acs = st.slider("Average Combat Score", 100, 400, 200)
    kpr = st.slider("Kills Per Round", 0.3, 1.2, 0.7)
    apr = st.slider("Assists Per Round", 0.0, 0.7, 0.25)
    fkpr = st.slider("First Kills Per Round", 0.0, 0.5, 0.15)
    fdpr = st.slider("First Deaths Per Round", 0.0, 0.5, 0.10)
    hs_percent = st.slider("Headshot %", 5, 50, 25)
    clutch_percent = st.slider("Clutch Success %", 0, 100, 20)
    min_rounds = st.slider("Minimum Rounds Played", 0, 100, 30)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

user_input = [[
    rating, acs, kpr, apr, fkpr, fdpr, hs_percent, clutch_percent
]]
user_scaled = scaler.transform(user_input)

# Compute similarity
df["Similarity"] = cosine_similarity(user_scaled, X_scaled)[0]

# Filter and show results
if "Rounds Played" in df.columns:
    filtered = df[df["Rounds Played"] >= min_rounds]
else:
    filtered = df

top_matches = filtered.sort_values(by="Similarity", ascending=False).head(3)
st.subheader("🎖️ Top 3 Pro Player Matches")
st.table(top_matches[["Player", "Similarity", "Rounds Played"]].round(3))
