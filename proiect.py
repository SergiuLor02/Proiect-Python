import pandas as pd
import plotly.express as px
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Încarcă fișierul CSV
df = pd.read_csv("food_recipes.csv")

# Elimină rândurile cu date lipsă esențiale
df.dropna(subset=['description', 'cuisine', 'course', 'diet', 'prep_time', 'cook_time', 'tags'], inplace=True)

# Conversie timp din format string în minute
def time_to_minutes(time_str):
    if pd.isnull(time_str):
        return np.nan
    total_minutes = 0
    hours = re.search(r'(\d+)\s*H', time_str)
    minutes = re.search(r'(\d+)\s*M', time_str)
    if hours:
        total_minutes += int(hours.group(1)) * 60
    if minutes:
        total_minutes += int(minutes.group(1))
    return total_minutes

df['prep_time_mins'] = df['prep_time'].apply(time_to_minutes)
df['cook_time_mins'] = df['cook_time'].apply(time_to_minutes)

# Elimină extremele din vote_count și rating
vc_low, vc_high = df['vote_count'].quantile([0.01, 0.99])
rating_low, rating_high = df['rating'].quantile([0.01, 0.99])
df = df[(df['vote_count'].between(vc_low, vc_high)) & (df['rating'].between(rating_low, rating_high))]

# Selecție de coloane utile
df_model = df[['rating', 'vote_count', 'cuisine', 'course', 'diet', 'prep_time_mins', 'cook_time_mins', 'category']].copy()

# 1. Adăugare timp total
df_model['total_time_mins'] = df_model['prep_time_mins'] + df_model['cook_time_mins']

# 2. Top 10 categorii după rating mediu
top_categorii = df_model.groupby("category")["rating"].mean().sort_values(ascending=False).head(10)
print("Top 10 categorii după rating mediu:")
print(top_categorii)

# 3. Vizualizare timp total vs rating, mărime după număr voturi
fig6 = px.scatter(
    df_model, x="total_time_mins", y="rating", size="vote_count",
    title="Timp total vs. Rating (dimensiune = număr voturi)",
    color="course", hover_data=["cuisine", "diet"]
)
fig6.show()

# 4. Histogramă rețete pe dietă și tip de masă
fig7 = px.histogram(df_model, x="diet", color="course", barmode="group",
                    title="Număr de rețete pe dietă și tipul mesei")
fig7.show()

# 5. Normalizare vote_count
scaler = MinMaxScaler()
df_model['vote_count_scaled'] = scaler.fit_transform(df_model[['vote_count']])

# Vizualizări originale
fig1 = px.histogram(df_model, x="rating", nbins=30, title="Distribuția ratingurilor", color_discrete_sequence=['indianred'])
fig2 = px.box(df_model, x="diet", y="rating", title="Ratinguri în funcție de dietă", color="diet")
fig3 = px.box(df_model, x="course", y="rating", title="Ratinguri în funcție de tipul mesei", color="course")
fig4 = px.scatter(df_model, x="prep_time_mins", y="rating", title="Timp de pregătire vs. Rating", color_discrete_sequence=["darkgreen"])
fig5 = px.scatter(df_model, x="cook_time_mins", y="rating", title="Timp de gătire vs. Rating", color_discrete_sequence=["darkblue"])

fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.show()
