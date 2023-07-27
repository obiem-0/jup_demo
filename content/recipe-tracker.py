import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

# Connect to the SQL database
conn = sqlite3.connect('recipes.db')

# Load the recipes data from the SQL database into a pandas DataFrame
recipes_df = pd.read_sql_query('SELECT * FROM recipes', conn)

# Preprocess the data and combine relevant attributes into a single "recipe_features" column
features = ['cuisine', 'ingredients', 'difficulty']
recipes_df['recipe_features'] = recipes_df[features].apply(
    lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Convert the recipe features into a matrix of token counts using CountVectorizer
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(recipes_df['recipe_features'])

# Calculate the cosine similarity between recipes based on their features
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Function to get recipe recommendations based on user preferences
def get_recommendations(recipe_title, cosine_sim=cosine_sim):
    idx = recipes_df[recipes_df['title'] == recipe_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar recipes
    recipe_indices = [i[0] for i in sim_scores]
    return recipes_df['title'].iloc[recipe_indices]

# Example usage:
user_recipe_preference = 'Spicy Chicken Curry'
recommended_recipes = get_recommendations(user_recipe_preference)

print(f"Recommended recipes for '{user_recipe_preference}':")
for recipe_title in recommended_recipes:
    print(recipe_title)
