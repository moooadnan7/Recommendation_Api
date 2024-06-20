from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np
import pandas as pd
import math
import json
import time
from sklearn.model_selection import train_test_split
import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings; warnings.simplefilter('ignore')
data = pd.read_csv('fashion_products_transformed.csv')
final_ratings_matrix = data.pivot(index = 'user-id', columns ='product-id', values = 'ratings').fillna(0)
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
sparse_ratings_matrix = csc_matrix(final_ratings_matrix)
U, sigma, Vt = svds(sparse_ratings_matrix, k=10)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
# Convert predicted ratings to dataframe
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = final_ratings_matrix.columns)
product_id_to_name = data.set_index('product-id')['product-name'].to_dict()
def recommend_items(userID):
    # index starts at 0
    user_idx = userID - 1
    # Get and sort the user's ratings
    sorted_user_ratings = final_ratings_matrix.iloc[user_idx].sort_values(ascending=False)
    # Get and sort the user's predictions
    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
    # Combine user ratings and predictions
    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    # Create a DataFrame with product IDs and names
    temp['product_id'] = temp.index
    temp['product_name'] = temp['product_id'].map(product_id_to_name)
    # Filter out already rated items
    temp = temp[temp.iloc[:, 0] == 0]
    # Sort by predicted ratings
    temp = temp.sort_values(temp.columns[1], ascending=False)
    # Select only product ID and product name
    recommendations = temp[['product_id', 'product_name']]
    return'\nBelow are the recommended items for user(user_id = {}):\n'.format(userID)
    return recommendations.head(10).to_string(index=False)
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
@app.route('/')
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        return f"Error rendering template: {e}", 500

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400

    try:
        recommendations = recommend_items(user_id)
        recommendations_list = recommendations.to_dict(orient='records')
        return jsonify(recommendations_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
