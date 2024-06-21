import joblib
import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Load the saved model components
U = joblib.load('U_matrix.pkl')
sigma = joblib.load('sigma_matrix.pkl')
Vt = joblib.load('Vt_matrix.pkl')
product_id_to_name = joblib.load('product_id_to_name.pkl')
final_ratings_matrix = joblib.load('final_ratings_matrix.pkl')
preds_df = joblib.load('preds_df.pkl')

def recommend_items(userID):
    user_idx = userID - 1
    sorted_user_ratings = final_ratings_matrix.iloc[user_idx].sort_values(ascending=False)
    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    temp['product_id'] = temp.index
    temp['product_name'] = temp['product_id'].map(product_id_to_name)
    temp = temp[temp.iloc[:, 0] == 0]
    temp = temp.sort_values(temp.columns[1], ascending=False)
    recommendations = temp[['product_id', 'product_name']]
    return recommendations.head(10)

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
    app.run(host="0.0.0.0",port=5000)  # Run Flask in debug mode for development