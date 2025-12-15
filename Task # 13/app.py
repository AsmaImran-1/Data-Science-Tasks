from fastapi import FastAPI
import pickle

app = FastAPI(title="E-Commerce Recommendation API")

# Load saved models and data
df = pickle.load(open("products.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
nn_model = pickle.load(open("nn_model.pkl", "rb"))

# Normalize descriptions for safe searching
df["Description_clean"] = df["Description"].str.strip().str.lower()

@app.get("/")
def home():
    return {"message": "E-Commerce Recommendation System API"}

@app.get("/recommend/{product}")
def recommend(product: str):
    try:
        # Clean user input
        product_clean = product.strip().lower()

        # Check if product exists
        if product_clean not in df["Description_clean"].values:
            return {"error": f"Product '{product}' not found"}

        # Get index of the product
        idx = df[df["Description_clean"] == product_clean].index[0]

        # Ensure combined_features column exists
        if "combined_features" not in df.columns:
            return {"error": "Data is missing 'combined_features' column. Run preprocessing again."}

        # Transform input product using vectorizer
        product_vector = vectorizer.transform([df.iloc[idx]["combined_features"]])

        # Get nearest neighbors
        distances, indices = nn_model.kneighbors(product_vector)

        # Collect unique recommendations, skip the input product itself
        recommended = []
        for i in indices[0]:
            desc = df.iloc[i]["Description"]
            if desc != df.iloc[idx]["Description"] and desc not in recommended:
                recommended.append(desc)

        return {
            "selected_product": df.iloc[idx]["Description"],
            "recommendations": recommended[:5]  # top 5 recommendations
        }

    except Exception as e:
        # Return actual error for debugging
        return {"error": str(e)}
