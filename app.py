from flask import Flask, request, jsonify
import os
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini API
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

# Reasoning prompt template
PROMPT_TEMPLATE = """
You are an expert in natural language processing for e-commerce product keyword extraction. Your task is to analyze the user query and extract keywords that represent specific products or product categories typically found in a retail catalog (e.g., clothing, electronics, cosmetics, home goods). Follow these steps:

1. Identify nouns or noun phrases in the query that correspond to tangible products or product categories.
2. Exclude non-product terms like adjectives, prepositions, or contextual modifiers (e.g., "for male," "best," "running").
3. Consider singular and plural forms of products (e.g., "shoe" and "shoes" should both map to "shoes").
4. For multi-word products (e.g., "hair shampoo," "noise-canceling headphones"), keep the full phrase if it represents a specific product.
5. Return the extracted keywords as a comma-separated list (e.g., "shoes, shampoo, sunglasses"). If no products are found, return an empty string.

Query: {query}

Output the extracted keywords as a comma-separated list.
"""

@app.route('/extract-keywords', methods=['POST'])
def extract_keywords():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query = data['query']
        if not isinstance(query, str) or not query.strip():
            return jsonify({"error": "Query must be a non-empty string"}), 400

        # Create the prompt
        prompt = PROMPT_TEMPLATE.format(query=query)

        # Initialize the Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Send the prompt to the Gemini API
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Convert the comma-separated string to a list
        if response_text:
            keywords = [keyword.strip() for keyword in response_text.split(",") if keyword.strip()]
        else:
            keywords = []

        return jsonify({"keywords": keywords}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
