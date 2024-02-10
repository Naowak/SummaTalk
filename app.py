from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('flaubert/flaubert_base_uncased')

# Load the vectorized database
df = pd.read_csv('rag_db.csv')
df.embedding = df.embedding.apply(lambda x: eval(x))

def find_closest_chunk(query, df, model, n=3):
    query_embedding = model.encode([query])[0]
    similarity = df.embedding.apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
    ids = similarity.sort_values(ascending=False).head(n)
    return df.loc[ids.index].text.tolist()

# Load the App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


# Buttons Behaviour 
@app.route('/rag', methods=['POST'])
def rag():
    data = request.get_json()  # Get the JSON data sent with the request
    prompt = data.get('prompt')  # Retrieve the 'prompt' value from the JSON data
    chunks = find_closest_chunk(prompt, df, model)
    return 'Success'


if __name__ == '__main__':
    app.run(debug=True)
