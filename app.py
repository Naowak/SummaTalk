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

def get_file_chunk(filename, df):
    chunks = df[df.filename == filename.split('.')[0]].text.tolist()
    print("chunkkk", chunks)
    return ''.join(chunks)

# Load the App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/result/<filename>')
def result(filename):
    #chunks = get_file_chunk(filename, df)
    return render_template('result.html', filename=filename)


# Buttons Behaviour 
@app.route('/rag', methods=['POST'])
def rag():
    data = request.get_json()  # Get the JSON data sent with the request
    prompt = data.get('prompt')  # Retrieve the 'prompt' value from the JSON data
    chunks = find_closest_chunk(prompt, df, model)
    return 'Success'

# Load a file
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print('No selected file')
        return redirect(request.url)
    if file:
        file.filename
        return redirect(url_for('result', filename=file.filename))
    return redirect(url_for('upload'))



if __name__ == '__main__':
    app.run(debug=True)
