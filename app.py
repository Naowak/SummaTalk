import requests
import markdown
from flask import Flask, render_template, redirect, url_for, request, jsonify
import concurrent.futures
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Prompts
PROMPT_INNER = """Resume le texte ci-dessus."""

PROMPT = """Tu es un expert pour faire des comptes rendues structuré et formel de vidéo.
Précédemment, tu as résumé plusieurs parties d'une même vidéo.
Je vais t'envoyer l'ensemble de ces-dits résumés que tu as déjà produit.
Tu dois maintenant me produire un seul et unique compte rendu faisant une synthèse de l'ensemble de ces résumés.
Le résultat final ne doit pas laisser paraitre qu'il s'agit d'une synthèse de plusieurs résumés. Mais bien d'un seul et unique document.
Tu dois trouver un titre au document, puis rédiger différentes parties et sous-parties pour structurer l'information.
Le résultat doit être écris en français, et sous le format markdown.
"""

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
    return chunks

def group_chunks_by(chunks, nb):
    return [chunks[i:i+nb] for i in range(0, len(chunks), nb)]

def make_completion_request(prompt):
    url = 'http://localhost:8000/v1/completions'
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "prompt": prompt,
        "max_tokens": 4096,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def make_several_completion_requests(prompts):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(make_completion_request, prompts)
    results = list(results)
    return results

def make_summary(filename):
    # Retrieve chunks
    chunks = group_chunks_by(get_file_chunk(filename, df), 8)
    prompts = []
    for chunk in chunks:
        prompt = ''.join(chunk) + "\n\n" + PROMPT_INNER + "\n\n"
        prompts.append(prompt)
    results = make_several_completion_requests(prompts)
    summaries = [result['choices'][0]['text'] for result in results]

    result = make_completion_request(PROMPT + "\"" + ''.join(summaries) + "\"\n\n")
    return result['choices'][0]['text'] 

# Load the App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/result')
def result():
    with open('./summary.txt', 'r') as f:
        summary = f.read()
    summary_html = markdown.markdown(summary)
    return render_template('result.html', summary=summary_html)


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
        # Make summary !
        print("summary begin")
        summary = make_summary(file.filename)
        with open('./summary.txt', 'w+') as f:
            f.write(summary)
        return jsonify({'summary': summary})
    return jsonify({'error': 'No file uploaded'}), 400




if __name__ == '__main__':
    app.run(debug=True)
