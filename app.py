import requests
from flask import Flask, render_template, redirect, url_for, request
import concurrent.futures
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Prompt
PROMPT = """Je vais t'envoyer un texte commencant et finissant par des guillemets.
- Tu es un secretaire.  
- On veut un compte rendu en markdown de ce texte tire de l'enregistrement d'une video. 

Instruction importantes:
- Ecrit entierement ta reponse en markdown.
- Tu dois ecrire en markdown.

Tes instructions :
- Structure l'information.
- Donne un titre au texte.
- Decoupe le texte en 2 ou 3 grandes parties en les numerotant et titre les.
- Decoupe les parties en sous partie en les numerorant.
- Fait un point pour chaque information.
- Fait des phrases avec un sujet, un verbe et un complement et un determinant pour les noms pour chaque point.
- Ecrit en francais.
- Si tu veux ecrire le texte en anglais, ecrit en francais.
- On veut que l'information soit structure.
- Ecrit entierement ta reponse en markdown, c'est important.
- Tu dois ecrire en markdown. 
- Ne repete pas les memes informations.
- Ne fait pas de conclusion.  
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
        prompt = PROMPT + "\"" + ''.join(chunk) + "\"\n\n"
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

@app.route('/summary/<filename>')
def summary(filename):
    return make_summary(filename)



if __name__ == '__main__':
    app.run(debug=True)
