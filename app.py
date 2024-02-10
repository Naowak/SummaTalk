from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/pagetwo')
def pagetwo():
    return render_template('index2.html')