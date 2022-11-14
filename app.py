
from flask import Flask, flash, redirect, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from model import predict_organ

predict_organ('lung')

organs = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.testing = True
app.config.update(
    TESTING=True,
    SECRET_KEY='192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf'
)

@app.route('/')
def index():
    select = "Hola desde python"
    return render_template("app.html", title=select, organs=organs)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' in request.files:
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    select = request.form.get('organ_select')
    return render_template("app.html", title=select, organs=organs)