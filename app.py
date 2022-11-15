
from flask import Flask, flash, redirect, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from model import predict_organ

organs = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']

result = ""
showResult = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.testing = True
app.config.update(
    TESTING=True,
    SECRET_KEY='192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf'
)

@app.route('/')
def index():
    return render_template("app.html", organs=organs, result=result)

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

                organ = request.form.get('organ_select')
                predict_organ(organ)
                result = './static/result.png'


                return render_template("app.html", organ=organ, organs=organs, result=result, showResult=True)
    
    return render_template("app.html", showResult=False)