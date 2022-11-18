
from flask import Flask, flash, redirect, render_template, request, url_for
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import pandas as pd
import os
from werkzeug.utils import secure_filename
from model import predict_organ

organs = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']
models = ['UNET', 'FPN', 'Both']

fpn_times = []
unet_times = []

result = ""
showResult = False

server = Flask(__name__)
server.config['UPLOAD_FOLDER'] = './uploads'
server.testing = True
server.config.update(
    TESTING=True,
    SECRET_KEY='192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf'
)

app = Dash(__name__, server=server, url_base_pathname='/dash/')
app.config['suppress_callback_exceptions'] = True

@app.server.route('/')
def index():
    return render_template("app.html", organs=organs, result=result, models=models)

@app.server.route('/', methods=['GET', 'POST'])
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
                file_path = os.path.join(app.server.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                organ = request.form.get('organ_select')
                model = request.form.get('model_select')

                if (model != 'Both'):
                    metrics = fpn_times if model == 'FPN' else unet_times

                    model_time = predict_organ(organ, model, file_path)
                    result = f'./static/predicts/{model}_{organ}.png'
                    metrics.append(model_time)

                    return render_template(
                        "app.html",
                        organ=organ,
                        organs=organs,
                        result=result,
                        showResult=True,
                        both=False,
                        models=models,
                        model=model,
                    )
                
                model_1 = 'UNET'
                model_2 = 'FPN'
                unet_time = predict_organ(organ, model_1, file_path)
                unet_times.append(unet_time)
                fpn_time = predict_organ(organ, model_2, file_path)
                fpn_times.append(fpn_time)
                result = f'./static/predicts/{model_1}_{organ}.png'
                result_2 = f'./static/predicts/{model_2}_{organ}.png'

                return render_template(
                    "app.html",
                    organ=organ,
                    organs=organs,
                    result=result,
                    showResult=True,
                    model=model_1,
                    models=models,
                    both=True,
                    model_m2=model_2,
                    result_m2=result_2,
                )
    
    return render_template("app.html", showResult=False)

app.layout = html.Div(
    className= 'dash-container',
    children=[
        html.Div(
            className='full-container',
            children=[
                html.Button('Reload', className='stats-basic', id='reload-button'),
                dcc.Graph(
                    id='execution-time',
                )
            ]
        ),
        dcc.Link(
            className='stats-container',
            children=['Return'],
            href="/",
            refresh=True,
        ),
    ]
)

@app.callback(
    Output('execution-time', 'figure'),
    Input('reload-button', 'value'))
def update_graph(_):
    cp_fpn = fpn_times[:]
    cp_unet = unet_times[:]

    if (len(cp_fpn) == 0 and len(cp_unet) == 0):
        cp_fpn.append(0)
        cp_unet.append(0)

    if (len(cp_fpn) > len(cp_unet)):
        for _ in range(len(cp_fpn)-len(cp_unet)):
            cp_unet.append(0)
    elif (len(cp_fpn) < len(cp_unet)):
        for _ in range(len(cp_unet)-len(cp_fpn)):
            cp_fpn.append(0)

    xs = len(cp_fpn)
    df = pd.DataFrame({
        "fpn": cp_fpn,
        "unet": cp_unet,
        "x": [x for x in range(1, xs+1)]
    })

    fig = px.line(
        df,
        x=df['x'],
        y=['fpn', 'unet'],
        color_discrete_sequence=['#80352D', '#FF6A59'],
    )
    fig.update_layout(xaxis_title="Number of Image", yaxis_title="Time")

    return fig


@app.server.route('/dash/', methods=['GET'])
def dashboard():
    return app.index()

@app.server.route('/dash/', methods=['POST'])
def returnScreen():
    parsed_path = request.url.split('/')[0]
    print(parsed_path)
    return redirect(url_for('/'))

if __name__ == "__main__":
    app.run(debug=True)