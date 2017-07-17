from flask import send_from_directory
import os
import re
import uuid
import scipy.misc
import base64
import json
from io import BytesIO
from PIL import Image
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from runner import build, execute

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'something unique and secret'
pix = build()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/tumor/', methods=['POST'])
def upload_tumor():
    if not request.data:
        return json.dumps({'result': 'failure'}), 200, {'ContentType': 'application/json'}
    data = json.loads(request.data.decode('UTF-8'))
    # return json.dumps({'result': 'success', 'url': data['url']}), 200, {'ContentType': 'application/json'}
    image_data = re.sub('^data:image/.+;base64,', '', data['url'])
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    filename = uuid.uuid4().hex + '.png'
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    scipy.misc.imsave(path, image)
    paths = execute(pix, path, with_gt=False)
    images = []
    for path in paths:
        with open(path, "rb") as f:
            images.append({
                'filename': os.path.split(path)[1],
                'url': base64.b64encode(f.read()).decode('UTF-8'),
            })
    return json.dumps({'result': 'success', 'results': images}), 200, {'ContentType': 'application/json'}



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            execute(pix, path, with_gt=False)
            return redirect(url_for('uploaded_file', filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/')
def hello_world():
    return 'Hello, World!'
