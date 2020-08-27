import os
import urllib.request
from app import app
from flask import Flask, request, redirect, jsonify, render_template
from werkzeug.utils import secure_filename
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
import shutil

import redis
import json
from datetime import datetime
import random

HOST_NAME="0.0.0.0"
PORT="6378"
PASSWORD=''

r = redis.Redis(
    host=HOST_NAME,
    port=PORT,
    password=PASSWORD,
)
NAME_QUEUE='queue'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# response = jsonify({'msg': 'no file part in the request'})
@app.route('/', methods=['GET'])
def index():
    return 'server is running'
    # res = response

@app.route('/detect_mask', methods=['GET'])
def detect_mask():
    return render_template('index.html')

@app.route('/label_classification', methods=['POST'])
def label_classification():
    data = request.get_json(force=True)
    print (data["with_mask"])
    print (data["without_mask"])
    print (data["detect_wrong"])
    # with_mask
    for _element_with_mask in data["with_mask"]:
        shutil.move(os.path.join(app.config['PATH_CROP_IMAGE_DETECT'], _element_with_mask.split('/')[4]),
        os.path.join(app.config['PATH_DETECT_WITH_MASK'], _element_with_mask.split('/')[4]))
    # without_mask
    for _element_without_mask in data["without_mask"]:
        shutil.move(os.path.join(app.config['PATH_CROP_IMAGE_DETECT'], _element_without_mask.split('/')[4]),
        os.path.join(app.config['PATH_DETECT_WITHOUT_MASK'], _element_without_mask.split('/')[4]))
    # detect_wrong
    for _element_detect_wrong in data["detect_wrong"]:
        shutil.move(os.path.join(app.config['PATH_CROP_IMAGE_DETECT'], _element_detect_wrong.split('/')[4]),
        os.path.join(app.config['PATH_DETECT_WRONG'], _element_detect_wrong.split('/')[4]))
    return {
        "status": "True",
        "msg": "successful"
    }

@app.route('/label_page', methods=['GET'])
def label_page():
    return render_template('label_page.html')

@app.route('/label_data', methods=['GET'])
def label_image():
    listOfFiles = os.listdir('./public/crop_detect')
    len_listOfFiles = len(listOfFiles)
    if len_listOfFiles == 0:
        return {
            'status': False,
            'msg': 'List image label is empty wait to update new data',
        }
    else:
        _arrImage = []
        _len_for = 9
        if len_listOfFiles < 9:
            _len_for = len_listOfFiles
        for _count in range(0,_len_for):
            _arrImage.append(listOfFiles[_count])

        return {
            'status': True,
            'msg': 'Get image is successful',
            'result': _arrImage,
        }

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        res = jsonify({'msg': 'no file part in the request'})
        res.status_code = 400
        return res
    file = request.files['file']
    if file.name == '':
        res = jsonify({'msg': 'No file selected for updating'})
        res.status_code = 400
        return res
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print (len(filename.split('.')))
        _index_tail = len(filename.split('.')) - 1
        print (filename.split('.')[_index_tail])
        print (str(random.randint(1000000,9999999)))
        print (str(random.randint(1000000,9999999)) + '.' + filename.split('.')[_index_tail])
        filename = str(random.randint(1000000,9999999)) + '.' + filename.split('.')[_index_tail]
        # filename = random.randint(1000000,9999999)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print (os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # genera id of task
        # change image name
        # craete template upload file
        ID_TASK=random.randint(1000000,9999999)
        _data = {
            'idTask': ID_TASK,
            'filename': filename,
            'input': os.path.join(app.config['UPLOAD_FOLDER'], filename),
            'outputResize': os.path.join(app.config['OUTPUT_RESIZE'], filename),
            'output': os.path.join(app.config['OUPUT_DETECT'], filename),
            'status': 'wait',
            'msg': 'File successfully uploaded',
            'lenResult': 1,
            'pathCropPiture': app.config['PATH_CROP_IMAGE_DETECT'],
        }
        r.rpush(NAME_QUEUE, json.dumps(_data))
        # wait to response
        while True:
            if r.get(ID_TASK):
                print ('waiting...')
                data_response = r.get(ID_TASK)
                data_response = json.loads(data_response)
                if data_response['status'] == 'done':
                    print (datetime.now())
                    print ('done')
                    print (data_response)
                    # clear key
                    r.delete(ID_TASK)
                    break
                else:
                    print ('not exists')
        res = jsonify(data_response)
        res.status_code = 201
        return res
    else:
        res = jsonify({'msg': 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        res.status_code = 400
        return res

if __name__ == "__main__":
    app.run(debug=True)