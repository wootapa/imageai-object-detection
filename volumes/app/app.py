from imageai.Detection import ObjectDetection
from flask import Flask, jsonify, request
from PIL import Image
from PIL import Image
import base64
import os
import numpy as np
import cv2
import math

LABELS_PATH = '/opt/model/labels.txt'
MODEL_PATH = '/opt/model/model.h5'
ISYOLO = os.environ.get('ISYOLO', 0) == '1'
ISRETINANET = os.environ.get('ISRETINANET', 0) == '1'
SCORE_THRESHOLD = 50

# Load imageai
detector = ObjectDetection()
if ISRETINANET:
    detector.setModelTypeAsRetinaNet()
if ISYOLO:
    detector.setModelTypeAsYOLOv3()
detector.setModelPath(MODEL_PATH)
detector.loadModel(detection_speed='normal') #"normal"(default), "fast", "faster" , "fastest" and "flash"

app = Flask(__name__, static_folder=os.path.dirname(os.path.realpath(__file__)))

# Response models
class Result:
    def __init__(self, category, score):
        self.category = category
        self.score = score

    def toJson(self):
        return {
            'category': self.category,
            'score': self.score
        }

class ResultSummary:
    def __init__(self, results, img_arr):
        # Downscale a bit
        img_arr = resize_np_image(img_arr)
        # Ensure RGB color
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        # Encode as JPG
        retval, img_arr = cv2.imencode('.jpg', img_arr)

        self.results = results
        self.image = ('data:image/jpeg;base64,' + base64.b64encode(img_arr).decode('ascii'))

    def toJson(self):
        return {
            'results': [result.toJson() for result in self.results],
            'image': self.image
        }

def resize_np_image(np_arr, new_width=640):
    height, width, channel = np_arr.shape
    ratio = float(new_width) / float(width)
    new_height = int(height * ratio)
    return cv2.resize(np_arr, (new_width, new_height))

def get_labels():
    labels_map = {}
    with open(LABELS_PATH, 'r') as f:
        for line in f:
            (key, val) = line.split(':')
            labels_map[int(key)] = val.rstrip().lower()
    return labels_map

def infer(img_arr_in):
    my_objects = detector.CustomObjects(person=True, truck=True, car=True, motorcycle=True, bird=True, cat=True, dog=True)
    img_arr, detections = detector.detectCustomObjectsFromImage(
        custom_objects=my_objects,
        input_image=img_arr_in,
        input_type='array',
        output_type='array',
        minimum_percentage_probability=SCORE_THRESHOLD,
        display_percentage_probability=False,
        display_object_name=False
    )

    result_list = []
    for d in detections:
        category = d['name']
        score = math.ceil(d['percentage_probability'])
        result = Result(category, score)
        result_list.append(result)
    result_list.sort(key=lambda x: x.score, reverse=True)
    return ResultSummary(result_list, img_arr)

@app.route('/')
def index():
   return app.send_static_file('index.html')

@app.route('/categories', methods=['GET'])
def categories():
    categories = []
    labels_map = get_labels()
    for key in labels_map:
        categories.append(labels_map[key])
    categories.sort()
    return jsonify(categories)

@app.route('/classify', methods=['POST'])
def classify():
    img = Image.open(request.files['file'])


    # Maybe convert png->jpeg
    if not img.mode == 'RGB':
        img = img.convert('RGB')

    # Infer...
    summary = infer(np.array(img))
    return jsonify({'summary':summary.toJson()})

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(port=5000, host=('0.0.0.0'), threaded=False)