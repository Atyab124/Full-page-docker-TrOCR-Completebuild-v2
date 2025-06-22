from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import ast
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import numpy as np
import subprocess
import threading
from math import ceil
from dotenv import load_dotenv
import os

port = 8080

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

#Global Executions
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
cuda=True
device = "cuda:0" 
model.to(device)

#Global Functions
def extract_list(input_string):
    start_index = input_string.find('[')
    end_index = input_string.rfind(']') + 1
    extracted_list_str = input_string[start_index:end_index]
    extracted_list = ast.literal_eval(extracted_list_str)
    return extracted_list

def trocr_processor(image_path, file_content, res):
    for i in range(len(file_content)):
        try:
            coordinates = file_content[i]
            image = crop_image_4_points(image_path, coordinates)
            image = Image.fromarray(image)
        except ValueError:
            continue
        image = image.convert('RGB')
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_length=50)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Append both generated text and coordinates
        res.append({"text": generated_text, "coordinates": coordinates})

def crop_image_4_points(image_path, points):
    image=cv2.imread(image_path)
    if len(points) != 4:
        raise ValueError("Four points are required.")
    points = np.array(points, dtype=np.float32)

    # Order the points: top-left, top-right, bottom-right, bottom-left
    sorted_points = np.array([points[np.argmin(points.sum(1))],
                            points[np.argmin(np.diff(points, axis=1))],
                            points[np.argmax(points.sum(1))],
                            points[np.argmax(np.diff(points, axis=1))]], dtype=np.float32)

    # Compute the width and height of the new image
    width = max(np.linalg.norm(sorted_points[1] - sorted_points[0]), np.linalg.norm(sorted_points[2] - sorted_points[3]))
    height = max(np.linalg.norm(sorted_points[3] - sorted_points[0]), np.linalg.norm(sorted_points[2] - sorted_points[1]))

    # Define the new image points
    new_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(sorted_points, new_points)

    # Perform the perspective transformation to obtain the cropped image
    cropped_image = cv2.warpPerspective(image, matrix, (int(width), int(height)))

    return cropped_image

def sort_boxes(arrayobj, lineheight=None, options=None):

    lineYCorrection = 7
    if lineheight:
        lineYCorrection = lineheight * 0.4

    sortedNodes = []

    availableNodes = arrayobj[:]  # make a copy of the input list
    while availableNodes:
        minY = float('inf')
        for node in availableNodes:
            try:
                y = node[3][1]
            except:
                continue
            if options and options == 'AVERAGE':
                y = (node[3][1] + node[0][1]) / 2
            minY = min(minY, y)

        topRow = []
        otherRows = []
        for node in availableNodes:
            a = node
            try:
                y = a[3][1]
            except:
                continue
            if options and options == 'AVERAGE':
                y = (a[3][1] + a[0][1]) / 2
            if y + lineYCorrection >= minY and y - lineYCorrection <= minY:
                topRow.append(node)
            else:
                otherRows.append(node)

        topRow.sort(key=lambda n: n[0])
        sortedNodes.extend(topRow)
        availableNodes = otherRows

    return sortedNodes

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/hand_OCR', methods=['POST'])
def ocr():

    load_dotenv()
    threads=int(os.environ["PARALLEL_THREADS"])
    global cuda

    if 'image' not in request.files:
        return 'error : No file part'

    file = request.files['image']

    if file.filename == '':
        return 'error : No selected file'
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

    #Paddle text detection model
    # command = (
    #     "python predict_det.py"
    #     f" --image_dir={image_path}"
    #     " --det_model_dir='en_PP-OCRv3_det_infer/'"
    #     f" --use_gpu={cuda}"
    #     " --det_limit_side_len=1920"
    #     " --det_db_thresh=0.5"
    #     " --det_db_box_thresh=0.4"
        
    command = (
        "python predict_det.py"
        f" --image_dir={image_path}"
        " --det_model_dir='en_PP-OCRv3_det_infer/'"
        f" --use_gpu=False"
        " --det_limit_side_len=1920"
        " --det_db_thresh=0.5"
        " --det_db_box_thresh=0.4"
    )

    #     command = (
    #     "paddleocr "
    #     f" --image_dir={image_path}"
    #     " --det_model_dir='en_PP-OCRv3_det_infer/'"
    #     f" --use_gpu={cuda}"
    #     " --det_limit_side_len=1920"
    #     " --det_db_thresh=0.5"
    #     " --det_db_box_thresh=0.4"
    #     " --use_angle_cls true"
    #     " --lang en"
    # )

    # Execute the command
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)

    # file_content processing
    filepath='inference_results/det_results.txt'

    with open(filepath, 'r') as file:
        file_content = file.read()

    file_content=extract_list(file_content)

    file_content = [[[int(cell) for cell in row] for row in table] for table in file_content]

    file_content=sort_boxes(file_content)



    length=len(file_content)

    step=ceil(len(file_content)/threads)
    steps=ceil(len(file_content)/step)
    c=[0]
    i=1
    fc={}
    res={}

    while i<steps:
        c.append(c[i-1]+step)
        fc[i]=file_content[c[i-1]:c[i]]
        res[i]=[]
        i=i+1

    fc[i]=file_content[c[i-1]:length]
    res[i]=[]

    t={}
    for i in range(len(res)):
        t[i+1]=threading.Thread(target=trocr_processor, args=(image_path, fc[i+1], res[i+1]))
        t[i+1].start()

    for i in range(len(t)):
        t[i+1].join()

        # Process file content and return text and coordinates
    result = []
    for i in range(len(res)):
        result.append(res[i + 1])

    os.remove(image_path)

    # Return both text and coordinates as JSON response
    return jsonify(result)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=port)
