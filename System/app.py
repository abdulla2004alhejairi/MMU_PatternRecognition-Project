from flask import Flask, request, render_template, redirect, url_for, flash
import os
import cv2
from werkzeug.utils import secure_filename
import joblib
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont, ImageOps
import traceback

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER_PARTS'] = 'static/uploads/part_recognition'
app.config['UPLOAD_FOLDER_COUNTING'] = 'static/uploads/counting'
app.config['PREDICTED_FOLDER_COUNTING'] = 'static/uploads/predicted_counting'

# Ensure the upload folders exist
for folder in [app.config['UPLOAD_FOLDER_PARTS'], app.config['UPLOAD_FOLDER_COUNTING'], app.config['PREDICTED_FOLDER_COUNTING']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load the trained model and LDA objects
try:
    model = joblib.load('best_knn_model.pkl')
    lda = joblib.load('lda_model.pkl')
except Exception as e:
    print(f"Error loading model or LDA object: {e}")
    exit(1)

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Store uploaded images and predictions
uploads = []


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Preprocess the image for model prediction.
    - Convert to grayscale
    - Apply thresholding
    - Resize to 256x256
    - Normalize
    - Transform using LDA
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not loaded correctly")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        __, segmented_image = cv2.threshold(
            gray_image, 127, 255, cv2.THRESH_BINARY)
        resized_image = cv2.resize(segmented_image, (256, 256))
        normalized_image = resized_image / 255.0  # Normalize to [0, 1]
        flattened_image = normalized_image.flatten().reshape(1, -1)
        lda_features = lda.transform(flattened_image)
        return lda_features
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


@app.route('/')
def index():
    """
    Render the main page with the upload form and the results table.
    """
    return render_template('index.html', uploads=uploads)


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and prediction.
    - Save the file to the upload folder
    - Preprocess the image
    - Predict the part using the loaded model
    - Store the result and display on the main page
    """
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(
                app.config['UPLOAD_FOLDER_PARTS'], filename)
            file.save(filepath)

            # Preprocess the image and predict
            lda_features = preprocess_image(filepath)
            if lda_features is None:
                flash('Error processing image')
                return redirect(url_for('index'))

            prediction = model.predict(lda_features)

            # Get the label (folder name) based on the prediction
            label = prediction[0]

            # Store the uploaded file path and prediction in the uploads list
            uploads.append((filename, label))

            flash(f'Part recognized as: {label}')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'An error occurred: {e}')
            return redirect(url_for('index'))
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(url_for('index'))


@app.route('/counting', methods=['GET'])
def counting():
    """
    Render the counting algorithm page.
    """
    return render_template('counting.html', counting_results=None)


@app.route('/upload_counting_image', methods=['POST'])
def upload_counting_image():
    """
    Handle file upload and counting algorithm prediction.
    - Save the file to the upload folder
    - Use Roboflow API to predict
    - Display the result on the counting page
    """
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('counting'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('counting'))

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER_COUNTING'], filename)

            # Open the image and remove orientation metadata
            image = Image.open(file)
            image = ImageOps.exif_transpose(image)

            # Save the correctly oriented image
            image.save(filepath)

            # Roboflow integration
            rf = Roboflow(api_key="JSY4rsJB5chn6jWuo0OA")
            project = rf.workspace().project("tpr-counting-algo")
            model = project.version(5).model

            # Infer on the uploaded image
            result = model.predict(filepath, confidence=50, overlap=50).json()

            if 'predictions' not in result:
                flash('No predictions found in the result')
                return redirect(url_for('counting'))

            object_count = len(result['predictions'])
            object_classes = [pred['class'] for pred in result['predictions']]

            colors = {
                'Black': 'white',
                'Gray': 'cyan',
                'Red': 'red',
                'Yellow': 'yellow'
            }

            draw = ImageDraw.Draw(image)

            # Dynamically adjust font size and border width
            image_width, image_height = image.size
            font_size = max(20, image_width // 30)
            border_width = max(5, image_width // 150)

            try:
                font = ImageFont.truetype("arial.ttf", size=font_size)
            except IOError:
                font = ImageFont.load_default()

            for prediction in result['predictions']:
                x = prediction['x']
                y = prediction['y']
                width = prediction['width']
                height = prediction['height']
                class_name = prediction['class']
                confidence_rate = prediction['confidence']

                x1 = x - (width / 2)
                y1 = y - (height / 2)
                x2 = x + (width / 2)
                y2 = y + (height / 2)

                draw.rectangle([x1, y1, x2, y2],
                               outline=colors.get(class_name, 'white'), width=border_width)

                text_bbox = font.getbbox(f"{class_name} {confidence_rate:.2f}")
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_background = [(x1, y1 - text_height),
                                   (x1 + text_width, y1)]
                draw.rectangle(text_background, fill=colors.get(class_name, 'white'))

                text_color = 'white' if colors.get(class_name) in ['red'] else 'black'
                draw.text((x1, y1 - text_height),
                          f"{class_name} {confidence_rate:.0%}", fill=text_color, font=font)

            prediction_image_path = os.path.join(app.config['PREDICTED_FOLDER_COUNTING'], f"pred_{filename}")
            image.save(prediction_image_path)

            counting_results = {
                'filename': filename,
                'object_count': object_count,
                'object_classes': object_classes,
                'prediction_image_path': prediction_image_path,
            }

            return render_template('counting.html', counting_results=counting_results)
        except Exception as e:
            flash(f'An error occurred: {e}')
            traceback.print_exc()
            return redirect(url_for('counting'))
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(url_for('counting'))

if __name__ == "__main__":
    app.run(debug=True)
