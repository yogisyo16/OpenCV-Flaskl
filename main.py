from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
import cv2
import numpy as np
import io
import os
from datetime import datetime
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def adjust_brightness(image, value):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.add(image, value)
    else:  # Color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        if value > 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = -value
            v[v < lim] = 0
            v[v >= lim] -= -value
            
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, value):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.convertScaleAbs(image, alpha=value, beta=0)
    else:  # Color image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.convertScaleAbs(l, alpha=value, beta=0)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_blur(image, value):
    # Ensure value is odd
    value = max(1, value)
    if value % 2 == 0:
        value += 1
    return cv2.GaussianBlur(image, (value, value), 0)

def apply_grey(image, value):
    value = value
    if value == 'true':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return image

def apply_edge(image, value):
    value = value
    if value == 'true':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)
    else:
        return image

def apply_processing(image, brightness_value=0, contrast_value=1.0, blur_value=5, grayscale_value='false', edge_value='false'):
    if 'brightness':
        return adjust_brightness(image, brightness_value)
    elif 'contrast':
        return adjust_contrast(image, contrast_value)
    elif 'grayscale':
        return apply_grey(image, grayscale_value)
    elif 'blur':
        return apply_blur(image, blur_value)
    elif 'edge':
        return apply_edge(image, edge_value)
    else:
        return image

def image_to_base64(image):
    if len(image.shape) == 2:  # Grayscale image
        _, buffer = cv2.imencode('.jpg', image)
    else:  # Color image
        _, buffer = cv2.imencode('.jpg', image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Read image file
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            original_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
            brightness_value = int(request.form.get('brightness_value', 0))
            contrast_value = float(request.form.get('contrast_value', 1.0))
            blur_value = int(request.form.get('blur_value', 5))
            grayscale_value = request.form.get('grayscale_value', 'false')
            edge_value = request.form.get('edge_value', 'false')

            
            # Apply processing
            processed_image = apply_processing(
                original_image, 
                brightness_value,
                contrast_value,
                blur_value,
                grayscale_value,
                edge_value

            )
            
            # Convert images to base64 for display
            processed_image_b64 = image_to_base64(processed_image)
            original_image_b64 = image_to_base64(original_image)
            
            return render_template('index.html', 
                                processed_image=processed_image_b64,
                                original_image=original_image_b64,
                                brightness_value=brightness_value,
                                contrast_value=contrast_value,
                                blur_value=blur_value,
                                grayscale_value=grayscale_value,
                                edge_value=edge_value)
    
    return render_template('index.html')

@app.route('/preview', methods=['POST'])
def preview():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        brightness_value = int(request.form.get('brightness_value', 0))
        contrast_value = float(request.form.get('contrast_value', 1.0))
        blur_value = int(request.form.get('blur_value', 5))
        grayscale = request.form.get('grayscale') == 'true'
        edge = request.form.get('edge') == 'true'
        
        if brightness_value != 0:
            image = adjust_brightness(image, brightness_value)
        
        if contrast_value != 1.0:
            image = adjust_contrast(image, contrast_value)
        
        if blur_value > 1:
            image = apply_blur(image, blur_value)
        
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if edge:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.Canny(image, 100, 200)
        
        return jsonify({
            'original_image': image_to_base64(cv2.imdecode(data, cv2.IMREAD_COLOR)),
            'processed_image': image_to_base64(image)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['POST'])
def download():
    try:
        if 'file' not in request.files:
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(url_for('index'))
        
        if not allowed_file(file.filename):
            return redirect(url_for('index'))
        
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        brightness_value = int(request.form.get('brightness_value', 0))
        contrast_value = float(request.form.get('contrast_value', 1.0))
        blur_value = int(request.form.get('blur_value', 5))
        grayscale = request.form.get('grayscale') == 'true'
        edge = request.form.get('edge') == 'true'
        
        if brightness_value != 0:
            image = adjust_brightness(image, brightness_value)
        
        if contrast_value != 1.0:
            image = adjust_contrast(image, contrast_value)
        
        if blur_value > 1:
            image = apply_blur(image, blur_value)
        
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if edge:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.Canny(image, 100, 200)
        
        _, buffer = cv2.imencode('.jpg', image)
        return send_file(
            io.BytesIO(buffer),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'processed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        )
        
    except Exception as e:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)