import os
# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template_string
import keras
from keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the pre-trained Keras model
MODEL_PATH = 'model.h5'
print(f"Loading Keras model from {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Categories mapped in training order
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Custom recycling tips corresponding to each class
RECYCLING_TIPS = {
    'cardboard': {
        'title': 'Cardboard Recycling Guide',
        'color': '#ffb03a',
        'tips': [
            'Flatten all boxes to save space in the recycling bin.',
            'Remove any plastic wraps, tape, bubble wrap, or packing peanuts.',
            'Ensure the cardboard is dry and free of greasy residues (like pizza boxes).'
        ]
    },
    'glass': {
        'title': 'Glass Recycling Guide',
        'color': '#42d885',
        'tips': [
            'Rinse jars and bottles with water to remove food residues.',
            'Metal lids can be recycled separately, but plastic caps should be thrown in the trash.',
            'Avoid breaking the glass before disposing, as small shards are harder to sort.'
        ]
    },
    'metal': {
        'title': 'Metal & Can Recycling Guide',
        'color': '#4ea5ff',
        'tips': [
            'Wash out food residue from aluminum and tin cans.',
            'You do not need to peel off paper labels; they are burned off during processing.',
            'Empty aerosol cans are recyclable in many areas; check local guidelines.'
        ]
    },
    'paper': {
        'title': 'Paper Recycling Guide',
        'color': '#a78bfa',
        'tips': [
            'Recycle clean newspapers, envelopes, junk mail, and printer paper.',
            'Do not recycle paper contaminated with grease, paint, or chemical spills.',
            'Shredded paper is often not accepted in standard curbside bins; bag it if required.'
        ]
    },
    'plastic': {
        'title': 'Plastic Recycling Guide',
        'color': '#f43f5e',
        'tips': [
            'Check the resin code number (1 to 7) on the bottom of the container.',
            'Rinse and compress plastic bottles to optimize space.',
            'Leave caps on if screwed tight, or throw them in the trash if loose.'
        ]
    },
    'trash': {
        'title': 'General Waste Guide',
        'color': '#9ca3af',
        'tips': [
            'This item belongs in general waste and cannot be recycled locally.',
            'Consider if the item can be repurposed, repaired, or composted.',
            'Dispose of it securely to avoid littering and environmental contamination.'
        ]
    }
}

def preprocess_image(image_path_or_file):
    """
    Applies the identical preprocessing pipeline used during training:
    1. Loads the image and converts it to RGB.
    2. Resizes the image to 224x224.
    3. Scales pixel values between [-1, 1] using MobileNetV2 preprocess_input.
    4. Adds batch dimension.
    """
    img = Image.open(image_path_or_file).convert('RGB')
    img = img.resize((224, 224))
    
    img_arr = np.array(img, dtype=np.float32)
    img_arr = preprocess_input(img_arr)
    img_arr = np.expand_dims(img_arr, axis=0) # Shape: (1, 224, 224, 3)
    return img_arr

@app.route('/')
def index():
    # We serve the frontend using render_template
    return send_from_directory('templates', 'index.html')

@app.route('/samples/<category>/<filename>')
def serve_sample(category, filename):
    """Serves the pre-configured sample images from the local dataset directory."""
    directory = os.path.join('Garbage', 'original_images', category)
    return send_from_directory(directory, filename)

@app.route('/classify', methods=['POST'])
def classify():
    """Endpoint to run classification on uploaded images or selected sample images."""
    try:
        # Check if this is an uploaded file or a sample image reference
        if 'image' in request.files:
            file = request.files['image']
            img_arr = preprocess_image(file)
        elif 'sample_path' in request.json:
            sample_path = request.json['sample_path']
            # Security check to ensure the path stays within Garbage folder
            normalized_path = os.path.normpath(sample_path)
            if not normalized_path.startswith('Garbage'):
                return jsonify({'error': 'Invalid sample path access'}), 400
            img_arr = preprocess_image(normalized_path)
        else:
            return jsonify({'error': 'No image or sample path provided'}), 400

        # Run model inference
        predictions = model.predict(img_arr)[0]
        
        # Format prediction results
        results = []
        for i, prob in enumerate(predictions):
            results.append({
                'class': CLASS_NAMES[i],
                'probability': float(prob)
            })
        
        # Sort by highest probability
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        top_prediction = results[0]
        
        # Append recycling info for the top prediction
        category_info = RECYCLING_TIPS.get(top_prediction['class'], {})
        
        return jsonify({
            'success': True,
            'prediction': top_prediction['class'],
            'probability': top_prediction['probability'],
            'all_predictions': results,
            'recycling_info': category_info
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Garbage Classification Web Server...")
    app.run(host='127.0.0.1', port=5000, debug=True)
