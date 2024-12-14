from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import platform
import pathlib
import cv2

app = Flask(__name__)

plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Load Models
classification_model = torch.load('models/EfficientNet_Without_Dropout.pth',map_location=torch.device('cpu'))
classification_model.eval()

# Path to YOLOv5 repo and trained model
path_trained_model = Path(r'models/YoloV5_best.pt')
yolo_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_trained_model,force_reload=True)

yolo_segmentation_model = YOLO("models/YoloV8_best.pt") 

# Define classification-specific transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

flood_mapping = {1: "Flood", 0:"No Flood"}

# Define the BGR color mapping for 13 classes
color_mapping = {
    0: ('red', (0, 0, 255)),   # Red in BGR
    1: ('blue', (255, 0, 0)),  # Blue in BGR
    2: ('green', (0, 255, 0)), # Green in BGR
    3: ('yellow', (0, 255, 255)), # Yellow in BGR
    4: ('purple', (128, 0, 128)), # Purple in BGR
    5: ('cyan', (255, 255, 0)), # Cyan in BGR
    6: ('magenta', (255, 0, 255)), # Magenta in BGR
    7: ('orange', (0, 165, 255)), # Orange in BGR
    8: ('pink', (203, 192, 255)), # Pink in BGR
    9: ('brown', (42, 42, 165)), # Brown in BGR
    10: ('lime', (0, 255, 0)),  # Lime Green in BGR
    11: ('teal', (128, 128, 0)), # Teal in BGR
    12: ('navy', (128, 0, 0)),  # Navy Blue in BGR
}
# Global variable for in-memory image byte array
img_byte_arr = None

# Home function to render the index.html
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global img_byte_arr 

    # Ensure file exists in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    model_type = request.form.get('model')  # Get model type

    # Classification Model
    if model_type == 'classification':
        img = Image.open(file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = classification_model(img_tensor)
            probability_class_1 = torch.sigmoid(outputs).item()   # Convert logit to probability
            probability_class_0 = 1 - probability_class_1  # Probability of class 0

            if probability_class_1 > 0.5:
                 label = 1
                 confidence = probability_class_1
            else:
                 label = 0
                 confidence = probability_class_0

        return jsonify({
            'type': 'classification',
            'result': {'label': flood_mapping[label], 'confidence': confidence*100}
        })

    # Object Detection
    elif model_type == 'detection':
        img = Image.open(file).convert('RGB')
        results = yolo_detection_model(img)
        detection_data = results.xyxy[0].cpu().numpy()

        # Render the image in-memory
        results.render()  # Annotate boxes
        results_img = Image.fromarray(results.ims[0])

        # Convert to bytes for serving
        img_byte_arr = BytesIO()
        results_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Prepare JSON response
        detection_results = [
            {
                "x": int(box[0]), "y": int(box[1]),
                "width": int(box[2] - box[0]), "height": int(box[3] - box[1]),
                "confidence": float(confidence),
                "label": yolo_detection_model.names[int(class_id)]
            }
            for *box, confidence, class_id in detection_data
        ]

        return jsonify({
            "type": "detection",
            "results": detection_results,
            "image_url": request.host_url + 'rendered-image'
        })

    # Segmentation Model
    elif model_type == 'segmentation':
        img = Image.open(file).convert('RGB')
        results = yolo_segmentation_model(img)
        print(results)

        orig_img = np.array(img)  # Original image as a NumPy array
        masks = results[0].masks.data.cpu().numpy() if results[0].masks else None
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes else None
        names = results[0].names  # Class names 

        if masks is not None:
           overlay_img = orig_img.copy()  # Create a copy for overlaying
           predicted_classes = []  # Store predicted class names

           for i, mask in enumerate(masks):
              # Resize mask to match the original image size
               mask = cv2.resize(mask, (overlay_img.shape[1], overlay_img.shape[0]))

              # Convert mask to binary (0 or 255)
               mask = (mask * 255).astype('uint8') 

              # Get color and class name
               class_id = cls_ids[i]
               color_name, color = color_mapping.get(class_id, ('unknown', (0, 0, 0)))
            
              # Create a 3-channel mask from the single-channel mask
               colored_mask = np.zeros_like(orig_img, dtype=np.uint8)  # Initialize a black mask
               colored_mask[mask == 255] = color  # Apply the color to the mask where mask is 255

             # Blend the mask with the original image
               alpha = 0.4  # Adjust transparency
               overlay_img = cv2.addWeighted(overlay_img, 1, colored_mask, alpha, 0)

               # Append the predicted class and color name
               predicted_classes.append(f"{color_name}: {names[class_id]}")

            # Convert overlay image to PIL format
           blended_img = Image.fromarray(cv2.cvtColor(overlay_img,cv2.COLOR_BGR2RGB))

           img_byte_arr = BytesIO()
           blended_img.save(img_byte_arr, format='PNG')
           img_byte_arr.seek(0)

           return jsonify({
            "type": "segmentation",
            "segmentation_url": request.host_url + 'rendered-image',
            "classes": predicted_classes})
        else:
          return jsonify({"type": "segmentation","error": "No masks found in segmentation result."})


@app.route('/rendered-image', methods=['GET'])
def rendered_image():
    # Return the in-memory rendered image
    return send_file(img_byte_arr, mimetype='image/png')



if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)



