# from flask import Flask, request, render_template, redirect, url_for
# import pickle
# import cv2
# import numpy as np
# from PIL import Image

# app = Flask(__name__)

# # Load the trained model

# with open('model.pkl','rb') as f:
#     model = pickle.load(f)


# def preprocess_image(image):
#     img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
#     img = cv2.resize(img, (64, 64))
#     img = img.flatten().reshape(1, -1)
#     return img

# @app.route('/',method=['GET','POST'])
# def index():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             image = Image.open(file)
#             img = preprocess_image(image)
#             prediction = model.predict(img)
#             labels = ['BENGIN CANCER','Normal','MALIGNANT CANCER']
#             result = labels[prediction[0]]

#             return render_template('result.html',result=result)
#             return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template, redirect, url_for
import pickle
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
with open(r'model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_image(image):
    # Ensure the image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (64, 64))  # Resize the image to (64, 64)
    img = img.flatten().reshape(1, -1)  # Flatten and reshape for model input
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                image = Image.open(file)
                img = preprocess_image(image)
                prediction = model.predict(img)
                
                labels = ['Bengin cancer', 'Malignant cases', 'Normal']
                result = labels[prediction[0]]
                
                return render_template('result.html', result=result)
            except Exception as e:
                return f"Error processing the image: {str(e)}"
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
