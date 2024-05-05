from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the Keras model

model = load_model("models/modelwqp.h5")

# Load the StandardScaler object

with open('models\scaler_params.pkl','rb') as f:
    d=pickle.load(f)
sc = StandardScaler()
sc.mean_ = d['mean']
sc.scale_ = d['scale']

#Define Minimum and Maximum Values for Water Quality
min__values = np.array([[6.5,60,500,-4,3,0,-2,0,1]],dtype=float)
max__values = np.array([[8.5,120,1000,4,30,400,2,80,5]],dtype=float)

# Define a function to preprocess input data
def preprocess_input(data):
    # Convert input dictionary to array and reshape it
    lis = ["ph","hardness","solids","chloramines","sulfates","conductivity","organic_carbon","trihalomethanes","turbidity"]
    features_arr = np.array([[float(data[key]) for key in lis]])
    # Perform any preprocessing steps here, like scaling
    # Example: features_arr_scaled = scaler.transform(features_arr)
    return features_arr

# Define a function for mul_factor
def find_mul_factor(processed_data):
    min_value = processed_data>=min__values
    max_value = processed_data<=max__values
    global mul_factor
    if np.all(min_value) and np.all(max_value):
        mul_factor = 100
    else:
        mul_factor = 10

@app.route('/')
def index():
    return render_template('index.html', prediction="Enter Data to predict")


@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the request
    data = request.form.to_dict()
    # Preprocess input data
    processed_data = preprocess_input(data)
    # Find mul_factor
    find_mul_factor(processed_data)
    # Fit the Preprocessed data
    processed_data = sc.transform(processed_data)
    # Predict water quality
    prediction = f"%.2f%% Safe Drinking Water"%(model.predict(processed_data)[0][0] * mul_factor)
    # Render the result in result.html
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
