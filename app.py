from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('D:\School\ktr-27-11/model.pkl', 'rb'))
@app.route('/')
def hello():
    return render_template('index.html')

# prediction function   
@app.route('/predict', methods = ['POST']) 
def predict(): 
    A = [float(x) for x in request.form.values()]
    model_probability = model.predict_proba([A])
    print(A)
    print(model_probability)
    prediction = "Xác suất đậu kì thi là: %0.0f"%model_probability[0][1]
    return render_template('index.html', result = prediction)
if __name__ == "__main__":
    app.run(debug=True)
    
    