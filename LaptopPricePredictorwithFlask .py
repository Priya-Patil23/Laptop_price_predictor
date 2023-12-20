from flask import Flask, render_template , request
import pickle
import numpy as np

model = pickle.load(open('laptop.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' ,methods = ['POST'])
def predict():
    company = request.form['company']
    typename = request.form['typename']
    CPU = request.form['CPU']
    GPU = request.form['GPU']
    weight = request.form['Weight']
    ppi = request.form['ppi']
    ram = request.form['Ram']
    hdd = request.form['HDD']
    ssd = request.form['SSD']
    os = request.form['OS']
    param = np.array([[company,typename,CPU,ram,os,weight,ssd,hdd,GPU,ppi]])
    prediction = model.predict(param)
    result = f"The Prdicted price of your laptops is : {np.round(np.exp(prediction) , 2)} "

    return render_template('index.html' , result = result)

if __name__ == '__main__':
    app.run(debug=True)
