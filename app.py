from flask import Flask,render_template,request
import joblib as jb
import numpy as np
app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    sex =(request.form['sex'])
    on_thyroxine =(request.form['on_thyroxine'])
    pregnant =(request.form['pregnant'])
    query_hypothyroid =(request.form['query_hypothyroid'])
    goitre =(request.form['goitre'])
    psych =(request.form['psych'])
    tsh_measured = (request.form['tsh_measured'])
    tsh = (request.form['tsh'])
    tt4 = (request.form['tt4'])
    fti = (request.form['fti'])
    
    arr=np.array([[sex, on_thyroxine, pregnant,
                   query_hypothyroid, goitre, psych,
                   tsh_measured, tsh, tt4, fti]])
    model=jb.load("thyroid_modelrandomf.pkl")
    result=model.predict(arr)[0]
    if result==1:
        output="You have thyroid "
    else:
        output="you are normal"
    return render_template("intro.html", data=output)

if __name__=="__main__":
    app.run()
