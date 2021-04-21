# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 09:43:00 2021

@author: Deepnil Vasava
"""
from flask import Flask, render_template, request, redirect, url_for
from ReadEmail import read_email_from_gmail
import pickle
import numpy as np

app = Flask(__name__)

loaded_model = pickle.load(open("finalized_model.sav", "rb"))


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/contact')
def Contact():
    return render_template('Contact.html')

@app.route('/about')
def About():
    return render_template('About.html')

@app.route('/login',methods = ['POST', 'GET'])
def Login():
    # if request.method == 'POST':
    #     emailid = request.form['email']
    #     passwd = request.form['pwd']
    #     return redirect(url_for('success',useremail = emailid,pwd=passwd))
    # else:
    return render_template('Login.html')


# @app.route('/result', methods=['POST', 'GET'])
# def result():
#     if request.method == 'POST':
#         result = request.form['Data']
#         l = []
#         l.append(result)
#         result_pred = loaded_model.best_estimator_.predict(np.array(l))
#         return render_template("text_pred.html", result=result_pred)

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        emailid = request.form['email']
        passwd = request.form['pwd']

        result = read_email_from_gmail(emailid, passwd)

        return render_template("result.html",result = result, Len = len(result))

@app.route('/success/<useremail>')
def success(useremail,pwd):
    return render_template("result.html",result = useremail)

if __name__ == '__main__':
    app.run()
    # host = "localhost", port = 8000, debug = True
