import os
import matplotlib.image as mpimg
from flask import Flask, render_template, request, redirect
from flask_socketio import SocketIO, send, join_room
from flask import Flask, flash, redirect, render_template, request, session, abort,url_for
import os
from flask import send_file

from PIL import Image, ImageTk
import pandas as pd
import sqlite3

import matplotlib.pyplot as plt
import csv
import appendingpdf as ap
from PIL import Image
import io
import es as esw
import esdemo as puti
import mergingpdf as mpdf
from inference import get_prediction
from commons import format_class_name, transform_image
import pdfdemo as pdfgen
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import pickle
import time
import json
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './input'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
socketio = SocketIO(app)
@app.route('/')
def index():
	if not session.get('logged_in'):
		return render_template("login.html")
	else:
		return render_template('home.html')
@app.route('/registerpage',methods=['POST'])
def reg_page():
    return render_template("register.html")
	
@app.route('/loginpage',methods=['POST'])
def log_page():
    return render_template("login.html")
@app.route('/Back')
def back():
    return render_template("home.html")
    
    
   
@app.route('/register',methods=['POST'])
def reg():
	name=request.form['name']
	username=request.form['username']
	password=request.form['password']
	email=request.form['emailid']
	mobile=request.form['mobile']
	conn= sqlite3.connect("Database")
	cmd="SELECT * FROM login WHERE username='"+username+"'"
	print(cmd)
	cursor=conn.execute(cmd)
	isRecordExist=0
	for row in cursor:
		isRecordExist=1
	if(isRecordExist==1):
	        print("Username Already Exists")
	        return render_template("usernameexist.html")
	else:
		print("insert")
		cmd="INSERT INTO login Values('"+str(name)+"','"+str(username)+"','"+str(password)+"','"+str(email)+"','"+str(mobile)+"')"
		print(cmd)
		print("Inserted Successfully")
		conn.execute(cmd)
		conn.commit()
		conn.close() 
		return render_template("inserted.html")

@app.route('/patreg',methods=['POST'])
def patreg():
	pid=request.form['pid']
	pname=request.form['pname']
	age=request.form['age']
	gender=request.form['g1']
	session['patname'] = request.form['pname']
	pdfgen.process(pid,pname,age,gender)
	return render_template("index.html")
	
@app.route('/login',methods=['POST'])
def log_in():
	#complete login if name is not an empty string or doesnt corss with any names currently used across sessions
	if request.form['username'] != None and request.form['username'] != "" and request.form['password'] != None and request.form['password'] != "":
		username=request.form['username']
		password=request.form['password']
		conn= sqlite3.connect("Database")
		cmd="SELECT username,password FROM login WHERE username='"+username+"' and password='"+password+"'"
		print(cmd)
		cursor=conn.execute(cmd)
		isRecordExist=0
		for row in cursor:
			isRecordExist=1
		if(isRecordExist==1):
			session['logged_in'] = True
			# cross check names and see if name exists in current session
			session['username'] = request.form['username']
			return redirect(url_for('index'))

	return redirect(url_for('index'))
	
@app.route("/logout",methods=['POST'])
def log_out():
    session.clear()
    return render_template("login.html")

@app.route('/segment', methods=['GET', 'POST'])
def segment_file():
	uname= session['patname']
	class_name=session['class']
	path=request.form['imf']
	print("Path==",path)
	esw.process(path,uname)
	ap.process(uname,uname,class_name)
	puti.process(uname)
	mpdf.process(uname)
	return render_template("result.html",class_name="Report_Generated")
	
	
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
         
              
        uname= session['patname']
        if not file:
                return
        img_bytes = file.read()
        file_tensor = transform_image(image_bytes=img_bytes) #######
        class_name = get_prediction(file_tensor)
        session['class'] = class_name
        return render_template('result.html',class_name=class_name)
@app.route('/download', methods=['POST'])
def download_file():
	path=""
	uname= session['patname']
	print("uname==",uname)
	path=str(uname)+"result.pdf"
	return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
