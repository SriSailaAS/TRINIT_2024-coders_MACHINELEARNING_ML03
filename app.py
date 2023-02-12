from flask import Flask, render_template, session, redirect,request
import json
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
filename = "XGBoost.pkl"
loaded_model = pickle.load(open(filename, "rb"))
ffn="xgb_pipeline.pkl"
fertilizer_model=pickle.load(open(ffn, "rb"))
pricedata=pd.read_csv("cost15.csv")

indtoferti={0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 5: 'DAP', 6: 'Urea'}

crops=['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

croptocost={'apple':'Apple','watermelon':'Water+Melon','mungbean':'Gram+Raw(Chholia)','banana':'Banana','blackgram': 'Alasande+Gram','coconut': 'Coconut','coffee' :'Coffee','grapes' :'Grapes','jute': 'Jute','maize':'Maize','mango ':'Mango','orange': 'Orange','papaya':'Papaya','pomegranate': 'Pomegranate','rice' :'Rice'}

soiltoind={'Black':0, 'Clayey':1,'Loamy':2, 'Red':3,'Sandy':4}

croptoind={'Barley':0,'Cotton':1,'Ground Nuts':2, 'Maize':3, 'Millets':4,'Oil seeds':5,'Paddy':6,'Pulses':7, 'Sugarcane':8, 'Tobacco':9, 'Wheat':10}

@app.route("/input", methods=["GET", "POST"])
def input():
    if request.method == "POST":
        with open("Rainfall.json", "r") as file:
            data = json.load(file)
        user_input=request.form["states"]+"_"+request.form["districts"]
        season=request.form["seasons"]
        print(data[user_input][season])
        with open("temphum.json","r") as file1:
            data1=json.load(file1)
        
        states=request.form["states"]
        with open("NPK.json","r") as file2:
            data2=json.load(file2)
    
        predictions=[
            {'Temperature' : data1[states]["Temperature"]},
            {'Humidity':data1[states]["Humidity"]},
            {'Rainfall':data[user_input][season]},
            {'ph':data2[states]["pH"]},
            {'Nitrogen content':data2[states]['N']},
            {'Phosphorous content':data2[states]['P']},
            {'Potassium Content':data2[states]['K']}
        ]
    return render_template('suggestions.html', predictions=predictions)

@app.route("/base")
def index():
    return render_template("base.html")

@app.route("/")    
def home():
    return render_template("index.html")  

@app.route("/about")
def about():
    return render_template("aboutus.html")

@app.route("/predictpage")
def predict():
    return render_template("predictpage.html")

@app.route("/fertilizer")
def predfer():
    return render_template("fertilizer.html")

@app.route("/fertipredict",methods = ['POST'])
def fertipredict():
    msg=""""""
    if request.method=="POST":
        fdata=[[]]
        fdata[0].append(int(request.form["temperature"]))
        fdata[0].append(int(request.form["humidity"]))
        fdata[0].append(int(request.form["Moisture"]))
        fdata[0].append(soiltoind[request.form["soils"]])
        fdata[0].append(croptoind[request.form["crops"]])
        fdata[0].append(int(request.form["N"]))
        fdata[0].append(int(request.form["K"]))
        fdata[0].append(int(request.form["P"]))
        fdata=np.array(fdata)
        pred=fertilizer_model.predict(fdata)[0]
        msg+="<b>"+"suggested fertilizer is "+str(indtoferti[pred])+"</b>"
        return msg
@app.route("/exactpredict",methods = ['POST'])
def predictpredicts():
    message = """"""
    if request.method == 'POST':
        rdata={}
        rdata["N"]=[int(request.form["N"])]
        rdata["P"]=[int(request.form["P"])]
        rdata["K"]=[int(request.form["K"])]
        rdata["temperature"]=[float(request.form["temperature"])]
        rdata["humidity"]=[float(request.form["humidity"])]
        rdata["ph"]=[float(request.form["ph"])]
        rdata["rainfall"]=[float(request.form["rainfall"])]
        state=request.form["states"]
        district=request.form["districts"]
        district=district[0]+district[1:].lower()
        df=pd.DataFrame.from_dict(rdata)
        predprob=loaded_model.predict_proba(df)[0]
        top3=np.flip(np.argsort(predprob))[:3]
        predprob=sorted(predprob,reverse=True)
        best_crop=crops[top3[0]]
        crop2=crops[top3[1]]
        crop3=crops[top3[2]]
        ind=0
        for prob in predprob[:3]:
            if prob>0.1:
                prob=prob*100
                message+="<b>"+"Best crop:"+str(crops[top3[ind]])+"</b>"+str(prob)[:4]+"%"+"<br>"
            else:
                message+="Average crops:"+str(crops[top3[ind]])+"  "
            ind+=1
        for ci in [best_crop]:
            if ci in croptocost.keys():
                cicrop=croptocost[ci]
                if pricedata[pricedata["district"]==district].shape[0]>0 and pricedata[pricedata["commodity_name"]==cicrop].shape[0]>0:
                    message+="<br>"+"<b>"+"District:"+district+"</b>"+"<br>"
                    datadest=pricedata[(pricedata["commodity_name"]==cicrop)&(pricedata["district"]==district)]
                    datadest["date"].str[-2:]
                    yearset=set(datadest["date"].str[-2:])
                    for y in yearset:
                        datay=datadest[datadest["date"].str[-2:]==y]
                        monthset=set(datay["date"].str[:2])
                        message+="20"+str(y)+":<br>"
                        for m in monthset:
                            cost=datay[datay["date"].str[:2]==m]
                            if m[1]=="/":
                                m=m[:1]
                            ma=cost["modal_price"].max()
                            mi=cost["modal_price"].min()
                            av=cost["modal_price"].mean()
                            if ma<500 and mi<500 and av<500:
                                message+=str(m)+"th month prices"+"::"+" Max:"+str(ma)+" Mean:"+str(av)[:4]+" Min:"+str(mi)+"<br>"
                                print(ma,mi,av)


          
        return message
if __name__ == "__main__":
    app.run(debug=True)