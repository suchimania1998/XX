# import requests
# import json
# api_url = "https://deploytestt.herokuapp.com/frame"
# todo ={"frame":"[[[2,17]]]"} 
# response = requests.post(api_url,data=todo)

# data = response.json()   # api key imit.ctc-HQZK687Y  #api key - soumya777prusty-VOUUTR5T
# print(data)

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch

from PIL import Image

import what3words
geocoder = what3words.Geocoder("HQZK687Y")
res = geocoder.convert_to_3wa(what3words.Coordinates(51.484463, -0.195405))
print(res)
# print(res['coordinates']['lng'])   ypur disease detcted at latitude-51.0000 and longitude--0.199098878
# and click here to view the location-
# print(res['coordinates']['lat'])
print(res["map"])



@app.route('/fertilizer.html')
def fertilizer_recommendation():
    title = 'Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


# render fertilizer recommendation result page


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = ' Harvestify-Fertilizer Suggestion '
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]
    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)