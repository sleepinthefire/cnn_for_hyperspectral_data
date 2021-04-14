from django.shortcuts import render, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile

from .models import Info
from .forms import SpectralUploadForm

import numpy as np
from PIL import Image
from io import BytesIO
import random
import base64

from plotly.offline import plot
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model


#First page, album of HS images
@login_required(login_url='/admin/login/')
def index(request):
    images = Info.objects.filter(user=request.user)
    return render(request, 
                  'analysis/index.html', 
                  { 'images':images,
                    'login_user':request.user, })

#data upload form
@login_required(login_url='/admin/login/')
def upload(request):

    if (request.method == 'POST'):
        form = SpectralUploadForm(request.POST, request.FILES)
        if form.is_valid():
            spectral_info = Info()
            spectral_info.user = request.user
            spectral_info.name = request.POST['name']
            spectral_info.spectral = request.FILES['spectral']
            spectral_info.metadata = request.FILES['metadata']
            spectral_info.tag = request.POST['tag']
            spectral_info.comment = request.POST['comment']

            date, integration = get_date_integration(request.FILES['metadata'])
            spectral_info.date = date
            spectral_info.integration = integration

            rgb = false_rgb(load_specim(request.FILES['spectral']))
            rgb_converted = img_converter(rgb)
            img_file = img_content(rgb_converted)
            spectral_info.image = img_file

            spectral_info.save()

    else:
        form = SpectralUploadForm()
        
    params = {
        'form':form,
    }
    return render(request, 'analysis/upload.html', params)

#see in detail of HS
@login_required(login_url='/admin/login/')
def detail(request, image_id):
    items = Info.objects.filter(user=request.user)
    loc = request.POST.getlist('loc') 
    roi_img_url = {}
    classifier = None

    #roisは切り取ったスペクトルの画像,　roi_imgは選択した場所を表示したバイナリデータのurl
    if (request.method == 'POST'):
        image = get_object_or_404(Info, pk=image_id)
        rois = [] 
        processing = request.POST['processing']
        classifier = request.POST['classifier']

        #load trained data
        if classifier == "chrysotile":
            model = load_model('analysis/static/analysis/model/chrysotile_model.h5')
        elif classifier == "serpentine":
            model = load_model('analysis/static/analysis/model/serpentine_model.h5')
        elif classifier =='c-s':
            model = load_model('analysis/static/analysis/model/chrysotile__serpentine_model.h5')

        roi_img = {}
        for a in loc:
            dat = a.split(':')[0]
            cod = a.split(':')[1]
            cod = cod[1:-1]
            if dat in roi_img.keys():
                roi_img[dat] = roi_img[dat] + [ cod ]
            elif dat not in roi_img.keys():
                roi_img[dat] = [ cod ]
        
        for dat, locs in roi_img.items():
            data = Info.objects.get(pk=dat)
            spectral = data.spectral
            markedImg = roi_mark(false_rgb(load_specim(spectral)), locs)
            roi_img_url[dat] = markedImg

        for a in loc:
            dat = a.split(':')[0]
            cod = a.split(':')[1]
            cod = cod[1:-1]
            info = Info.objects.get(pk=dat) 
            array_data = info.spectral
            array = load_specim(array_data)
            cod = cod.split(',')
            cod = map((lambda x:int(x)), cod) 
            rois.append(roi(array, cod))

        minmax = False
        snv = False
        if processing == 'minmax':
            minmax = True
        elif processing == 'snv':
            snv = True

        graphData = {}
        predict = {}
        
        if classifier != "None":
            for name, intensity in zip(loc, roi_mean(rois, minmax=minmax, snv=snv)):
                graphData[name] = intensity
                result = model.predict(np.array(intensity).reshape(1,204,1))
                predict[name] = result.argmax()

        htmlResp = mkGraph(graphData)


    else:
        processing = 'Normal'
        image = get_object_or_404(Info, pk=image_id)
        graphData = {}
        predict = {}
        htmlResp = mkGraph(graphData)

    params = {"image":image, "login_user":request.user, 'graph':htmlResp, "classifier":classifier,
              'items':items, 'locs':loc, "markedImg":roi_img_url, "predict":predict}    
    return render(request, "analysis/detail.html", params)

#read XML and get integaration and date info
def get_date_integration(xml_file):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file)
    root = tree.getroot()
    date = root[2][0].text
    integration = root[2][4].text
    return date, integration + 'ms'

#load specim iq dat date file
def load_specim(dat_file):
    raw = np.fromfile(dat_file, dtype='float32').reshape(512,204,512)
    raw = np.rot90(raw, 1, axes=(1,2))
    raw = np.rot90(raw, 1, axes=(0, 1))
    raw = np.rot90(raw, 2, axes=(1, 2))  
    return raw

#make false rgb array from dat file
def false_rgb(ndarray):
    r = ndarray[:, :, 70].reshape(512,512,1)
    g = ndarray[:, :, 53].reshape(512,512,1)
    b = ndarray[:, :, 19].reshape(512,512,1)
    rgb = np.concatenate([r, g, b], axis=2)
    return rgb

#convert false rgb array in range of 0-1 to 0-255
def img_converter(ndarray):
    armax = ndarray.max()
    armin = ndarray.min()
    
    denominator = armax -armin
    numerator = ndarray - armin
    
    converted = numerator / denominator
    converted = converted * 255
    converted = converted.astype('uint8')
    return converted


#image file for django's Image field
def img_content(ndarray):
    img = Image.fromarray(ndarray)
    img_io = BytesIO()
    img.save(img_io, format='JPEG', quality=100)
    img_content = ContentFile(img_io.getvalue(), 'img.jpg')
    return img_content

#extract regeion of interest from array
def roi(array, coordinate):
    (x, y, w, h) = coordinate
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    roi = array[y:y+h, x:x+w, :]

    return roi

#get averaged intensity    
def roi_mean(rois, minmax=False, snv=False):
    intensities = []
    for i in range(len(rois)):
        roi = rois[i]
        intensity = []
        for layer in range(roi.shape[2]):
            intensity.append(np.mean(roi[:, :, layer]))
        if minmax == True:
            from sklearn import preprocessing
            intensity = preprocessing.minmax_scale(intensity)
        if snv == True:
            from sklearn import preprocessing
            intensity = preprocessing.scale(intensity)
        intensities.append(intensity[::-1]) 
    return intensities     


#maek ploty graph html
def mkGraph(dict_items):  
    fig = go.Figure()
    for name, intensity in dict_items.items():
        fig.add_trace(go.Scatter(x=wavelength, y=intensity, name=name))
    fig.update_layout(legend=dict(x=0, xanchor='left', y=-0.1, yanchor='top'))
    plot_fig = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_fig


#
def roi_mark(rgbArray, coordinates):
    for cod in coordinates:
        cod = cod.split(',')
        cod = map((lambda x:int(x)), cod)
        (x, y, w, h) = cod
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        line = 1

        rgbArray[y-line:y, x-line:x+w+line, :] = 1
        rgbArray[y:y+h, x-line:x, :] = 1
        rgbArray[y+h:y+h+line, x-line:x+w+line, :] = 1
        rgbArray[y:y+h, x+w:x+w+line, :] = 1
    rgbArray = img_converter(rgbArray)
    img = Image.fromarray(rgbArray)
    img_io = BytesIO()
    img.save(img_io, format='JPEG', quality=100)

    data64 = base64.b64encode(img_io.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8') 
        

wavelength = [397.32,
    400.20,
    403.09,
    405.97,
    408.85,
    411.74,
    414.63,
    417.52,
    420.40,
    423.29,
    426.19,
    429.08,
    431.97,
    434.87,
    437.76,
    440.66,
    443.56,
    446.45,
    449.35,
    452.25,
    455.16,
    458.06,
    460.96,
    463.87,
    466.77,
    469.68,
    472.59,
    475.50,
    478.41,
    481.32,
    484.23,
    487.14,
    490.06,
    492.97,
    495.89,
    498.80,
    501.72,
    504.64,
    507.56,
    510.48,
    513.40,
    516.33,
    519.25,
    522.18,
    525.10,
    528.03,
    530.96,
    533.89,
    536.82,
    539.75,
    542.68,
    545.62,
    548.55,
    551.49,
    554.43,
    557.36,
    560.30,
    563.24,
    566.18,
    569.12,
    572.07,
    575.01,
    577.96,
    580.90,
    583.85,
    586.80,
    589.75,
    592.70,
    595.65,
    598.60,
    601.55,
    604.51,
    607.46,
    610.42,
    613.38,
    616.34,
    619.30,
    622.26,
    625.22,
    628.18,
    631.15,
    634.11,
    637.08,
    640.04,
    643.01,
    645.98,
    648.95,
    651.92,
    654.89,
    657.87,
    660.84,
    663.81,
    666.79,
    669.77,
    672.75,
    675.73,
    678.71,
    681.69,
    684.67,
    687.65,
    690.64,
    693.62,
    696.61,
    699.60,
    702.58,
    705.57,
    708.57,
    711.56,
    714.55,
    717.54,
    720.54,
    723.53,
    726.53,
    729.53,
    732.53,
    735.53,
    738.53,
    741.53,
    744.53,
    747.54,
    750.54,
    753.55,
    756.56,
    759.56,
    762.57,
    765.58,
    768.60,
    771.61,
    774.62,
    777.64,
    780.65,
    783.67,
    786.68,
    789.70,
    792.72,
    795.74,
    798.77,
    801.79,
    804.81,
    807.84,
    810.86,
    813.89,
    816.92,
    819.95,
    822.98,
    826.01,
    829.04,
    832.07,
    835.11,
    838.14,
    841.18,
    844.22,
    847.25,
    850.29,
    853.33,
    856.37,
    859.42,
    862.46,
    865.50,
    868.55,
    871.60,
    874.64,
    877.69,
    880.74,
    883.79,
    886.84,
    889.90,
    892.95,
    896.01,
    899.06,
    902.12,
    905.18,
    908.24,
    911.30,
    914.36,
    917.42,
    920.48,
    923.55,
    926.61,
    929.68,
    932.74,
    935.81,
    938.88,
    941.95,
    945.02,
    948.10,
    951.17,
    954.24,
    957.32,
    960.40,
    963.47,
    966.55,
    969.63,
    972.71,
    975.79,
    978.88,
    981.96,
    985.05,
    988.13,
    991.22,
    994.31,
    997.40,
   1000.49,
   1003.58]