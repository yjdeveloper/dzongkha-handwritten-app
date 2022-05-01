from django.shortcuts import render
import base64
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from keras.preprocessing import image
# from django.core.files.storage import FileSystemStorage
from keras import backend as K
import re

classes = {
    '0':'༠',
    '1': '༡',
    '2': '༢',
    '3': '༣',
    '4': '༤',
    '5': '༥',
    '6': '༦',
    '7': '༧',
    '8': '༨',
    '9': '༩',
}

# Create your views here.
# Create your views here.
def index(request):
    return render(request, 'index.html', {})

def about(request):
    return render(request, 'about.html', {})


def predict(request):
    K.clear_session()
    image_data = request.POST['image_data']
    image_data = re.sub("^data:image/png;base64,", "", image_data)
    image_data = base64.b64decode(image_data)
    
    fh = open("imageToSave.png", "wb")
    fh.write(image_data)
    fh.close()

    img = cv.imread("imageToSave.png", cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (64, 64), interpolation = cv.INTER_AREA)
    x = img / 255
    x = x.reshape(1, 64, 64, 1)
    model = load_model('startup/model/dzo-net.h5')
    pred = model.predict(x)

    predictedLabel = str(np.argmax(pred[0]))
    label = classes[predictedLabel]
    return render(request, 'index.html', context = {
        'label': label
    })
