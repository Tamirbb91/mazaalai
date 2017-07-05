from watson_developer_cloud import VisualRecognitionV3
import numpy as np
import time
from pprint import pprint
import json
from os.path import join, dirname
from os import environ
from watson_developer_cloud import VisualRecognitionV3 as VR
# 94216d12207f5d595bf87c3b75736217771b1e9d --batkaas service
visual_recognition = VisualRecognitionV3('2016-05-20', api_key='43226187a2a9ba06d7d663cf93ec7dbfca4bead1')

# print(json.dumps(visual_recognition.classify(images_url='http://www.robots.ox.ac.uk/~timork/Saliency/images/face_0001.jpg'), indent=2, classifier_ids = 'etta_911372789'))
#sdk_vr = VR(version = '2016-05-20', api_key = '43226187a2a9ba06d7d663cf93ec7dbfca4bead1')

# resp = sdk_vr.classify(images_url = 'http://www.robots.ox.ac.uk/~timork/Saliency/images/face_0001.jpg')
# pprint(resp)

#*********************CREATING BATKAS CLASSIFIER*********************************
# with open('./eta.zip', 'rb') as eta, \
#     open('./mar.zip', 'rb') as mar:
#     visual_recognition.create_classifier('EtavsMar', eta_positive_examples=eta, negative_examples=mar)
#images_url = 'http://www.robots.ox.ac.uk/~timork/Saliency/images/face_0001.jpg'
# 
#with open("face.jpg", "r", encoding="ISO-8859-1") as f:
#resp = visual_recognition.classify(images_url = 'http://www.robots.ox.ac.uk/~timork/Saliency/images/face_0001.jpg', owners = 'me')
resp = visual_recognition.classify(images_file = open('face_01.jpg', 'rb'), owners = 'me')
pprint(resp)