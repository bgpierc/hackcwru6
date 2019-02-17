import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
import tensorflow as tf
import os
import extract_plates
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

letters = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']


keys = { 0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A', 11:'B', 12:'C', 13:'D',
            14:'E', 15:'F', 16:'G', 17:'H', 18:'J', 19:'K', 20:'L', 21:'M', 22:'N', 23:'P', 24:'Q', 25:'R', 26:'S', 27:'T',
            28:'U', 29:'V', 30:'W', 31:'X', 32:'Y', 33:'Z'}
# onehot_encoder = OneHotEncoder(sparse=False)
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(list(letters))
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

#print(onehot_encoded[2])


def get_training_data(filepath = '/home/bp0017/Documents/nigerian_prince/License-Plate-Recognition-Nigerian-vehicles-master/training_data/train20X20/'):
	train_features = []
	train_labels = []
	test_features = []
	test_labels = []
	for letter in letters:
		for i in range(5):
			path = filepath+letter+"/"+letter + "_"+str(i) + ".jpg"
			img = cv2.imread(path,0)
			ret,binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
			# cv2.imshow("bin",binary)
			# cv2.waitKey(0)
			train_features.append(binary.reshape(-1))
			train_labels.append(letter)

		for j in range(5,10):
			path = filepath+letter+"/"+letter + "_"+str(i) + ".jpg"
			img = cv2.imread(path,0)
			ret,binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
			# cv2.imshow("bin",binary)
			# cv2.waitKey(0)
			test_features.append(binary.reshape(-1))
			#train_features.append(binary)
			test_labels.append(letter)
			
		#test_labels.append(letter)
	train_features = np.asarray(train_features)
	#print(train_features.shape)
		# train_features = np.asarray(train_features)
		# train_features = rgb2gray(train_features)
		# train_features = np.asarray(train_features)
		#print(train_features)
	return (np.array(train_features), np.array(train_labels),np.array(test_features),np.array(test_labels))


def svm_train():
	train_features,train_labels,test_features,test_labels = get_training_data()
	svc_model = SVC(kernel='linear', probability=True)
	accuracy_result = cross_val_score(svc_model, train_features, train_labels,cv=4 )

	svc_model.fit(train_features, train_labels)

	current_dir = '/home/bp0017/Documents/hackathon/'
	save_directory = os.path.join(current_dir, 'models/svc/')
	if not os.path.exists(save_directory):
		os.makedirs(save_directory)
	joblib.dump(svc_model, save_directory+'/svc.pkl')

def svm_test():
	test_features,test_labels = get_test_data()
	current_dir = os.path.dirname(os.path.realpath(__file__))
	model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
	model = joblib.load(model_dir)
	classification_result = []
	for char in test_features:
	    char = char.reshape(1, -1);
	    result = model.predict(char)
	    classification_result.append(result)
	print(classification_result)

	for pred,act in zip(classification_result,test_labels):
		print("Pred: "+ str(pred) + " Actual: " + str(act) + "\n")

def get_test_data():
	chars = extract_plates.get_chars(extract_plates.getPlate()[0])
	print(chars)
	return chars,[3,7,8,'D','X','R']

#svm_train()
svm_test()