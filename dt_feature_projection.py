import numpy as np
# import matplotlib.pyplot as plt
import pandas
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from sklearn.neighbors import NearestNeighbors

import os


class Ensemble :

	def __init__(self,num_classes):
		self.num_classifiers=0
		self.num_classes=num_classes
		self.classifier_list=[]

	def add_classifier(self,classifier):
		self.classifier_list.append(classifier)
		self.num_classifiers=self.num_classifiers+1


	def get_profile(self,sample):
		#Decision profile
		DP = np.zeros([self.num_classifiers, self.num_classes])
		for i in range(0,self.num_classifiers):
			#base classifier type
			classifier=self.classifier_list[i]
			#numpy array size 1 x num_classes
			prediction=classifier.predict(sample)
			DP[i,:]=prediction
		return DP


#not sure this is necessary yet, but oh well
class KerasClassifier :

	#wrap trained model in this class
	def __init__(self,model,num_classes):
		self.model=model
		self.num_classes=num_classes

	def predict(self,sample):
		return self.model.predict(np.array([sample])).reshape((1,self.num_classes))


#build decision templates
def build_decision_templates(ens,num_classifiers,num_classes,trainX,trainY):
	DT = np.zeros([num_classes, num_classifiers, num_classes])
	counts = np.zeros(num_classes)

	for i in range(trainX.shape[0]):
		# class to which trainX[i] belongs
	    c=np.argmax(trainY[i])
	    DT[c] = DT[c] + ens.get_profile(trainX[i])
	    counts[c] += 1

	for c in range(DT.shape[0]):
		if counts[c] !=0:
			DT[c] = np.divide(DT[c], counts[c])
		else :
			# print("Oops empty!")
			DT[c]=DT[c]+np.inf

	return DT

####Prediction rule 0 : simple kNN classifier with majority voting
def train_and_test_0(trainX,trainY,testX,testY,voting_choice=0,k_val=5):
	#build nearest neighbours data structure for efficient search
	nbrs = NearestNeighbors(n_neighbors=k_val, algorithm='ball_tree').fit(trainX)

	#predict
	tot_test=testX.shape[0]
	# tot_test=10
	correct=0

	for i in range(0,tot_test):
		sample=testX[i]

		[dist_nbrs,indices]= nbrs.kneighbors(sample.reshape(1,-1))
		# print(indices)
		X_nbrs = trainX[indices[0]]
		Y_nbrs= trainY[indices[0]]

		if voting_choice==0:
			total_votes = np.sum(Y_nbrs,axis=0)
		else:
			# print(np.reshape(dist_nbrs[0],(-1,1)))
			# print(Y_nbrs)
			temp=np.divide(Y_nbrs,np.reshape(dist_nbrs[0],(-1,1)))
			# print(temp)
			total_votes = np.sum(temp,axis=0)
		# print(total_votes)

		label = np.argmax(total_votes)
		true_label=np.argmax(testY[i])

		# print("Prediction :"+str(label))
		# print("Actual :"+str(true_label))
		if label==true_label:
			correct+=1

	accuracy=correct/tot_test
	print("Prediction rule 0")
	print("Total test accuracy is :")
	print(accuracy)



####Prediction rule 1 : Decision Templates (Kuncheva 2001)
def train_and_test_1(ens,num_classifiers,num_classes,trainX,trainY,testX,testY):

	DT=build_decision_templates(ens,num_classifiers,num_classes,trainX,trainY)
	#predict
	tot_test=testX.shape[0]
	correct=0

	for i in range(0,tot_test):
		sample=testX[i]
		DP= ens.get_profile(sample)
		# print(DT)
		# print(DP)
		distances = np.sum(((DT - DP)**2),axis=(1,2))
		label = np.argmin(distances,axis=(0))
		true_label=np.argmax(testY[i])
		# print(distances)
		# print("Prediction :"+str(label))
		# print("Actual :"+str(true_label))
		if label==true_label:
			correct+=1

	accuracy=correct/tot_test
	print("Prediction rule 1")
	print("Total test accuracy is :")
	print(accuracy)


####Prediction rule 2.1 : create DTs from the kNN in feature space
def train_and_test_2_1(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=5):
	#build nearest neighbours data structure for efficient search
	nbrs = NearestNeighbors(n_neighbors=k_val, algorithm='ball_tree').fit(trainX)

	#predict
	tot_test=testX.shape[0]
	# tot_test=10
	correct=0

	for i in range(0,tot_test):
		sample=testX[i]

		[dist_nbrs,indices]= nbrs.kneighbors(sample.reshape(1,-1))
		# print(indices)
		X_nbrs = trainX[indices[0]]
		Y_nbrs= trainY[indices[0]]

		#build decision templates
		DT_nbrs=build_decision_templates(ens,num_classifiers,num_classes,X_nbrs,Y_nbrs)

		DP= ens.get_profile(sample)
		# print(DT)
		# print(DP)
		distances = np.sum(((DT_nbrs - DP)**2),axis=(1,2))

		label = np.argmin(distances,axis=(0))
		true_label=np.argmax(testY[i])

		# print(distances)

		# print("Prediction :"+str(label))
		# print("Actual :"+str(true_label))
		if label==true_label:
			correct+=1

	accuracy=correct/tot_test
	print("Prediction rule 2.1")
	print("Total test accuracy is :")
	print(accuracy)




####Prediction rule 3 : find k-NN in DP space. find which classifiers did well for the corresponding points,
####					and then weigh them accordingly ending in a weighted majority voting
def train_and_test_3(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=5):
	#build the decision profile list
	DP_X_flat = np.zeros([trainX.shape[0],num_classifiers*num_classes])
	for i in range(0,trainX.shape[0]):
		DP_X_flat[i]= np.reshape(ens.get_profile(trainX[i]),num_classifiers*num_classes)

	#build nearest neighbours data structure for efficient search
	nbrs = NearestNeighbors(n_neighbors=k_val, algorithm='ball_tree').fit(DP_X_flat)

	#predict
	tot_test=testX.shape[0]
	# tot_test=30
	correct=0

	for i in range(0,tot_test):
		sample=testX[i]
		DP=ens.get_profile(sample)
		DP_flat= np.reshape(DP,(1,num_classifiers*num_classes))

		[dist_dp_nbrs,indices]= nbrs.kneighbors(DP_flat)
		# print(indices)
		# print(dist_dp_nbrs)

		X_nbrs = trainX[indices[0]]
		Y_nbrs= trainY[indices[0]]
		Y_nbrs_labels= np.argmax(Y_nbrs,axis=1)

		classifier_scores = np.zeros(num_classifiers)

		for j in range(0,X_nbrs.shape[0]):
			xnbr= X_nbrs[j]
			dp_xnbr=ens.get_profile(xnbr)
			# print(dp_xnbr)
			predictions = np.argmax(dp_xnbr,axis=1)
			# print(predictions)
			# print(Y_nbrs_labels[i])
			truth = np.zeros(num_classifiers)+Y_nbrs_labels[j]
			# print(predictions==truth)
			classifier_scores=classifier_scores+(predictions==truth)
			# print(classifier_scores)
			# break
		
		# print(DP)
		predictions = np.argmax(DP,axis=1)
		predictions_one_hot= np.zeros([num_classifiers,num_classes]) 
		for j in range(0,num_classifiers):
			predictions_one_hot[j][predictions[j]]=1
		# print(predictions_one_hot)
		# print(classifier_scores)
		weighted_votes= np.multiply(predictions_one_hot,np.reshape(classifier_scores,(-1,1)))
		# print(weighted_votes)
		total_votes=np.sum(weighted_votes,axis=0)
		# print(total_votes)
		label = np.argmax(total_votes,axis=(0))
		true_label=np.argmax(testY[i])

		# print("Prediction :"+str(label))
		# print("Actual :"+str(true_label))
		if label==true_label:
			correct+=1
		
		# print(correct)
	accuracy=float(correct)/tot_test
	print("Prediction rule 3")
	print("Total test accuracy is :")
	print(accuracy)




####Prediction rule 4 : find k-NN for each classifier in their DP space. find which classifiers did well in the neighboring points,
####					and then weigh them accordingly ending in a weighted majority voting
def train_and_test_4(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=5):
	#build the decision profile list
	DP_X = np.zeros([trainX.shape[0],num_classifiers,num_classes])
	for i in range(0,trainX.shape[0]):
		DP_X[i]= ens.get_profile(trainX[i])

	nbrs=[]
	for c in range(num_classifiers):
		# DP_X_c = np.zeros([trainX.shape[0],num_classes])
		DP_X_c = DP_X[:,c,:] #DP by classifier c for all training data

		#build nearest neighbours data structure for efficient search
		nbrs.append( NearestNeighbors(n_neighbors=k_val, algorithm='ball_tree').fit(DP_X_c) ) #might want to shift outside
	

	#predict
	tot_test=testX.shape[0]
	# tot_test = 30
	correct=0

	for i in range(0,tot_test):
		sample=testX[i]
		DP=ens.get_profile(sample)

		classifier_scores = np.zeros(num_classifiers)
		#for each classifier, find k-NN
		for c in range(num_classifiers):
			# # DP_X_c = np.zeros([trainX.shape[0],num_classes])
			# DP_X_c = DP_X[:,c,:] #DP by classifier c for all training data
			DP_c=ens.get_profile(sample)[c] #num_classes size
			DP_c=np.reshape(DP_c, (1,DP_c.shape[0]))

			# #build nearest neighbours data structure for efficient search
			# nbrs = NearestNeighbors(n_neighbors=k_val, algorithm='ball_tree').fit(DP_X_c) #might want to shift outside

			# print(i,' ',c)############
			[dist_dp_nbrs,indices]= nbrs[c].kneighbors(DP_c)

			X_nbrs = trainX[indices[0]]
			Y_nbrs= trainY[indices[0]]

			for j in range(0,X_nbrs.shape[0]): #to get prediction of classifier c for nbrs
				xnbr= X_nbrs[j]
				dp_xnbr_c=ens.get_profile(xnbr)[c] #num_classes size

				prediction = np.argmax(dp_xnbr_c)
				truth = np.argmax(Y_nbrs[j]) #because Y labels are one-hot
				classifier_scores[c] += int(prediction==truth)

		predictions = np.argmax(DP,axis=1) #by each classifier
		predictions_one_hot= np.zeros([num_classifiers,num_classes])
		for j in range(0,num_classifiers):
			predictions_one_hot[j][predictions[j]]=1
		
		weighted_votes = np.multiply(predictions_one_hot,np.reshape(classifier_scores,(-1,1)))
		total_votes=np.sum(weighted_votes,axis=0)
		label = np.argmax(total_votes,axis=(0))
		true_label=np.argmax(testY[i])
		if label==true_label:
			correct+=1
		
		# print(correct)

	accuracy=float(correct)/tot_test
	print("Prediction rule 4")
	print("Total test accuracy is :")
	print(accuracy)








def get_data(filename): # Satimage dataset
    data = pandas.read_csv(filename, sep=r"\s+", header=None)
    data = data.values

    dataX = np.array(data[:,range(data.shape[1]-1)])
    dataY = np.array(data[np.arange(data.shape[0]),data.shape[1]-1])

    # convert dataY to one-hot, 6 classes
    num_classes = 6
    dataY = np.array([x-2 if x==7 else x-1 for x in dataY]) # re-named class 7 to 6(5)
    dataY_onehot = np.zeros([dataY.shape[0], num_classes])
    dataY_onehot[np.arange(dataY_onehot.shape[0]), dataY] = 1

    return dataX, dataY_onehot


#get training data
trainX, trainY = get_data("data/sat.trn")
testX, testY = get_data("data/sat.tst")
num_classes = trainY.shape[1]

#build ensemble
def build_ensemble_1(trainX,trainY,num_classes,save=0):
	###############Build ensemble of MLPs with bagging (sort of)#####################
	ens = Ensemble(num_classes)
	num_classifiers=10
	half=int(num_classifiers/2)

	#employ bagging
	total_samples = trainX.shape[0]
	max_samples = (total_samples/2)
	max_samples=math.ceil(max_samples)

	#NN with single hidden layer,trained on all features x 5
	epochs=150#30
	for i in range(0,half):
		bag = np.random.randint(low=0, high=total_samples-1, size=max_samples)
		tX_part= trainX[bag]
		tY_part= trainY[bag]
		model1 = Sequential()
		model1.add(Dense(8, input_dim=trainX.shape[1], activation="sigmoid"))
		model1.add(Dense(num_classes, activation="softmax"))
		model1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
		model1.fit(tX_part,tY_part, epochs=epochs, batch_size=10) # epochs=150

		if(save):
			save_model_to_file(model1,i)
			print("Saved model to disk")
		ens.add_classifier(KerasClassifier(model1,num_classes))

	#NN with 2 hidden layers x 5
	for i in range(half,num_classifiers):
		bag = np.random.randint(low=0, high=total_samples-1, size=max_samples)
		tX_part= trainX[bag]
		tY_part= trainY[bag]
		model2 = Sequential()
		model2.add(Dense(8, input_dim=trainX.shape[1], activation="sigmoid"))
		model2.add(Dense(8, activation="sigmoid"))
		model2.add(Dense(num_classes, activation="softmax"))
		model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
		model2.fit(tX_part, tY_part, epochs=epochs, batch_size=10) # epochs=150

		if(save):
			save_model_to_file(model2,i)
			print("Saved model to disk")
		ens.add_classifier(KerasClassifier(model2,num_classes))
	#################################################################

	return ens


#build ensemble - load models from file
def build_ensemble_from_file(num_classifiers,num_classes):
	ens = Ensemble(num_classes)
	# num_classifiers=10
	half=int(num_classifiers/2)

	for i in range(num_classifiers):
		filename = 'saved_models/'+str(i)
		json_file = open(filename+'.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights(filename+'.h5')

		ens.add_classifier(KerasClassifier(loaded_model,num_classes))
	print("Loaded models from disk")

	return ens


def save_model_to_file(model,id):
	filename = 'saved_models/'+str(id)
	model_json = model.to_json()
	with open(filename+'.json', "w") as json_file:
		json_file.write(model_json)
	model.save_weights(filename+'.h5')

# ens = build_ensemble_1(trainX,trainY,num_classes,save=1)
ens = build_ensemble_from_file(10,num_classes)
num_classifiers=ens.num_classifiers



# train_and_test_1(ens,num_classifiers,num_classes,trainX,trainY,testX,testY)
# train_and_test_2_1(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=10)
# train_and_test_0(trainX,trainY,testX,testY,0,k_val=10)
# train_and_test_0(trainX,trainY,testX,testY,1,k_val=10)
train_and_test_3(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=10)
# train_and_test_4(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=10)



##Prediction rule 2.1 : weighted voting for DPs (simple voting is simple kNN)



################################
########### RESULTS

#PREDICTION RULE 3: 78.3%		(1567/2000)
#PREDICTION RULE 4: 75.8%		(1516/2000)
