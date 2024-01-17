import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from scipy.io import loadmat

data = loadmat('3raPruebajose.mat')
print(data)

Blanc = np.vstack((data['MuestraB16'], data['MuestraB17'], data['MuestraB18'], data['MuestraB19'], data['MuestraB20'], 
                   data['MuestraB21'], data['MuestraB22'], data['MuestraB23'], data['MuestraB24']))

Blanc = Blanc[:, 2:19]

Letter = np.vstack((data['MuestraL16'], data['MuestraL17'], data['MuestraL18'], data['MuestraL19'], data['MuestraL20'], 
                    data['MuestraL21'], data['MuestraL22'], data['MuestraL23'], data['MuestraL24']))
Letter = Letter[:, 2:19]

Multip = np.vstack((data['MuestraM16'], data['MuestraM17'], data['MuestraM17'], data['MuestraM19'], data['MuestraM20'], 
                    data['MuestraM21'], data['MuestraM22'], data['MuestraM23'], data['MuestraM24']))
Multip = Multip[:, 2:19]
#print(Multip)



from window import f
NoagreVen = f(Letter)
AgresiVen = f(Multip)

print(NoagreVen)

DatosFinal = np.vstack((AgresiVen, NoagreVen))

# Dimension reduction
pca = PCA()
Variablesfinal = pca.fit_transform(DatosFinal)
Clase1 = pca.transform(AgresiVen)
Clase2 = pca.transform(NoagreVen)

# Selection of training and test classes
Clase1entre = Clase1[:round(Clase1.shape[0] * 0.8), :]
Clase1test = Clase1[round(Clase1.shape[0] * 0.8):, :]

Clase2entre = Clase2[:round(Clase2.shape[0] * 0.8), :]
Clase2test = Clase2[round(Clase2.shape[0] * 0.8):, :]

testcla = np.vstack((Clase1test, Clase2test))

# Classifiers - Support Vector Machine
data3 = np.vstack((Clase1entre, Clase2entre))
theclass = np.zeros(data3.shape[0])
theclass[:data3.shape[0] // 2] = 1

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(data3, theclass, test_size=0.01,random_state=109) # 70% training and 30% test



#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

y_pred = clf.predict(testcla)

#cl = SVC(kernel='rbf', C=np.inf)

theclass2 = np.zeros(testcla.shape[0])
theclass2[:testcla.shape[0] // 2] = 1
theclass2t = theclass2.T

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(theclass2t, y_pred))


print(y_pred)
print(theclass2)
print(y_test)
#cl.fit(data3, theclass)

#svmclasi = cl.predict(np.vstack((Clase1test, Clase2test)))

# Evaluation Metrics
theclass2 = np.zeros(Clase1test.shape[0] + Clase2test.shape[0])
theclass2[:Clase1test.shape[0]] = 1

#print(X_test)

import matplotlib.pyplot as plt

plt.hist(AgresiVen, bins=5)
plt.show()

plt.hist(NoagreVen, bins=5)
plt.show()
#plt.hist(Clase2entre, bins=5)