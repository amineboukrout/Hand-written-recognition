from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import cv2
from sklearn.linear_model import LogisticRegression
import numpy as np

print(cv2.__version__)
# mnist = fetch_mldata("MNIST original")
# x_train, y_train = mnist.data / 255., mnist.target

from sklearn.datasets import load_digits
digits = load_digits()

print(digits.data.shape)
print(digits.target.shape)

'''plt.figure(figsize=(20,4))
data=enumerate(zip(digits.data[0:5],digits.target[0:5]))
for index,(image,label) in data:
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('training: %i\n' %label,fontsize=20)
plt.show()
plt.close()'''

'''xtrain=digits.data[0:int(0.75*len(digits.data))]
ytrain=digits.target[0:int(0.75*len(digits.target))]
xtest=digits.data[int(0.75*len(digits.data)):len(digits.data)]
ytest=digits.target[int(0.75*len(digits.target)):len(digits.target)]'''

xtrain,xtest,ytrain,ytest=train_test_split(digits.data,digits.target,test_size=0.33)
print(len(xtrain)==len(ytrain))
print(len(ytest)==len(xtest))

#xtrain,ytrain=xtrain/255.,ytrain
'''plt.figure()
plt.imshow(np.reshape(xtrain[0],(8,8)),cmap=plt.cm.gray)
plt.show()'''

#https://github.com/monycky/ann-handwritten/blob/master/Classifier.py
#https://github.com/ankitshaw/Document-Scanner-and-OCR/blob/master/ocr.py
#https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

'''mlpclf=MLPClassifier(solver='lbfgs', activation='logistic')
mlpclf.fit(xtrain, ytrain)

for i in range(len(xtest)):
    img=xtest[i]
    res=cv2.resize(img,dsize=(8,8),interpolation=cv2.INTER_CUBIC)
    res=res.reshape((8*8))
    res=res/255.
    print(ytest[i], mlpclf.predict([res])) #clf.predict([res]) is an array
    plt.imshow(np.reshape(img,(8, 8)), cmap=plt.cm.gray)
    # plt.title('prediction: %i' % clf.predict([res]), fontsize=28)
    # plt.show()
print(mlpclf.score(xtest, ytest))'''

##### classifer 2 logistic regression
logclf=LogisticRegression()
logclf.fit(xtrain,ytrain)
l=logclf.predict(xtest[0].reshape(1,-1))
print('pred=',l,'\tlabelactual=',ytest[0])

for i in range(len(xtest)):
    img=xtest[i]
    print(ytest[i], logclf.predict(img.reshape(1,-1))[0]) #clf.predict([res]) is an array
    plt.imshow(np.reshape(img,(8, 8)), cmap=plt.cm.gray)
    plt.title('prediction: %i' % logclf.predict([img]), fontsize=28)
    plt.show()
print(logclf.score(xtest, ytest))