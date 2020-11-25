from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import cv2
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import os
import random
from tkinter import *
import time

# print(cv2.__version__)

#loading dataset
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
print(digits.target.shape)


plt.figure(figsize=(20,4))
data=enumerate(zip(digits.data[0:5],digits.target[0:5]))
for index,(image,label) in data:
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('training: %i\n' %label,fontsize=20)
plt.show()

xtrain,xtest,ytrain,ytest=train_test_split(digits.data,digits.target,test_size=0.33)
print(len(xtrain)==len(ytrain))
print(len(ytest)==len(xtest))

logclf=LogisticRegression(solver='saga',max_iter=10000
                          ,n_jobs=4,multi_class='ovr')

logclf.fit(xtrain,ytrain)
l=logclf.predict(xtest[0].reshape(1,-1))[0]
print('pred=',l,'\tlabelactual=',ytest[0])


for i in range(len(xtest)):
    img=xtest[i]
    print('the label: ',ytest[i],'\tthe prediction: ',logclf.predict(img.reshape(1,-1))[0]) #clf.predict([res]) is an array
    # plt.imshow(np.reshape(img,(8, 8)), cmap=plt.cm.gray)
    # plt.title('prediction: %i' % logclf.predict([img]), fontsize=28)
    # plt.show()
print(logclf.score(xtest, ytest))

# yt=np.array(ytest)
# print(len(yt))
# yt=np.delete(yt,yt[np.where(yt==3)][0])
# print(len(yt))

yt=ytest
xt=xtest
truthva=True
while truthva:
    a=random.randint(0,9)
    b=random.randint(0,9)
    print(a,'+v',b,'=',end='')
    if a+b>9:
        print('wrong input')
        continue
    print(a+b)
    root=Tk()
    T=Text(root,height=7,width=7)
    T.pack()
    T.insert(END,str(str(a)+'+'+str(b)+'='+str(a+b)),'big')
    mainloop()
    ind=np.where(yt==int(a+b))[0][0]
    x=yt[ind]
    print(int(a+b)==int(x))
    y=xtest[ind]
    yt = np.delete(yt, yt[ind])
    xt=np.delete(xt,xt[ind])
    plt.imshow(np.reshape(y,(8, 8)), cmap=plt.cm.gray)
    plt.title('prediction: %i' % logclf.predict([y]), fontsize=28)
    plt.show()

print(yt)