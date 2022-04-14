import pandas as pd
import sklearn
import numpy as np

data= pd.read_excel('mldatas.xlsx')
d=data.iloc[:,2:3]

h=data.iloc[:,3:]
name_nationality=data.iloc[:,0:2]
skills=data.iloc[:,7:]
deneme=data.iloc[268,0:1]
age=data.iloc[:,3:4]
wage=data.iloc[:,4:5].values
ss=len(wage)
j=-1
k=0
l=0
count=0
countt1=0
sor=data.sort_values(by=['Wage'])


corr_skills=skills.corr()

Alınanwage= int (input("Bütçe Belirleyiniz:"))
AlınanPo=int (input("0-100 Arası bir Potansiyel Değer Giriniz:"))
for u in range(len(data)):
    
    if(sor.iloc[u,4:5].values<=Alınanwage):
        
        count=count+1
    

butceyegoresırala=sor.iloc[:count,:]

sor2=butceyegoresırala.sort_values(by=['Potantial Overall'])
sor2=sor2.reset_index(drop=True)



for u in range(len(sor2)):
    
    if(sor2.iloc[u,6:7].values<=AlınanPo):
        
        countt1=countt1+1
    

print(count)
print(countt1)


PreprocessingData=sor2.iloc[countt1:,:]
# #print(len(sor3))

UltimateData=PreprocessingData.iloc[:,7:]

PreprocessingData1=PreprocessingData.iloc[:,:2]




PreprocessingData3=PreprocessingData.iloc[:,3:7]


dataframe=pd.DataFrame(data)





from sklearn import preprocessing

ohe = preprocessing.OneHotEncoder()
d= ohe.fit_transform(d).toarray()
#le = preprocessing.LabelEncoder()
#d[:,0] = le.fit_transform(b.iloc[:,2].values)
position_ohe=pd.DataFrame(data=d,index=range(335),columns=["bek","cb","cm","gk","kanat","st"])
position_skills=pd.concat([position_ohe,h],axis=1)
data_ohe=pd.concat([name_nationality,position_skills],axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(skills,position_ohe,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'gini')
rfc.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix
y_pred = rfc.predict(x_test)
TotalResult=rfc.predict(UltimateData)
randomForest_cm = confusion_matrix(y_test.values.argmax(axis=1),y_pred.argmax(axis=1),normalize=('true'))
#print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(x_train,y_train)
#print(y_test.iloc[0,0])
knn_predict = knn.predict(x_test)

TotalResult1=knn.predict(UltimateData)
#print("confusion matrix for KNN")
knn_cm = confusion_matrix(y_test.values.argmax(axis=1),knn_predict.argmax(axis=1),normalize=('true'))


Y_train=y_train
X_train = np.array(X_train)
Y_train=np.array(Y_train)



X_train=X_train.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)




# from sklearn.svm import SVC
# svc = SVC(kernel='linear')
# svc.fit(X_train,Y_train)

# svm_predict = svc.predict(x_test)

# svm_cm = confusion_matrix(y_test.values.argmax(axis=1),svm_predict.argmax(axis=1),normalize=('true'))





# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train,y_train)

# naive_bayes_predict = gnb.predict(x_test)

# naive_bayes_cm = confusion_matrix(y_test.values.argmax(axis=1),naive_bayes_predict.argmax(axis=1),normalize=('true'))





from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(x_train,y_train)

decisionTree_predict = dtc.predict(x_test)

decision_tree_cm = confusion_matrix(y_test.values.argmax(axis=1),decisionTree_predict.argmax(axis=1),normalize=('true'))

TotalResult2=dtc.predict(UltimateData)



sonuc5=pd.DataFrame(data=TotalResult1,index=range(len(TotalResult1)),columns=["bek","cb","cm","gk","kanat","st"])

PreprocessingData2=pd.concat([sonuc5,PreprocessingData3.set_index(sonuc5.index)], axis=1)

Gosterilecekveri=pd.concat([PreprocessingData1.set_index(PreprocessingData2.index),PreprocessingData2],axis=1)
# birlestir=sonuc5.append(PreprocessingData3)




#import matplotlib.pyplot as pltimport scikitplot as skplt
#Normalized confusion matrix for the K-NN modelprediction_labels = knn_classifier.predict(X_test)skplt.metrics.plot_confusion_matrix(y_test, prediction_labels, normalize=True)plt.show()
