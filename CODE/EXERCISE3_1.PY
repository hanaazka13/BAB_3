from sklearn import svm

# Height[cm], Weight[kg], Shoesize[UK]
X = [
    [170, 70, 10],  
    [180, 80, 12],  
    [170, 65, 8],   
    [160, 55, 7],   
    [175, 75, 11], 
    [155, 50, 6]    
]

# 0 = Male, 1 = Female
y = [0, 0, 1, 1, 0, 1]

# Inisialisasi dan training model SVM
clf = svm.SVC()
clf.fit(X, y)

# Prediksi 
p = clf.predict([[170, 70, 11]])
print(p)