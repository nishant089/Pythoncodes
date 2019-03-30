import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

my_list = list(range(1000000))
my_arr = np.array(range(1000000))

%time for i in range(10): my_list2 = my_list * 2
%time for i in range(10): my_arr2 = my_arr * 2

plt.scatter([1, 2, 3], [4, 5, 6])
plt.show()

plt.plot([1, 2, 3], [4, 5, 6])
plt.show()

a = pd.Series([1, 2, 3, 4, 5], ['a', 'b', 'c', 'd', 'e'])

b = pd.DataFrame({1 : [1, 2, 3, 4, 5],
                  2 : [1, 2, 3, 5, 5]})

c = pd.DataFrame([[1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5]])


dataset = pd.read_csv('dataset/censusdata.csv',names = ['age','workclass','fnlwgt','education',
                                                        'education-num','marital-status',
                                                        'occupation','relationship','race',
                                                        'sex','capital-gain','capital-loss',
                                                        'hours-per-week','native-country','salary']) 

print(dataset.describe())

plt.scatter(dataset['age'], dataset['workclass'])
plt.show()

pd.scatter_matrix(dataset)
dataset.replace("?","NaN")
X = dataset.iloc[:, [0,2,4,10,11,12]].values
y = dataset.iloc[:, [1,3,5,6,7,8,9,13,14]].values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imp.fit(X[:,0:5])
X[:,0:5] = imp.transform(X[:,0:5])

print(X)
print(y)

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
y[:,7] = lab.fit_transform(y[:,7])
y[:,6] = lab.fit_transform(y[:,6])
y[:,5] = lab.fit_transform(y[:,5])
y[:,4] = lab.fit_transform(y[:,4])
y[:,3] = lab.fit_transform(y[:,3])
y[:,8] = lab.fit_transform(y[:,8])
y[:,2] = lab.fit_transform(y[:,2])
y[:,1] = lab.fit_transform(y[:,1])
y[:,0] = lab.fit_transform(y[:,0])
print(y)


lab.classes_

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [2])
X = one.fit_transform(X)
X = X.toarray()
print(X)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)







































