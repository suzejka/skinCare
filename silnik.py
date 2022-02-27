import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv("daneSkinCare.csv", sep=';')
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
dataset.info()

#label_encoder = preprocessing.LabelEncoder()
#label_encoder = OneHotEncoder(handle_unknown='ignore')

typ_cery = dataset['Typ cery']
glowny_problem = dataset['Główny problem']
poboczny_problem = dataset['Poboczny problem']
mycie = dataset['Mycie']
serum_dzien = dataset['Serum na dzień']
krem_dzien = dataset['Krem na dzień']
spf = dataset['SPF']
serum_noc = dataset['Serum na noc']
krem_noc = dataset['Krem na noc']
punktowo = dataset['Punktowo']
maseczka = dataset['Maseczka']
peeling = dataset['Peeling']

dum_cera = pd.get_dummies(dataset['Typ cery'])
dum_glowny_problem = pd.get_dummies(dataset['Główny problem'])
dum_poboczny_problem = pd.get_dummies(dataset['Poboczny problem'])
dum_mycie = pd.get_dummies(dataset['Mycie'])
dum_serum_dzien = pd.get_dummies(dataset['Serum na dzień'])
dum_krem_dzien = pd.get_dummies(dataset['Krem na dzień'])
dum_spf = pd.get_dummies(dataset['SPF'])
dum_serum_noc = pd.get_dummies(dataset['Serum na noc'])
dum_krem_noc = pd.get_dummies(dataset['Krem na noc'])
dum_punktowo = pd.get_dummies(dataset['Punktowo'])
dum_maseczka = pd.get_dummies(dataset['Maseczka'])
dum_peeling = pd.get_dummies(dataset['Peeling'])


# mergedDataset = dataset.merge(dum_cera, how = 'left', on='Typ cery')
# mergedDataset = dataset.merge(dum_glowny_problem, how = 'left', on='Główny problem')
# mergedDataset = dataset.merge(dum_poboczny_problem, how = 'left', on='Poboczny problem')
# mergedDataset = dataset.merge(dum_mycie, dum_serum_dzien, how = 'left', on='Mycie')
# mergedDataset = dataset.merge(dum_serum_dzien, how = 'left', on='Serum na dzień')
# mergedDataset = dataset.merge(dum_krem_dzien, how = 'left', on='Krem na dzień')
# mergedDataset = dataset.merge(dum_spf, how = 'left', on='SPF')
# mergedDataset = dataset.merge(dum_serum_noc, how = 'left', on='Serum na noc')
# mergedDataset = dataset.merge(dum_krem_noc, how = 'left', on='Krem na noc')
# mergedDataset = dataset.merge(dum_punktowo, how = 'left', on='Punktowo')
# mergedDataset = dataset.merge(dum_maseczka, how = 'left', on='Maseczka')
# mergedDataset = dataset.merge(dum_peeling, how = 'left', on='Peeling')

print(dataset.head())
'''
X = dum_df.values[:, 1:5]
Y_mycie = dataset.values[:, 6]

X_train, X_test, y_train_mycie, y_test_mycie = train_test_split(X, Y_mycie, test_size = 0.25, random_state = 100)

cols_names = ['Typ cery', 'Główny problem', 'Poboczny problem', 'Wrażliwa','Wiek']

model = DecisionTreeClassifier(max_depth=10)

model = model.fit(X_train, y_train_mycie)

y_pred_gini = model.predict(X_test)

print ("Accuracy : ", accuracy_score(y_test_mycie, y_pred_gini))

#c_names = [dataset['Typ cery'], dataset['Główny problem'], dataset['Poboczny problem'], dataset['Mycie']]


fig = plt.figure(figsize=(10,10))
_ = tree.plot_tree(model, filled=True, rounded=True, feature_names=cols_names)
fig.savefig("skinCare_tree.pdf")
plt.show()

print(X_test)'''