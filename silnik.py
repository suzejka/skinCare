from msilib.schema import Error
from operator import concat
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing, tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn
from imblearn.over_sampling import SMOTE
import numpy
from collections import Counter

graph_counter = 1

def makeSingleProblemTree(problem_name, dum_df, dataset):
    problem_index = 0
    global graph_counter
    match problem_name:
        case 'Mycie':
            problem_index = 6
        case 'Serum na dzień':
            problem_index = 7
        case 'Krem na dzień':
            problem_index = 8
        case 'SPF' :
            problem_index = 9
        case 'Serum na noc':
            problem_index = 10
        case 'Krem na noc':
            problem_index = 11
        case 'Punktowo':
            problem_index = 12
        case 'Maseczka':
            problem_index = 13
        case 'Peeling':
            problem_index = 14
        case _:        
            print('Error')
    X = dum_df.values[:, 1:6]
    Y_problem = dataset.values[:, problem_index]
    X_train, X_test, y_train, y_test = train_test_split(X, Y_problem, test_size = 0.25, random_state = 100)
    model = DecisionTreeClassifier(max_depth=10)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print ("{} - Accuracy : {}".format(problem_name,accuracy_score(y_test, y_pred)*100))
    unique_values = dataset[problem_name].unique()
    fig = plt.subplots(figsize=(5,5))
    tree.plot_tree(model, 
                    filled=True, 
                    rounded=True, 
                    feature_names=cols_names, 
                    class_names=unique_values)
    # fig.savefig("skinCare_tree.pdf")
    matrix = confusion_matrix(y_test, y_pred)
    # sn.heatmap(matrix, annot=True)
    
    graph_counter = graph_counter + 1
    return model

def createDummies(dataset_to_encode):
    dum_cera = pd.get_dummies(dataset_to_encode['Typ cery'])
    dum_glowny_problem = pd.get_dummies(dataset_to_encode['Główny problem'])
    dum_poboczny_problem = pd.get_dummies(dataset_to_encode['Poboczny problem'])
    dum_wrazliwa = pd.get_dummies(dataset_to_encode['Wrażliwa'])
    dum_wiek = pd.get_dummies(dataset_to_encode['Wiek'])
    dum_mycie = pd.get_dummies(dataset_to_encode['Mycie'])
    dum_serum_dzien = pd.get_dummies(dataset_to_encode['Serum na dzień'])
    dum_krem_dzien = pd.get_dummies(dataset_to_encode['Krem na dzień'])
    dum_spf = pd.get_dummies(dataset_to_encode['SPF'])
    dum_serum_noc = pd.get_dummies(dataset_to_encode['Serum na noc'])
    dum_krem_noc = pd.get_dummies(dataset_to_encode['Krem na noc'])
    dum_punktowo = pd.get_dummies(dataset_to_encode['Punktowo'])
    dum_maseczka = pd.get_dummies(dataset_to_encode['Maseczka'])
    dum_peeling = pd.get_dummies(dataset_to_encode['Peeling'])

    frames = [
                dum_cera,
                dum_glowny_problem, 
                dum_poboczny_problem,
                dum_wrazliwa,
                dum_wiek,
                dum_mycie, 
                dum_serum_dzien, 
                dum_krem_dzien, 
                dum_spf, 
                dum_serum_noc, 
                dum_krem_noc, 
                dum_punktowo,
                dum_maseczka,
                dum_peeling
                ]

    dum_df = pd.concat(frames, axis=1) 
    return dum_df

def handleImbalancedData(dataset):
    smote = SMOTE()
    fig, axs = plt.subplots(3, 3)
    plt.rcParams["figure.figsize"] = (10,6)
    fig.tight_layout(pad=1.5)
    tmp = 0
    axs_y = 0

    for i in cols_names:
        dataset[i] = preprocessing.LabelEncoder().fit_transform(numpy.asarray(dataset[i]))
        
    for i in range(6,15):
        X,y = dataset.values[:, 1:6], dataset.values[:, i]
        y = preprocessing.LabelEncoder().fit_transform(numpy.asarray(y))
        
        X, y = smote.fit_resample(X, y)
        counter = Counter(y)
        
        for k,v in counter.items(): # k = klasa, v = ile elementow danej klasy, per = procenty
            per = v / len(y) * 100
    #         print('Class=%d, n=%d (%.2f%%)' % (k, v, per))

        axs_x = tmp%3
        
        if tmp == 3 or tmp == 6:
            axs_y += 1

        axs[axs_x, axs_y].set_title('{}'.format(dataset.columns[i]))
        axs[axs_x, axs_y].bar(counter.keys(), counter.values())
        tmp = tmp+1
    
    plt.show()

def predictMyObject(model, my_object, dataset):
    dataset = dataset.append(my_object, ignore_index=True)
    dum_df = createDummies(dataset)
    predition = model.predict(my_object)
    return predition

def main():
    dataset = pd.read_csv("daneSkinCare.csv", sep=';')

    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

    dum_df = createDummies(dataset)
    print(dum_df)
    dum_df.to_csv('dum.dataframe.csv')

    handleImbalancedData(dataset)
    
    for i in decision_column_names:
        problemModel = makeSingleProblemTree(i, dum_df, dataset)
        
        # result = predictMyObject(problemModel, my_dataframe, dataset)
        # print(result)
    # plt.show()

my_dataframe = {'Typ cery': ['Tłusta'],
                'Wrażliwa': [0],
                'Główny problem': ['Nadprodukcja sebum'],
                'Poboczny problem': ['Niedoskonałości'],
                'Wiek': [21]}

cols_names = ['Typ cery', 'Główny problem', 'Poboczny problem', 'Wrażliwa','Wiek']
decision_column_names = ['Mycie','Serum na dzień','Krem na dzień','SPF','Serum na noc','Krem na noc','Punktowo','Maseczka','Peeling']

smote = SMOTE()

if __name__ == '__main__':
    main()