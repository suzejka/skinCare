import pickle
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings 
import requests
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import textCleaner
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

ENCODER = None
ENCODERS = {}
LABELS_DESCRIPTION = {}
SKIN_TYPE = None
IS_SENSITIVE = None
MAIN_PROBLEM = None
SECOND_PROBLEM = None
AGE = None
RESULT_SKIN_CARE = {}
ACCURACY = {}
PRODUCTS = {}
LINK = None
CHOSEN_PRODUCT = None
DATASET = None
VALIDATION_DATASET = None

ASKED_COLUMN_NAMES = ['Typ cery', 'Główny problem', 'Poboczny problem', 'Wrażliwa','Wiek']
CATEGORICAL_COLUMN_NAMES = ['Typ cery', 'Główny problem', 'Poboczny problem']
DECISION_COLUMN_NAMES = ['Mycie',
'Serum na dzień',
'Krem na dzień',
'SPF',
'Serum na noc',
'Krem na noc',
'Punktowo',
'Maseczka',
'Peeling'
]
ALL_COLUMNS = ASKED_COLUMN_NAMES + DECISION_COLUMN_NAMES
ALL_CATEGORICAL_COLUMNS = CATEGORICAL_COLUMN_NAMES + DECISION_COLUMN_NAMES

def send_message_to_telegram(message):
    chatId = '5303880405'
    botToken = '5660046213:AAHCSDYbdW7E5rc5MnoL1n8QCY-Qh8M1ZgI'
    url = f"https://api.telegram.org/bot{botToken}/sendMessage?chat_id={chatId}&text={message}"
    requests.get(url)

def create_message(inputDict, message):
    result = message + '\n'
    for i in inputDict.keys():
        result += f"{str(i)}: {str(inputDict[i])}" + '\n'
    return result

def clear_text(text):
    text = str(text).replace("'","").replace("[","").replace("]","").replace("\\xa0", " ")
    return text

def getProblemColumnIndex(problemName):
    match problemName:
        case 'Mycie':
            return 6
        case 'Serum na dzień':
            return 7
        case 'Krem na dzień':
            return  8
        case 'SPF' :
            return  9
        case 'Serum na noc':
            return  10
        case 'Krem na noc':
            return  11
        case 'Punktowo':
            return  12
        case 'Maseczka':
            return  13
        case 'Peeling':
            return  14
        case _:
            send_message_to_telegram("Błąd! Nie rozpoznano kategorii produktu.")
            raise ValueError("Nie rozpoznano kategorii produktu.")

def make_single_problem_tree(problemName, dumDf, dataset, maxDepth = None, criterion = None):
    global ACCURACY
    problemIndex = getProblemColumnIndex(problemName)
    X = dumDf.values[:, 1:6]
    yProblem = dumDf.values[:, problemIndex]

    X_train, X_test, y_train, y_test = train_test_split(X, yProblem, test_size = 0.25, random_state = 100)
    model = DecisionTreeClassifier(max_depth=maxDepth, criterion=criterion)
    model = model.fit(X_train, y_train)
    yPrediction = model.predict(X_test)
    ACCURACY[problemName] = f"{str(accuracy_score(y_test, yPrediction) * 100)}%"
    return model

def tuneTreeModel(labeledData):
    global DATASET
    models = {}
    for singleDecisionColumn in DECISION_COLUMN_NAMES:
        X = labeledData.values[:, 1:6]
        problemIndex = getProblemColumnIndex(singleDecisionColumn)
        yProblem = labeledData.values[:, problemIndex]    

        decisionTree = DecisionTreeClassifier()

        scaler = StandardScaler()
        pca = decomposition.PCA()

        pipe = Pipeline(steps=[('std_slc', scaler),
                           ('pca', pca),
                           ('dec_tree', decisionTree)])

        n_components = list(range(1, X.shape[1]+1))

        criterion = ['gini', 'entropy']
        max_depth = [4,6,8,10,12,15,20]


        parameters = dict(pca__n_components=n_components,
                      dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth) # https://www.projectpro.io/recipes/optimize-hyper-parameters-of-decisiontree-model-using-grid-search-in-python#mcetoc_1g1ajorna9

        gridSearch = GridSearchCV(pipe, parameters)#, n_jobs = -1, verbose = 1)
        gridSearch.fit(X, yProblem)

        bestCriterion = gridSearch.best_estimator_.get_params()['dec_tree__criterion']
        bestDepth = gridSearch.best_estimator_.get_params()['dec_tree__max_depth']

        problemModel = make_single_problem_tree(singleDecisionColumn, labeledData, DATASET, maxDepth=bestDepth, criterion=bestCriterion)
        filename = "{filename}.sv".format(filename = singleDecisionColumn.replace(" ", "_"))
        pickle.dump(problemModel, open(filename,'wb'))
        models[singleDecisionColumn] = problemModel
    return models
        

def create_label_encoding(datasetToEncode):
    global ENCODERS, LABELS_DESCRIPTION
    for categoricalColumn in ALL_CATEGORICAL_COLUMNS:
        ENCODERS[categoricalColumn] = LabelEncoder()
        uniqueValues = list(datasetToEncode[categoricalColumn].unique())
        ENCODERS[categoricalColumn] = ENCODERS[categoricalColumn].fit(uniqueValues)
        datasetToEncode[categoricalColumn] = ENCODERS[categoricalColumn].transform(datasetToEncode[categoricalColumn])
    return datasetToEncode
     

def buildModels(convertedDataset, convertedValidationDataset):
    problemModels = tuneTreeModel(convertedDataset)
    print(convertedValidationDataset)
    for modelName in problemModels:
        X = convertedValidationDataset.values[:, 1:6]
        problemIndex = getProblemColumnIndex(modelName)
        yProblem = convertedValidationDataset.values[:, problemIndex]    
        #predict validation set and print accuracy
        validationPrediction = problemModels[modelName].predict(X) 
        #print model accuracy
        print(f"Accuracy of {modelName} model: {accuracy_score(yProblem, validationPrediction) * 100}%")
            #confusing matrix
      
def main():
    global PRODUCTS, DATASET, ACCURACY
    PRODUCTS = pd.read_csv("products.csv", sep=';') # pobranie produktów z pliku    
    DATASET = pd.read_csv("daneSkinCare.csv") # pobranie danych z pliku
    VALIDATION_DATASET = pd.read_csv("validationDataset.csv") # pobranie danych walidacyjnych z pliku

    PRODUCTS = PRODUCTS.to_dict()

    DATASET = DATASET.loc[:, ~DATASET.columns.str.contains('^Unnamed')] # usuwanie niepotrzebnych kolumn
    DATASET = textCleaner.clear_data(DATASET) # czyszczenie danych
    DATASET.to_csv("DATASET.csv", index=False, encoding='utf-16', sep=',') # UTF-16 to encoding, który obsługuje polskie znaki

    labeledValidationDataset = create_label_encoding(VALIDATION_DATASET) # zakodowanie danych walidacyjnych    
    labeledDataset = create_label_encoding(DATASET) # tworzenie kodowania etykiet

    buildModels(labeledDataset, labeledValidationDataset) # budowanie modeli
    print(ACCURACY)

if __name__ == '__main__':
    main()