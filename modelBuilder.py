import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings 
import requests
import textCleaner
from sklearn.metrics import accuracy_score
import json
from sdv.tabular import GaussianCopula
import optuna
from optuna.samplers import TPESampler
import os
import shutil
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

labeledDataset_global = None

best_model_decision_tree = {}
best_model_knn = {}
best_model_random_forest = {}

optuna_score_knn = {}
optuna_score_decision_tree = {}
optuna_score_random_forest = {}

optuna_best_params_knn = {}
optuna_best_params_decision_tree = {}
optuna_best_params_random_forest = {}

problem_global = None

X_train = None
X_test = None
y_train = None
y_test = None

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

def create_synthetic_data(dataset):
    '''
    Funkcja tworzy syntetyczne dane.
    '''       
    model = GaussianCopula()
    model.fit(dataset)
    synthetic_data = model.sample(5000)
    synthetic_data = synthetic_data[ALL_COLUMNS]
    synthetic_data.to_csv('synthetic_data.csv', index=False)
    return synthetic_data

def send_message_to_telegram(message):
    '''
    Funkcja wysyła wiadomość do twórcy.
    '''
    chatId = '5303880405'
    botToken = '5660046213:AAHCSDYbdW7E5rc5MnoL1n8QCY-Qh8M1ZgI'
    url = f"https://api.telegram.org/bot{botToken}/sendMessage?chat_id={chatId}&text={message}"
    requests.get(url)

def create_message(inputDict, message):
    '''
    Funkcja tworzy wiadomość do wysłania do twórcy.
    '''
    result = message + '\n'
    for i in inputDict.keys():
        result += f"{str(i)}: {str(inputDict[i])}" + '\n'
    return result

def get_problem_column_index(problemName):
    '''
    Funkcja zwraca indeks kolumny z danym problemem.
    '''
    if problemName == 'Mycie':
        return 5
    elif problemName == 'Serum na dzień':
        return 6
    elif problemName == 'Krem na dzień':
        return  7
    elif problemName == 'SPF' :
        return  8
    elif problemName == 'Serum na noc':
        return  9
    elif problemName == 'Krem na noc':
        return  10
    elif problemName == 'Punktowo':
        return  11
    elif problemName == 'Maseczka':
        return  12
    elif problemName == 'Peeling':
        return  13
    else :
        send_message_to_telegram("Błąd! Nie rozpoznano kategorii produktu.")
        raise ValueError("Nie rozpoznano kategorii produktu.")

def normalize_data(data):
    result = data.copy()
    for feature_name in data.columns:
        max_value = data[feature_name].max()
        min_value = data[feature_name].min()
        result[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)
    return result

def make_single_problem_tree(problemName, dumDf, modelName, **kwargs):
    '''
    Funkcja tworzy drzewo decyzyjne dla pojedynczego problemu.
    '''
    global X_train, X_test, y_train, y_test
    problemIndex = get_problem_column_index(problemName)
    X = dumDf.values[:, 0:5]
    yProblem = dumDf.values[:, problemIndex]
    X_train, X_test, y_train, y_test = train_test_split(X, yProblem, test_size = 0.25)

    if modelName == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(max_depth = kwargs['max_depth'], 
        criterion = kwargs['criterion'], 
        min_samples_leaf = kwargs['min_samples_leaf'], 
        min_samples_split = kwargs['min_samples_split']
        )
    elif modelName == 'KNeighborsClassifier':
        model = KNeighborsClassifier(n_neighbors = kwargs['n_neighbors'], 
        weights = kwargs['weights'],
        metric = kwargs['metric'],
        )
    elif modelName == 'RandomForestClassifier':
        model = RandomForestClassifier(max_depth = kwargs['max_depth'], 
        criterion = kwargs['criterion'], 
        max_features = kwargs['maxFeatures'], 
        min_samples_leaf = kwargs['min_samples_leaf'], 
        min_samples_split = kwargs['min_samples_split'], 
        n_estimators = kwargs['nEstimators']
        )

    model = model.fit(X_train, y_train)
 
    return model

def remove_temporary_files():
    '''
    Funkcja usuwa tymczasowe pliki.
    '''
    shutil.rmtree("temp_models")

def does_path_exist(path):
    '''
    Funkcja sprawdza czy ścieżka istnieje.
    '''
    return bool(os.path.exists(path))

def load_the_best_model(trial, classifierName, problemName):
    '''
    Funkcja ładuje najlepszy model.
    '''
    path = f"temp_models/{problemName}"
    with open("{0}/{1}_{2}_{3}.hdf5".format(path, classifierName, problemName, trial.number), "rb") as fin:
        best_model = pickle.load(fin)
    return best_model

def save_the_best_model(trial, classifier, model, problemName):
    '''
    Funkcja zapisuje najlepszy model.
    '''
    path = f"temp_models/{problemName}"
    if not does_path_exist(path):
        os.makedirs(path)
    with open("{0}/{1}_{2}_{3}.hdf5".format(path, classifier, problemName, trial.number), "wb") as fout:
        pickle.dump(model, fout)

def get_the_best_model_and_best_score_for_problem(problem):
    '''
    Funkcja zwraca najlepszy model i najlepszy wynik dla danego problemu.
    '''
    global best_model_knn, best_model_decision_tree, best_model_random_forest
    best_score = max(best_model_decision_tree[problem][1], best_model_knn[problem][1], best_model_random_forest[problem][1])
    if best_score == best_model_decision_tree[problem][1]:
        return best_model_decision_tree[problem][0], best_score
    elif best_score == best_model_knn[problem][1]:
        return best_model_knn[problem][0], best_score
    else:
        return best_model_random_forest[problem][0], best_score

def tune_random_forest_optuna():
    '''
    Funkcja znajduje najlepsze parametry dla modelu RandomForestClassifier.
    '''
    global problem_global, best_model_random_forest

    def objective(trial):
        global problem_global, labeledDataset_global, X_train, X_test, y_train, y_test
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 4, 50, step=2)
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)

        model = make_single_problem_tree(problem_global, labeledDataset_global, criterion=criterion, max_depth=max_depth, maxFeatures=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, nEstimators=n_estimators, modelName='RandomForestClassifier')
        model.fit(X_train, y_train)
        yPrediction = model.predict(X_test)

        save_the_best_model(trial, 'RandomForestClassifier', model, problem_global)

        return accuracy_score(y_test, yPrediction)

    for problem in DECISION_COLUMN_NAMES:
        problem_global = problem
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=100)
        optuna_score_random_forest[problem] = study.best_value
        optuna_best_params_random_forest[problem] = study.best_params
        best_model = load_the_best_model(study.best_trial, 'RandomForestClassifier', problem)
        best_model_random_forest[problem] = [best_model, study.best_value]
        remove_temporary_files()
            
def tune_decision_tree_optuna():
    '''
    Funkcja dokonuje optymalizacji parametrów drzewa decyzyjnego.
    '''
    global problem_global, best_model_decision_tree
    def objective(trial):
        global problem_global, labeledDataset_global, X_train, X_test, y_train, y_test
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 4, 50, step=2)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        model = make_single_problem_tree(problem_global, labeledDataset_global, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, modelName='DecisionTreeClassifier')
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        save_the_best_model(trial, 'DecisionTreeClassifier', model, problem_global)

        return accuracy_score(y_test, prediction)
        
    for problem in DECISION_COLUMN_NAMES:
        problem_global = problem
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        optuna_score_decision_tree[problem] = study.best_value
        optuna_best_params_decision_tree[problem] = study.best_params
        best_model = load_the_best_model(study.best_trial, 'DecisionTreeClassifier', problem)
        best_model_decision_tree[problem] = [best_model, study.best_value]
        remove_temporary_files()
        
def tune_knn_optuna():
    '''
    Funkcja dokonuje optymalizacji hiperparametrów dla klasyfikatora KNN.
    '''
    global problem_global, best_model_knn

    def objective(trial):
        global problem_global, labeledDataset_global, X_train, X_test, y_train, y_test         
        n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
        weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
        metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
        
        model = make_single_problem_tree(problem_global, labeledDataset_global, n_neighbors=n_neighbors, weights=weights, metric=metric, modelName='KNeighborsClassifier')       
        model.fit(X_train, y_train)
        yPrediction = model.predict(X_test)

        save_the_best_model(trial, 'KNeighborsClassifier', model, problem_global)

        return accuracy_score(y_test, yPrediction)
    
    for problem in DECISION_COLUMN_NAMES:
        problem_global = problem
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=100)
        optuna_score_knn[problem] = study.best_value
        optuna_best_params_knn[problem] = study.best_params
        best_model = load_the_best_model(study.best_trial, 'KNeighborsClassifier', problem)
        best_model_knn[problem] = [best_model, study.best_value]
        remove_temporary_files()
        
def create_label_encoding(datasetToEncode):
    '''
    Funkcja tworzy kodowanie etykiet dla kolumn kategorycznych.
    '''
    global ENCODERS, LABELS_DESCRIPTION
    for categoricalColumn in ALL_CATEGORICAL_COLUMNS:
        ENCODERS[categoricalColumn] = LabelEncoder() # add categories
        uniqueValues = list(datasetToEncode[categoricalColumn].unique())
        ENCODERS[categoricalColumn] = ENCODERS[categoricalColumn].fit(uniqueValues)
        datasetToEncode[categoricalColumn] = ENCODERS[categoricalColumn].transform(datasetToEncode[categoricalColumn])
    return datasetToEncode

def tune_models():
    '''
    Funkcja wywołuje funkcje do optymalizacji parametrów modeli.
    '''
    tune_decision_tree_optuna()
    tune_knn_optuna()
    tune_random_forest_optuna()

def choose_best_model_for_problem(problem):
    global optuna_best_params_decision_tree, optuna_best_params_knn, optuna_best_params_random_forest, ACCURACY

    bestModel, bestScore = get_the_best_model_and_best_score_for_problem(problem)
    
    ACCURACY[problem] = bestScore
    filename = "{filename}.sv".format(filename = problem.replace(" ", "_"))
    pickle.dump(bestModel, open(filename,'wb'))

def build_models():
    '''
    Funkcja buduje modele drzewa decyzyjnego dla wszystkich problemów.
    '''
    tune_models()

    for problem in DECISION_COLUMN_NAMES:
        choose_best_model_for_problem(problem)

def main():
    global PRODUCTS, DATASET, ACCURACY, labeledDataset_global

    PRODUCTS = pd.read_csv("products.csv", sep=';') # pobranie produktów z pliku    
    DATASET = pd.read_csv("daneSkinCare.csv") # pobranie danych z pliku
    VALIDATION_DATASET = pd.read_csv("validationDataset.csv") # pobranie danych walidacyjnych z pliku

    PRODUCTS = PRODUCTS.to_dict()
    DATASET = textCleaner.clear_data(DATASET) # czyszczenie danych

    synthetic = create_synthetic_data(DATASET) # tworzenie danych syntetycznych

    DATASET = DATASET.append(synthetic, ignore_index=True) # dodanie danych syntetycznych do zbioru danych

    DATASET.to_csv("DATASET.csv", index=False, encoding='utf-16', sep=',') # UTF-16 to encoding, który obsługuje polskie znaki

    labeledValidationDataset = create_label_encoding(VALIDATION_DATASET) # zakodowanie danych walidacyjnych    
    labeledDataset_global = create_label_encoding(DATASET) # tworzenie kodowania etykiet
    labeledDataset_global.to_csv("labeledDataset.csv", index=False, sep=',')
    # labeledDataset_global = normalize_data(labeledDataset_global)
    build_models() # budowanie modeli
    
    with open('accuracy.json', 'w') as fp:
        json.dump(ACCURACY, fp)

if __name__ == '__main__':
    main()