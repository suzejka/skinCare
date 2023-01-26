import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
from helpers.model_to_file_helper import save_model, load_the_best_model, remove_temporary_files, save_accuracy_of_model_to_file, save_model_for_analysis, create_path_if_does_not_exist
import text_cleaner as cleaner
import telegram_bot_for_messages as bot
from sklearn.metrics import accuracy_score
import json
from sdv.tabular import GaussianCopula
import optuna
from helpers.data_preparation_helper import get_problem_column_index
warnings.filterwarnings("ignore")

ENCODERS = {}
ACCURACY = {}
PRODUCTS = {}
DATASET = None

LABELED_DATASET_GLOBAL = None

BEST_MODEL_DECISION_TREE = {}
BEST_MODEL_KNN = {}
BEST_MODEL_RANDOM_FOREST = {}

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
    Function creates synthetic data.
    '''       
    model = GaussianCopula()
    model.fit(dataset)
    synthetic_data = model.sample(5000)
    synthetic_data = synthetic_data[ALL_COLUMNS]
    return synthetic_data

def make_single_problem_tree(problemName, dumDf, modelName, **kwargs):
    '''
    Function creates a single problem tree.
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

def get_the_best_model_and_best_score_for_problem(problem):
    '''
    Funcion returns the best model and the best score for given problem.
    '''
    global BEST_MODEL_KNN, BEST_MODEL_DECISION_TREE, BEST_MODEL_RANDOM_FOREST
    best_score = max(BEST_MODEL_DECISION_TREE[problem][1], BEST_MODEL_KNN[problem][1], BEST_MODEL_RANDOM_FOREST[problem][1])
    if best_score == BEST_MODEL_DECISION_TREE[problem][1]:
        return BEST_MODEL_DECISION_TREE[problem][0], best_score
    elif best_score == BEST_MODEL_KNN[problem][1]:
        return BEST_MODEL_KNN[problem][0], best_score
    else:
        return BEST_MODEL_RANDOM_FOREST[problem][0], best_score

def tune_random_forest_optuna():
    '''
    Function finds the best parameters for RandomForrestClassifier
    '''
    global problem_global, BEST_MODEL_RANDOM_FOREST

    def objective(trial):
        global problem_global, LABELED_DATASET_GLOBAL, X_train, X_test, y_train, y_test
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 4, 50, step=2)
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)

        model = make_single_problem_tree(problem_global, LABELED_DATASET_GLOBAL, criterion=criterion, max_depth=max_depth, maxFeatures=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, nEstimators=n_estimators, modelName='RandomForestClassifier')
        model.fit(X_train, y_train)
        yPrediction = model.predict(X_test)

        save_model(trial, 'RandomForestClassifier', model, problem_global)

        return accuracy_score(y_test, yPrediction)

    current_model_accuracy = {}
    counter = 0
    for problem in DECISION_COLUMN_NAMES:
        problem_global = problem
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        best_model = load_the_best_model(study.best_trial, 'RandomForestClassifier', problem)
        BEST_MODEL_RANDOM_FOREST[problem] = [best_model, study.best_value]
        current_model_accuracy[problem] = study.best_value
        save_model_for_analysis(trial=study.best_trial, model=best_model, classifier='RandomForestClassifier', problemName=counter)
        remove_temporary_files()
        counter += 1
    
    save_accuracy_of_model_to_file('RandomForestClassifier', current_model_accuracy)

def tune_decision_tree_optuna():
    '''
    Function finds the best parameters for DecisionTreeClassifier.
    '''
    global problem_global, BEST_MODEL_DECISION_TREE
    def objective(trial):
        global problem_global, LABELED_DATASET_GLOBAL, X_train, X_test, y_train, y_test
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 4, 50, step=2)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        model = make_single_problem_tree(problem_global, LABELED_DATASET_GLOBAL, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, modelName='DecisionTreeClassifier')
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        save_model(trial, 'DecisionTreeClassifier', model, problem_global)

        return accuracy_score(y_test, prediction)

    current_model_accuracy = {}
    counter = 0
    for problem in DECISION_COLUMN_NAMES:
        problem_global = problem
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        best_model = load_the_best_model(study.best_trial, 'DecisionTreeClassifier', problem)
        BEST_MODEL_DECISION_TREE[problem] = [best_model, study.best_value] 
        current_model_accuracy[problem] = study.best_value
        save_model_for_analysis(trial=study.best_trial, model=best_model, classifier='DecisionTreeClassifier', problemName=counter)
        remove_temporary_files()
        counter += 1
    
    save_accuracy_of_model_to_file('DecisionTreeClassifier', current_model_accuracy)
        
def tune_knn_optuna():
    '''
    Function finds the best parameters for the KNeighborsClassifier model.
    '''
    global problem_global, BEST_MODEL_KNN

    def objective(trial):
        global problem_global, LABELED_DATASET_GLOBAL, X_train, X_test, y_train, y_test         
        n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
        weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
        metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
        
        model = make_single_problem_tree(problem_global, LABELED_DATASET_GLOBAL, n_neighbors=n_neighbors, weights=weights, metric=metric, modelName='KNeighborsClassifier')       
        model.fit(X_train, y_train)
        yPrediction = model.predict(X_test)

        save_model(trial, 'KNeighborsClassifier', model, problem_global)

        return accuracy_score(y_test, yPrediction)
    
    current_model_accuracy = {}
    counter = 0

    for problem in DECISION_COLUMN_NAMES:
        problem_global = problem
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        best_model = load_the_best_model(study.best_trial, 'KNeighborsClassifier', problem)
        BEST_MODEL_KNN[problem] = [best_model, study.best_value]
        current_model_accuracy[problem] = study.best_value
        save_model_for_analysis(trial=study.best_trial, model=best_model, classifier='KNeighborsClassifier', problemName=counter)
        remove_temporary_files()
        counter += 1

    save_accuracy_of_model_to_file('KNeighborsClassifier', current_model_accuracy)
        
def create_label_encoding(datasetToEncode):
    '''
    Function that encodes categorical columns to numerical values.
    '''
    global ENCODERS
    for categoricalColumn in ALL_CATEGORICAL_COLUMNS:
        ENCODERS[categoricalColumn] = LabelEncoder()
        uniqueValues = list(datasetToEncode[categoricalColumn].unique())
        ENCODERS[categoricalColumn] = ENCODERS[categoricalColumn].fit(uniqueValues)
        datasetToEncode[categoricalColumn] = ENCODERS[categoricalColumn].transform(datasetToEncode[categoricalColumn])
    return datasetToEncode

def save_highest_accuracy_for_the_best_models():
    '''
    Function that saves the highest accuracy for the best models.
    '''
    with open('files_for_documentation/accuracy_of_all_models/accuracy_of_DecisionTreeClassifier.json', "r") as file:
        dt_accuracy = json.load(file)
    with open("files_for_documentation/accuracy_of_all_models/accuracy_of_KNeighborsClassifier.json", "r") as file:
        knn_accuracy = json.load(file)
    with open("files_for_documentation/accuracy_of_all_models/accuracy_of_RandomForestClassifier.json", "r") as file:
        rf_accuracy = json.load(file)

    highest_accuracy = {
        category: max(
            dt_accuracy[category], knn_accuracy[category], rf_accuracy[category]
        )
        for category in dt_accuracy
    }
    with open("prepared_data/accuracy.json", "w") as file:
        json.dump(highest_accuracy, file)

def tune_models():
    '''
    Function that tunes models for each problem.
    '''
    tune_decision_tree_optuna()
    tune_knn_optuna()
    tune_random_forest_optuna()

def choose_best_model_for_problem(problem):
    global ACCURACY

    bestModel, bestScore = get_the_best_model_and_best_score_for_problem(problem)
    
    ACCURACY[problem] = bestScore
    filename = "models/{filename}.sv".format(filename = problem.replace(" ", "_"))
    pickle.dump(bestModel, open(filename,'wb'))

def build_models():
    '''
    Function that builds models for each problem.
    '''
    tune_models()

    for problem in DECISION_COLUMN_NAMES:
        choose_best_model_for_problem(problem)

def main():
    global PRODUCTS, DATASET, ACCURACY, LABELED_DATASET_GLOBAL

    PRODUCTS = pd.read_csv("raw_data/products.csv", sep=';')
    DATASET = pd.read_csv("raw_data/daneSkinCare.csv")

    PRODUCTS = PRODUCTS.to_dict()
    DATASET = cleaner.clean_data(DATASET)

    synthetic = create_synthetic_data(DATASET)

    # adding synthetic data to dataset
    DATASET = DATASET.append(synthetic, ignore_index=True)

    # checking if folder exists, if not, create it
    folder = "prepared_data"
    create_path_if_does_not_exist(folder)

    # export dataset to csv using utf-16 encoding to support polish characters
    DATASET.to_csv("prepared_data/DATASET.csv", index=False, encoding='utf-16', sep=',') 
   
    LABELED_DATASET_GLOBAL = create_label_encoding(DATASET)

    # export labeled dataset to csv
    LABELED_DATASET_GLOBAL.to_csv("prepared_data/labeledDataset.csv", index=False, sep=',')

    # building models and saving them to file
    build_models()
    
    # saving accuracy of models to file
    save_highest_accuracy_for_the_best_models()

if __name__ == '__main__':
    main()