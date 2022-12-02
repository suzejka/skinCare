import os
import pickle
import shutil
import json

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

def save_model(trial, classifier, model, problemName):
    '''
    Funkcja zapisuje najlepszy model.
    '''
    path = f"temp_models/{problemName}"
    create_path_if_does_not_exist(path)
    with open("{0}/{1}_{2}_{3}.hdf5".format(path, classifier, problemName, trial.number), "wb") as fout:
        pickle.dump(model, fout)

def save_model_for_analysis(trial, classifier, model, problemName):
    '''
    Funkcja zapisuje najlepszy model do analizy.
    '''
    path = f"files_for_documentation/models_for_analysis/{problemName}"
    create_path_if_does_not_exist(path)
    with open("{0}/{1}.hdf5".format(path, classifier), "wb") as fout:
        pickle.dump(model, fout)

def read_accuray_from_file():
    '''
    Odczytuje accuracy z pliku
    '''
    path = "prepared_data"
    create_path_if_does_not_exist(path)
    with open('{0}/accuracy.json'.format(path)) as json_file:
        ACCURACY = json.load(json_file)
    return ACCURACY

def save_accuracy_of_model_to_file(model_name, accuracy):
    '''
    Zapisuje accuracy do pliku
    '''
    path = "files_for_documentation/accuracy_of_all_models"
    create_path_if_does_not_exist(path)
    with open("{0}/accuracy_of_{1}.json".format(path, model_name), 'w') as outfile:
        json.dump(accuracy, outfile)

def create_path_if_does_not_exist(path):
    '''
    Ustawia ścieżkę do pliku
    '''
    if not does_path_exist(path):
        os.makedirs(path)