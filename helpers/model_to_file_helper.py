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

def save_the_best_model(trial, classifier, model, problemName):
    '''
    Funkcja zapisuje najlepszy model.
    '''
    path = f"temp_models/{problemName}"
    if not does_path_exist(path):
        os.makedirs(path)
    with open("{0}/{1}_{2}_{3}.hdf5".format(path, classifier, problemName, trial.number), "wb") as fout:
        pickle.dump(model, fout)

def read_accuray_from_file():
    '''
    Odczytuje accuracy z pliku
    '''
    with open('accuracy.json') as json_file:
        ACCURACY = json.load(json_file)
    return ACCURACY