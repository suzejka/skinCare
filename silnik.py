from msilib.schema import Error
from operator import concat
from pickle import FALSE
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing, tree
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import seaborn as sn
import numpy as np
from collections import Counter
import time
import streamlit as st

graph_counter = 1

encoder = None
encoders = {}
labelsDescription = {}
typ_cery = None
czy_wrazliwa = None
glowny_problem = None
poboczny_problem = None
wiek = None
resultSkinCare = {}
models = []
accuracy = {}
products = {}

cols_names = ['Typ cery', 'Główny problem', 'Poboczny problem', 'Wrażliwa','Wiek']
categorical_cols_names = ['Typ cery', 'Główny problem', 'Poboczny problem']
decision_column_names = ['Mycie','Serum na dzień','Krem na dzień','SPF','Serum na noc','Krem na noc','Punktowo','Maseczka','Peeling']
all_columns = cols_names + decision_column_names
all_categorical_columns = categorical_cols_names + decision_column_names

def clearText(text):
    text = str(text).replace("'","").replace("[","").replace("]","")
    return text

def makeSingleProblemTree(problem_name, dum_df, dataset):
    problem_index = 0
    global graph_counter, accuracy
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
    accuracy[problem_name] = str(accuracy_score(y_test, y_pred)*100) # "{} - Accuracy : {}".format(problem_name,accuracy_score(y_test, y_pred)*100)
    
    unique_values = dataset[problem_name].unique()
    
    graph_counter = graph_counter + 1
    return model

def predictMyObject(model, my_object, column_name):
    global encoders
    
    for i in categorical_cols_names:
        labelsDescription = encoders[i].classes_
        labelsDescription = labelsDescription.tolist()
        for l in labelsDescription:
            if my_object[i].item() == l:
                print("Klasa - ", my_object[i].item(), "Label -> ", labelsDescription.index(l))
                my_object[i] = labelsDescription.index(l)
        # my_object[i] = np.where(labelsDescription == my_object[i])
    prediction = model.predict(my_object)
    prediction = encoders[column_name].inverse_transform(prediction)
    return prediction

def setPhoto(category, side):
    global products
    if side == 'left':
        link = str(resultSkinCare.get(category))
        value = clearText(str(products.get(clearText(link)))).replace("{","").replace("0: ","").replace("}","")
        if value != "0": 
            col1, col2, = st.columns([1,3])
            with col1:
                if value != "0":
                    st.image(value, width=150)
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown(clearText(resultSkinCare.get(category)))
        else:
            st.markdown(clearText(resultSkinCare.get(category)))
    else:
        link = str(resultSkinCare.get(category))
        value = clearText(str(products.get(clearText(link)))).replace("{","").replace("0: ","").replace("}","")
        if value != "0": 
            col1, col2, = st.columns([3,1])
            with col1:
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown(clearText(resultSkinCare.get(category)))
            with col2:
                st.image(value, width=150)
        else:
            st.markdown(clearText(resultSkinCare.get(category)))

def showGUI(dum_df, dataset, products):
    global typ_cery, czy_wrazliwa, glowny_problem, poboczny_problem, wiek, accuracy

    st.set_page_config(
     page_title="System rekomendacyjny, do tworzenia planów pielęgnacyjnych",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# Praca inżynierska. *s20943*"
        }
    )

    form = st.form("my_form")
    form.subheader('Jaki masz typ cery?')
    typ_cery = form.radio(
     "",
     ('Tłusta', 
     'Normalna', 
     'Mieszana',
     'Sucha'))

    form.subheader('Czy Twoja cera jest wrażliwa?')
    czy_wrazliwa = form.radio(
     "",
     ('Tak', 
     'Nie'))

    if czy_wrazliwa == 'Tak':
        czy_wrazliwa = 1
    elif czy_wrazliwa == 'Nie':
        czy_wrazliwa = 0

    form.subheader('Jaki jest Twój największy problem z cerą?')        
    glowny_problem = form.radio(
     "",
     ('Nadprodukcja sebum', 
     'Niedoskonałości', 
     'Podrażnienie',
     'Przebarwienia',
     'Rozszerzone pory',
     'Suche skórki',
     'Szara cera',
     'Widoczne naczynka'))

    form.subheader('Czy jeszcze z czymś się zmagasz?')        
    poboczny_problem = form.radio(
     "",
     ('Nie mam więcej problemów z cerą',
     'Nadprodukcja sebum', 
     'Niedoskonałości', 
     'Podrażnienie',
     'Przebarwienia',
     'Rozszerzone pory',
     'Suche skórki',
     'Szara cera',
     'Widoczne naczynka'))

    if poboczny_problem == 'Nie mam więcej problemów z cerą':
        poboczny_problem = 'Brak'
    
    form.subheader('Ile masz lat?')
    wiek = form.slider("", 16, 60)
        
    clicked = form.form_submit_button("Wyślij")
    if clicked:
        
        my_dataframe = {'Typ cery': typ_cery,
                    'Główny problem': glowny_problem,
                    'Poboczny problem': poboczny_problem,
                    'Wrażliwa': czy_wrazliwa,
                    'Wiek': wiek}
        df = pd.DataFrame.from_dict([my_dataframe])
        for i in decision_column_names:
            problemModel = makeSingleProblemTree(i, dum_df, dataset)
            result = predictMyObject(problemModel, df, i)
            #print(i, " - " ,result)
            resultSkinCare[i] = result

        st.session_state.accuracy = df
        with st.spinner('Tworzę Twój plan pielęgnacyjny...'):
            time.sleep(4)
        st.success('Skończone!')
   
        st.header('Proponowana pielęgnacja')
        st.subheader('Mycie')
        setPhoto('Mycie', 'left')
        st.subheader('Serum na dzień')
        setPhoto('Serum na dzień', 'right')
        st.subheader('Krem na dzień')
        setPhoto('Krem na dzień', 'left')
        st.subheader('Krem przeciwsłoneczny')
        setPhoto('SPF', 'right')
        st.subheader('Serum na noc')
        setPhoto('Serum na noc', 'left')
        st.subheader('Krem na noc')
        setPhoto('Krem na noc', 'right')
        st.subheader('Punktowo')
        setPhoto('Punktowo', 'left')
        st.subheader('Maseczka')
        setPhoto('Maseczka', 'right')
        st.subheader('Peeling')
        setPhoto('Peeling', 'left')

        devClicked = st.button("Strefa dewelopera")
        #if devClicked:
            #open("dev_page.py")
        df = pd.DataFrame({"Kategoria": accuracy.keys(), "Dokładność": accuracy.values()})
         
        st.dataframe(data=df)

        st.stop()

def createLabelEncoding(dataset_to_encode):
    global encoders, labelsDescription
    
    for i in all_categorical_columns:
        encoders[i] = LabelEncoder()
        unique_values = list(dataset_to_encode[i].unique())
        encoders[i] = encoders[i].fit(unique_values)
        dataset_to_encode[i] = encoders[i].transform(dataset_to_encode[i])
        
    return dataset_to_encode

def main():
    global products
    products = pd.read_csv("products.csv", sep=';')
    products = products.to_dict()
    dataset = pd.read_csv("daneSkinCare.csv", sep=';')
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
    dum_df = createLabelEncoding(dataset)
    showGUI(dum_df, dataset, products)

if __name__ == '__main__':
    main()