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

graphCounter = 1

encoder = None
encoders = {}
labelsDescription = {}
skinType = None
isSensitive = None
mainProblem = None
secondProblem = None
age = None
resultSkinCare = {}
models = []
accuracy = {}
products = {}

askedColumnNames = ['Typ cery', 'Główny problem', 'Poboczny problem', 'Wrażliwa','Wiek']
categoricalColumnNames = ['Typ cery', 'Główny problem', 'Poboczny problem']
decisionColumnNames = ['Mycie','Serum na dzień','Krem na dzień','SPF','Serum na noc','Krem na noc','Punktowo','Maseczka','Peeling']
allColumns = askedColumnNames + decisionColumnNames
allCategoricalColumns = categoricalColumnNames + decisionColumnNames

def clearText(text):
    text = str(text).replace("'","").replace("[","").replace("]","")
    return text

def makeSingleProblemTree(problemName, dum_df, dataset):
    problemIndex = 0
    global graphCounter, accuracy
    # match problemName:
    #     case 'Mycie':
    #         problemIndex = 6
    #     case 'Serum na dzień':
    #         problemIndex = 7
    #     case 'Krem na dzień':
    #         problemIndex = 8
    #     case 'SPF' :
    #         problemIndex = 9
    #     case 'Serum na noc':
    #         problemIndex = 10
    #     case 'Krem na noc':
    #         problemIndex = 11
    #     case 'Punktowo':
    #         problemIndex = 12
    #     case 'Maseczka':
    #         problemIndex = 13
    #     case 'Peeling':
    #         problemIndex = 14
    #     case _:        
    #         print('Error')

    if(problemName == 'Mycie'):
        problemIndex = 6
    if(problemName == 'Serum na dzień'):
        problemIndex = 7
    if(problemName == 'Krem na dzień'):
        problemIndex = 8
    if(problemName == 'SPF'):
        problemIndex = 9
    if(problemName ==  'Serum na noc'):
        problemIndex = 10
    if(problemName ==  'Krem na noc'):
        problemIndex = 11
    if(problemName ==  'Punktowo'):
        problemIndex = 12
    if(problemName ==  'Maseczka'):
        problemIndex = 13
    if(problemName ==  'Peeling'):
        problemIndex = 14

    X = dum_df.values[:, 1:6]
    Y_problem = dataset.values[:, problemIndex]
    X_train, X_test, y_train, y_test = train_test_split(X, Y_problem, test_size = 0.25, random_state = 100)
    model = DecisionTreeClassifier(max_depth=10)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy[problemName] = str(accuracy_score(y_test, y_pred)*100) # "{} - Accuracy : {}".format(problem_name,accuracy_score(y_test, y_pred)*100)
    
    unique_values = dataset[problemName].unique()
    
    graphCounter = graphCounter + 1
    return model

def predictMyObject(model, myObject, columnName):
    global encoders
    
    for i in categoricalColumnNames:
        labelsDescription = encoders[i].classes_
        labelsDescription = labelsDescription.tolist()
        for l in labelsDescription:
            if myObject[i].item() == l:
                print("Klasa - ", myObject[i].item(), "Label -> ", labelsDescription.index(l))
                myObject[i] = labelsDescription.index(l)
        # my_object[i] = np.where(labelsDescription == my_object[i])
    prediction = model.predict(myObject)
    prediction = encoders[columnName].inverse_transform(prediction)
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

def showGUI(dumDf, dataset, products):
    global skinType, isSensitive, mainProblem, secondProblem, age, accuracy

    st.set_page_config(
     page_title="System rekomendacyjny, do tworzenia planów pielęgnacyjnych")

    form = st.form("my_form")
    form.subheader('Jaki masz typ cery?')
    skinType = form.radio(
     "",
     ('Tłusta', 
     'Normalna', 
     'Mieszana',
     'Sucha'))

    form.subheader('Czy Twoja cera jest wrażliwa?')
    isSensitive = form.radio(
     "",
     ('Tak', 
     'Nie'))

    if isSensitive == 'Tak':
        isSensitive = 1
    elif isSensitive == 'Nie':
        isSensitive = 0

    form.subheader('Jaki jest Twój największy problem z cerą?')        
    mainProblem = form.radio(
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
    secondProblem = form.radio(
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

    if secondProblem == 'Nie mam więcej problemów z cerą':
        secondProblem = 'Brak'
    
    form.subheader('Ile masz lat?')
    age = form.slider("", 16, 60)
        
    clicked = form.form_submit_button("Wyślij")
    if clicked:
        
        resultDict = {'Typ cery': skinType,
                    'Główny problem': mainProblem,
                    'Poboczny problem': secondProblem,
                    'Wrażliwa': isSensitive,
                    'Wiek': age}
        resultDataframe = pd.DataFrame.from_dict([resultDict])
        for i in decisionColumnNames:
            problemModel = makeSingleProblemTree(i, dumDf, dataset)
            result = predictMyObject(problemModel, resultDataframe, i)
            #print(i, " - " ,result)
            resultSkinCare[i] = result

        st.session_state.accuracy = resultDataframe
        with st.spinner('Tworzę Twój plan pielęgnacyjny...'):
            time.sleep(3)
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
        resultDataframe = pd.DataFrame({"Kategoria": accuracy.keys(), "Dokładność": accuracy.values()})
         
        st.dataframe(data=resultDataframe)

        st.stop()

def createLabelEncoding(datasetToEncode):
    global encoders, labelsDescription
    
    for i in allCategoricalColumns:
        encoders[i] = LabelEncoder()
        uniqueValues = list(datasetToEncode[i].unique())
        encoders[i] = encoders[i].fit(uniqueValues)
        datasetToEncode[i] = encoders[i].transform(datasetToEncode[i])
        
    return datasetToEncode

def main():
    global products
    products = pd.read_csv("products.csv", sep=';')
    products = products.to_dict()
    dataset = pd.read_csv("daneSkinCare.csv", sep=';')
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
    encodedDataframe = createLabelEncoding(dataset)
    showGUI(encodedDataframe, dataset, products)

if __name__ == '__main__':
    main()