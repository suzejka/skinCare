from pickle import FALSE
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import time
import streamlit as st
import warnings 
import requests
warnings.filterwarnings("ignore")

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
link = None
chosenProduct = None

askedColumnNames = ['Typ cery', 'Główny problem', 'Poboczny problem', 'Wrażliwa','Wiek']
categoricalColumnNames = ['Typ cery', 'Główny problem', 'Poboczny problem']
decisionColumnNames = ['Mycie','Serum na dzień','Krem na dzień','SPF','Serum na noc','Krem na noc','Punktowo','Maseczka','Peeling']
allColumns = askedColumnNames + decisionColumnNames
allCategoricalColumns = categoricalColumnNames + decisionColumnNames

def sendToTelegram(message):
    chatId = '5303880405'
    botToken = '5660046213:AAHCSDYbdW7E5rc5MnoL1n8QCY-Qh8M1ZgI'
    url = f"https://api.telegram.org/bot{botToken}/sendMessage?chat_id={chatId}&text={message}"
    requests.get(url)

def createMessage(inputDict, message):
    result = message + '\n'
    for i in inputDict.keys():
        result += str(i) + ": " + str(inputDict[i]) + '\n'
    return result

def clearText(text):
    text = str(text).replace("'","").replace("[","").replace("]","").replace("\\xa0", " ")
    return text

def clearSkinTypeMisspelledData(datasetToClean):
    datasetToClean.loc[(datasetToClean["Typ cery"] == "tłusta") | 
                        (datasetToClean["Typ cery"] == "tlusta") | 
                        (datasetToClean["Typ cery"] == "Tlusta"), "Typ cery"] = "Tłusta"
    datasetToClean.loc[(datasetToClean["Typ cery"] == "mieszana"), "Typ cery"] = "Mieszana"
    datasetToClean.loc[(datasetToClean["Typ cery"] == "sucha"), "Typ cery"] = "Sucha"
    datasetToClean.loc[(datasetToClean["Typ cery"] == "normalna"), "Typ cery"] = "Normalna"

def clearMainOrsecondProblemMisspelledData(datasetToClean, mainOrsecondProblem):
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "nadprodukcja sebum") | 
                        (datasetToClean[mainOrsecondProblem] == "nadprodukcja Sebum"), mainOrsecondProblem] = "Nadprodukcja sebum"
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "niedoskonałości") | 
                        (datasetToClean[mainOrsecondProblem] == "niedoskonalosci") |
                        (datasetToClean[mainOrsecondProblem] == "niedoskonałosci") |
                        (datasetToClean[mainOrsecondProblem] == "niedoskonalości") |
                        (datasetToClean[mainOrsecondProblem] == "Niedoskonalości") | 
                        (datasetToClean[mainOrsecondProblem] == "Niedoskonalosci") |
                        (datasetToClean[mainOrsecondProblem] == "Niedoskonałosci"), mainOrsecondProblem] = "Niedoskonałości"
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "podrażnienie") | 
                        (datasetToClean[mainOrsecondProblem] == "podraznienie") |
                        (datasetToClean[mainOrsecondProblem] == "Podraznienie") , mainOrsecondProblem] = "Podrażnienie"
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "przebarwienia"), mainOrsecondProblem] = "Przebarwienia"
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "rozszerzone pory") | 
                        (datasetToClean[mainOrsecondProblem] == "Rozszerzone Pory") |
                        (datasetToClean[mainOrsecondProblem] == "rozszerzone Pory") , mainOrsecondProblem] = "Rozszerzone pory"
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "suche skórki") | 
                        (datasetToClean[mainOrsecondProblem] == "suche Skórki") |
                        (datasetToClean[mainOrsecondProblem] == "Suche skorki") |
                        (datasetToClean[mainOrsecondProblem] == "suche skorki") |
                        (datasetToClean[mainOrsecondProblem] == "suche Skorki"), mainOrsecondProblem] = "Suche skórki"
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "szara cera") | 
                        (datasetToClean[mainOrsecondProblem] == "szara Cera") |
                        (datasetToClean[mainOrsecondProblem] == "Szara Cera"), mainOrsecondProblem] = "Szara cera"
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "widoczne naczynka") | 
                        (datasetToClean[mainOrsecondProblem] == "Widoczne Naczynka") |
                        (datasetToClean[mainOrsecondProblem] == "widoczne Naczynka"), mainOrsecondProblem] = "Widoczne naczynka"
    if mainOrsecondProblem == "Poboczny problem":
        datasetToClean.loc[(datasetToClean["Poboczny problem"] == "brak"), "Poboczny problem"] = "Brak"

def clearData(datasetToClean):
    datasetToClean.dropna()
    clearSkinTypeMisspelledData(datasetToClean)
    clearMainOrsecondProblemMisspelledData(datasetToClean, "Główny problem")
    clearMainOrsecondProblemMisspelledData(datasetToClean, "Poboczny problem")
    datasetToClean.drop_duplicates(inplace=True)
    datasetToClean = datasetToClean[datasetToClean["Wiek"] >= 16]

def makeSingleProblemTree(problemName, dumDf, dataset):
    problemIndex = 0
    global accuracy
    match problemName:
        case 'Mycie':
            problemIndex = 6
        case 'Serum na dzień':
            problemIndex = 7
        case 'Krem na dzień':
            problemIndex = 8
        case 'SPF' :
            problemIndex = 9
        case 'Serum na noc':
            problemIndex = 10
        case 'Krem na noc':
            problemIndex = 11
        case 'Punktowo':
            problemIndex = 12
        case 'Maseczka':
            problemIndex = 13
        case 'Peeling':
            problemIndex = 14
        case _:   
            sendToTelegram("Błąd! Nie rozpoznano kategorii produktu.")
            raise ValueError("Nie rozpoznano kategorii produktu.")

    X = dumDf.values[:, 1:6]
    yProblem = dataset.values[:, problemIndex]
    X_train, X_test, y_train, y_test = train_test_split(X, yProblem, test_size = 0.25, random_state = 100)
    model = DecisionTreeClassifier(max_depth=16)
    model = model.fit(X_train, y_train)
    yPrediction = model.predict(X_test)
    accuracy[problemName] = str(accuracy_score(y_test, yPrediction)*100) # "{} - Accuracy : {}".format(problem_name,accuracy_score(y_test, y_pred)*100)
   
    return model

def predictMyObject(model, objectToPredict, columnName):
    global encoders
    
    for categoricalColumn in categoricalColumnNames:
        labelsDescription = encoders[categoricalColumn].classes_
        labelsDescription = labelsDescription.tolist()
        for label in labelsDescription:
            if objectToPredict[categoricalColumn].item() == label:
                objectToPredict[categoricalColumn] = labelsDescription.index(label)
    prediction = model.predict(objectToPredict)
    prediction = encoders[columnName].inverse_transform(prediction)
    return prediction

def createLabelEncoding(datasetToEncode):
    global encoders, labelsDescription
    
    for categoricalColumn in allCategoricalColumns:
        encoders[categoricalColumn] = LabelEncoder()
        uniqueValues = list(datasetToEncode[categoricalColumn].unique())
        encoders[categoricalColumn] = encoders[categoricalColumn].fit(uniqueValues)
        datasetToEncode[categoricalColumn] = encoders[categoricalColumn].transform(datasetToEncode[categoricalColumn])
        
    return datasetToEncode   
    

def setPhoto(category, side):
    global products, link, chosenProduct
    if side == 'left':
        link = str(resultSkinCare.get(category))
        chosenProduct = clearText(str(products.get(clearText(link)))).replace("{","").replace("0: ","").replace("}","")
        if chosenProduct != "0": 
            col1, col2, = st.columns([1,3])
            with col1:
                if chosenProduct != "0":
                    st.image(chosenProduct, width=150)
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
        chosenProduct = clearText(str(products.get(clearText(link)))).replace("{","").replace("0: ","").replace("}","")
        if chosenProduct != "0": 
            col1, col2, = st.columns([3,1])
            with col1:
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown(clearText(resultSkinCare.get(category)))
            with col2:
                try:
                    st.image(chosenProduct, width=150)
                except:
                    st.error("Wystąpił błąd! Proszę spróbować później.")
                    sendToTelegram("Błąd podczas wyświetlania zdjęcia " + chosenProduct)
        else:
            st.markdown(clearText(resultSkinCare.get(category)))
           

def showGUI(convertedDataset, plainDataset):
    global skinType, isSensitive, mainProblem, secondProblem, age, accuracy, link, chosenProduct   

    st.set_page_config(
     page_title="System rekomendacyjny, do tworzenia planów pielęgnacyjnych",
     menu_items={
        'Report a bug': "https://forms.gle/5KV7rdhNi8epigL26",
        'About': "# Praca inżynierska. *s20943*"
        },
    page_icon="skincareIcon.png"
    )

    col1, col2, = st.columns([1,3])
    with col1:
        st.image("skincareIcon.png", width=120)
    with col2:
        st.title("Kreator planów pielęgnacyjnych")

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
    age = form.slider("", 16, 100)
        
    clicked = form.form_submit_button("Wyślij")
    if clicked:
        
            userData = {'Typ cery': skinType,
                        'Główny problem': mainProblem,
                        'Poboczny problem': secondProblem,
                        'Wrażliwa': isSensitive,
                        'Wiek': age}
            
            userDataFrame = pd.DataFrame.from_dict([userData])
            st.session_state.accuracy = userDataFrame
            with st.spinner('Tworzę Twój plan pielęgnacyjny...'):
                for singleDecisionColumn in decisionColumnNames:
                    problemModel = makeSingleProblemTree(singleDecisionColumn, convertedDataset, plainDataset)
                    result = predictMyObject(problemModel, userDataFrame, singleDecisionColumn)
                    resultSkinCare[singleDecisionColumn] = result

            st.success('Skończone!')
    
            st.header('Proponowana pielęgnacja')
            counter = 0
            for name in decisionColumnNames:
                st.subheader(name)
                try:
                    side = 'right'
                    if(counter % 2 == 0):
                        side = 'left'
                    setPhoto(name, side)
                    counter += 1
                except:
                    st.error("Wystąpił błąd! Proszę spróbować później.")
                    sendToTelegram(createMessage(userData, "Błąd podczas wyświetlania produktu - " + name 
                    + "\nLink - " + str(link) 
                    + "\nProdukt - " + str(chosenProduct))
                    )
            helpMessage = "1. Przedstawione produkty to tylko i wyłącznie PROPOZYCJA pielęgnacji! Użycie programu nie zastąpi wizyty u specjalisty!\n2. Jeżeli zaproponowana maseczka składa się z dwóch produktów, oznacza to, że na początku należy nałożyć pierwszy produkt i następnie (bez zmywania) nałożyć maseczkę. W przypadku kwasu salicylowego, należy odczekać 15/20 minut przed nałożeniem maseczki. \n3. Jeżeli proponowana maseczka zawiera w sobie glinkę, należy pamiętać, że glinka nigdy nie powinna zasychać, dlatego warto dodać do maseczki kilka kropel ulubionego oleju kosmetycznego lub nałożoną maseczkę zwilżać poprzez spryskiwanie twarzy wodą."
            st.caption("")
            st.caption("")
            st.caption("")
            st.caption(helpMessage)
            # devClicked = st.button("Strefa dewelopera")
            # if devClicked:
            #     #open("dev_page.py")
            userDataFrame = pd.DataFrame({"Kategoria": accuracy.keys(), "Dokładność": accuracy.values()})
            
            #     st.dataframe(data=df)

            #for i in df:
            #   print(df[i].to_string(index = False))

            st.stop()

def main():
    global products
    products = pd.read_csv("products.csv", sep=';')
    products = products.to_dict()
    dataset = pd.read_csv("daneSkinCare.csv", sep=';')
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
    clearData(dataset)
    labeledDataset = createLabelEncoding(dataset)
    showGUI(labeledDataset, dataset)

if __name__ == '__main__':
    main()