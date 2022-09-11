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

botToken = '5660046213:AAHCSDYbdW7E5rc5MnoL1n8QCY-Qh8M1ZgI'
chatId = '5303880405'

askedColumnNames = ['Typ cery', 'Główny problem', 'Poboczny problem', 'Wrażliwa','Wiek']
categoricalColumnNames = ['Typ cery', 'Główny problem', 'Poboczny problem']
decisionColumnNames = ['Mycie','Serum na dzień','Krem na dzień','SPF','Serum na noc','Krem na noc','Punktowo','Maseczka','Peeling']
allColumns = askedColumnNames + decisionColumnNames
allCategoricalColumns = categoricalColumnNames + decisionColumnNames

def send_message(chatId, message):
    url = f"https://api.telegram.org/bot{botToken}/sendMessage?chat_id={chatId}&text={message}"
    requests.get(url)

def createMessage(inputData, message):

    for i in len(inputData.columns):
        result += inputData.columns[i] + inputData[i] + "\n"    
    result += message

    return result

def clearText(text):
    text = str(text).replace("'","").replace("[","").replace("]","").replace("\\xa0", " ")
    return text

def makeSingleProblemTree(problemName, dumDf, dataset):
    problemIndex = 0
    global graphCounter, accuracy
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
            print('Error')
    X = dumDf.values[:, 1:6]
    Y_problem = dataset.values[:, problemIndex]
    X_train, X_test, y_train, y_test = train_test_split(X, Y_problem, test_size = 0.25, random_state = 100)
    model = DecisionTreeClassifier(max_depth=16)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy[problemName] = str(accuracy_score(y_test, y_pred)*100) # "{} - Accuracy : {}".format(problem_name,accuracy_score(y_test, y_pred)*100)
    
    unique_values = dataset[problemName].unique()
    
    graphCounter = graphCounter + 1
    return model

def predictMyObject(model, my_object, columnName):
    global encoders
    
    for i in categoricalColumnNames:
        labelsDescription = encoders[i].classes_
        labelsDescription = labelsDescription.tolist()
        for l in labelsDescription:
            if my_object[i].item() == l:
                #print("Klasa - ", my_object[i].item(), "Label -> ", labelsDescription.index(l))
                my_object[i] = labelsDescription.index(l)
        # my_object[i] = np.where(labelsDescription == my_object[i])
    prediction = model.predict(my_object)
    prediction = encoders[columnName].inverse_transform(prediction)
    return prediction

def createLabelEncoding(datasetToEncode):
    global encoders, labelsDescription
    
    for i in allCategoricalColumns:
        encoders[i] = LabelEncoder()
        uniqueValues = list(datasetToEncode[i].unique())
        encoders[i] = encoders[i].fit(uniqueValues)
        datasetToEncode[i] = encoders[i].transform(datasetToEncode[i])
        
    return datasetToEncode

def setPhoto(category, side):
    global products        
    if side == 'left':
        link = str(resultSkinCare.get(category))
        #print("----------------------- link" + link)
        value = clearText(str(products.get(clearText(link)))).replace("{","").replace("0: ","").replace("}","")
        if value != "0": 
            col1, col2, = st.columns([1,3])
            with col1:
                if value != "0":
                    try:
                        st.image(value, width=150)
                    except:
                        st.error("Wystąpił błąd! Proszę spróbować później.")
                        send_message(chatId, "Błąd podczas wyświetlania zdjęcia " + value)
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
        #print("----------------------- link" + link)
        value = clearText(str(products.get(clearText(link)))).replace("{","").replace("0: ","").replace("}","")
        if value != "0": 
            #print("----------------------- val" + value)
            col1, col2, = st.columns([3,1])
            with col1:
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown(clearText(resultSkinCare.get(category)))
            with col2:
                #print("-----------------------" + value)
                try:
                    st.image(value, width=150)
                except:
                    st.error("Wystąpił błąd! Proszę spróbować później.")
                    send_message(chatId, "Błąd podczas wyświetlania zdjęcia " + value)
        else:
            st.markdown(clearText(resultSkinCare.get(category)))
           

def showGUI(dum_df, dataset, products):
    global skinType, isSensitive, mainProblem, secondProblem, age, accuracy

    st.set_page_config(
     page_title="System rekomendacyjny, do tworzenia planów pielęgnacyjnych",
     menu_items={
        'Report a bug': "https://forms.gle/5KV7rdhNi8epigL26",
        'About': "# Praca inżynierska. *s20943*"
        }
    )

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
        
            myDataframe = {'Typ cery': skinType,
                        'Główny problem': mainProblem,
                        'Poboczny problem': secondProblem,
                        'Wrażliwa': isSensitive,
                        'Wiek': age}
            
            df = pd.DataFrame.from_dict([myDataframe])
            for i in decisionColumnNames:
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
            try:
                setPhoto('Mycie', 'left')
            except:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                send_message(chatId, createMessage(myDataframe, "Błąd podczas wyświetlania produktu do mycia"))
            st.subheader('Serum na dzień')
            try:
                setPhoto('Serum na dzień', 'right')
            except:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                send_message(chatId, createMessage(myDataframe, "Błąd podczas wyświetlania produktu serum na dzień"))
            st.subheader('Krem na dzień')
            try:
                setPhoto('Krem na dzień', 'left')
            except:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                send_message(chatId, createMessage(myDataframe, "Błąd podczas wyświetlania produktu krem na dzień"))
            st.subheader('Krem przeciwsłoneczny')
            try:
                setPhoto('SPF', 'right')
            except:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                send_message(chatId, createMessage(myDataframe, "Błąd podczas wyświetlania produktu SPF"))
            st.subheader('Serum na noc')
            try:
                setPhoto('Serum na noc', 'left')
            except:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                send_message(chatId, createMessage(myDataframe, "Błąd podczas wyświetlania produktu serum na noc"))
            st.subheader('Krem na noc')
            try:
                setPhoto('Krem na noc', 'right')
            except:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                send_message(chatId, createMessage(myDataframe, "Błąd podczas wyświetlania produktu krem na noc"))
            st.subheader('Punktowo')
            try:
                setPhoto('Punktowo', 'left')
            except:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                send_message(chatId, createMessage(myDataframe, "Błąd podczas wyświetlania produktu punktowego"))
            st.subheader('Maseczka')
            try:
                setPhoto('Maseczka', 'right')
            except:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                send_message(chatId, createMessage(myDataframe, "Błąd podczas wyświetlania produktu maseczka"))
            st.subheader('Peeling')
            try:
                setPhoto('Peeling', 'left')
            except:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                send_message(chatId, createMessage(myDataframe, "Błąd podczas wyświetlania produktu peeling"))

            # devClicked = st.button("Strefa dewelopera")
            # if devClicked:
            #     #open("dev_page.py")
            df = pd.DataFrame({"Kategoria": accuracy.keys(), "Dokładność": accuracy.values()})
            
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
    dum_df = createLabelEncoding(dataset)
    showGUI(dum_df, dataset, products)

if __name__ == '__main__':
    main()