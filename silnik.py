import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import warnings 
import telegram_bot_for_messages as bot
from helpers.model_to_file_helper import read_accuray_from_file
from helpers import page_helper as ph
import traceback
from helpers import photos_helper
warnings.filterwarnings("ignore")

ENCODERS = {}
SKIN_TYPE = None
IS_SENSITIVE = None
MAIN_PROBLEM = None
SECOND_PROBLEM = None
AGE = None
RESULT_SKIN_CARE = {}
ACCURACY = {}
PRODUCTS = {}
PREDICTED_PRODUCT = None
CHOSEN_PRODUCT_LINK = None

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

def predict_my_object(model, objectToPredict, columnName):
    '''
    Predicts new object
    '''
    global ENCODERS
    for categoricalColumn in CATEGORICAL_COLUMN_NAMES:
        labelsDescription = ENCODERS[categoricalColumn].classes_
        labelsDescription = labelsDescription.tolist()
        for label in labelsDescription:
            if objectToPredict[categoricalColumn].item() == label:
                objectToPredict[categoricalColumn] = labelsDescription.index(label)
    prediction = model.predict(objectToPredict)
    prediction = ENCODERS[columnName].inverse_transform(prediction)
    return prediction

def create_label_encoding(datasetToEncode):
    '''
    Creates label encoding
    '''
    global ENCODERS
    for categoricalColumn in ALL_CATEGORICAL_COLUMNS:
        ENCODERS[categoricalColumn] = LabelEncoder()
        uniqueValues = list(datasetToEncode[categoricalColumn].unique())
        ENCODERS[categoricalColumn] = ENCODERS[categoricalColumn].fit(uniqueValues)
        datasetToEncode[categoricalColumn] = ENCODERS[categoricalColumn].transform(datasetToEncode[categoricalColumn])
    return datasetToEncode

def create_form():
    '''
    Creates form for input data
    '''
    global SKIN_TYPE, IS_SENSITIVE, MAIN_PROBLEM, SECOND_PROBLEM, AGE
    form = st.form("my_form")
    form.subheader('Jaki masz typ cery?')
    SKIN_TYPE = form.radio(
     "",
     ('Tłusta',
     'Normalna',
     'Mieszana',
     'Sucha'))

    form.subheader('Czy Twoja cera jest wrażliwa?')
    IS_SENSITIVE = form.radio(
     "",
     ('Tak', 
     'Nie'))

    if IS_SENSITIVE == 'Tak':
        IS_SENSITIVE = 1
    elif IS_SENSITIVE == 'Nie':
        IS_SENSITIVE = 0

    form.subheader('Jaki jest Twój największy problem z cerą?')
    MAIN_PROBLEM = form.radio(
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
    SECOND_PROBLEM = form.radio(
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

    if SECOND_PROBLEM == 'Nie mam więcej problemów z cerą':
        SECOND_PROBLEM = 'Brak'

    form.subheader('Ile masz lat?')
    AGE = form.slider("", 16, 100)
    return form

def predict_result_using_input_data(userDataFrame):
    '''
    Predicts result using input data
    '''
    global RESULT_SKIN_CARE, DECISION_COLUMN_NAMES
    for singleDecisionColumn in DECISION_COLUMN_NAMES:
        filename = "{column}.sv".format(column = singleDecisionColumn.replace(" ", "_"))
        problemModel = pickle.load(open(filename, 'rb'))
        result = predict_my_object(problemModel, userDataFrame, singleDecisionColumn)
        RESULT_SKIN_CARE[singleDecisionColumn] = result

def show_gui():
    '''
    Shows user interface
    '''
    global SKIN_TYPE, IS_SENSITIVE, MAIN_PROBLEM, SECOND_PROBLEM, AGE, ACCURACY, PREDICTED_PRODUCT, CHOSEN_PRODUCT_LINK, PRODUCTS, PREDICTED_PRODUCT, CHOSEN_PRODUCT_LINK, RESULT_SKIN_CARE, ACCURACY

    ph.set_configuration_of_page()

    st.markdown("<h1 style='text-align: center; color: black;'>Witaj w kreatorze planów pielęgnacyjnych!</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: black;'>Wypełnij poniższy formularz, a system wygeneruje dla Ciebie przykładowy plan pielęgnacyjny.</h1>", unsafe_allow_html=True)

    form = create_form()

    if form.form_submit_button("Wyślij"):
        userData = {'Typ cery': SKIN_TYPE,
                    'Główny problem': MAIN_PROBLEM,
                    'Poboczny problem': SECOND_PROBLEM,
                    'Wrażliwa': IS_SENSITIVE,
                    'Wiek': AGE}

        userDataFrame = pd.DataFrame.from_dict([userData])
        st.session_state.accuracy = userDataFrame
        with st.spinner('Tworzę Twój plan pielęgnacyjny...'):
            predict_result_using_input_data(userDataFrame)

        st.success('Skończone!')
        st.header('Proponowana pielęgnacja')
        st.info("Przedstawione produkty to tylko i wyłącznie PROPOZYCJA pielęgnacji! Użycie programu nie zastąpi wizyty u specjalisty!")
        
        counter = 0
        for name in DECISION_COLUMN_NAMES:
            header_label = ph.if_SPF_change_name_for_user(name)
            st.subheader(header_label)
            try:
                photos_helper.set_left_or_right_photo(name, counter, PRODUCTS, PREDICTED_PRODUCT, CHOSEN_PRODUCT_LINK, RESULT_SKIN_CARE, ACCURACY)
                counter += 1
            except Exception:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                bot.send_message_to_telegram(bot.create_message(userData,
                "Błąd podczas wyświetlania produktu - " + name
                + "\nProdukt - " + str(PREDICTED_PRODUCT) 
                + "\nLink - " + str(CHOSEN_PRODUCT_LINK))
                + "\n" + traceback.format_exc())
        ph.show_important_information()        
        st.stop()

def read_products():
    '''
    Read products from file
    '''
    global PRODUCTS
    PRODUCTS = pd.read_csv("raw_data/products.csv", sep=';')
    PRODUCTS = PRODUCTS.to_dict()

def main():
    global ACCURACY
    read_products()
    dataset = pd.read_csv("prepared_data/DATASET.csv", encoding='utf-16')
    create_label_encoding(dataset)
    ACCURACY = read_accuray_from_file()
    show_gui()

if __name__ == '__main__':
    main()