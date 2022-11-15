from inspect import stack
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import warnings 
import telegramBot as bot
import textCleaner as cleaner
import traceback
import json
warnings.filterwarnings("ignore")

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
    Funkcja przewidująca obiekt
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
    Tworzy kodowanie etykiet
    '''
    global ENCODERS, LABELS_DESCRIPTION
    for categoricalColumn in ALL_CATEGORICAL_COLUMNS:
        ENCODERS[categoricalColumn] = LabelEncoder()
        uniqueValues = list(datasetToEncode[categoricalColumn].unique())
        ENCODERS[categoricalColumn] = ENCODERS[categoricalColumn].fit(uniqueValues)
        datasetToEncode[categoricalColumn] = ENCODERS[categoricalColumn].transform(datasetToEncode[categoricalColumn])
    return datasetToEncode

def show_photo_using_link(link):
    '''
    Wyświetla zdjęcie produktu
    '''
    if link != "0":
        try:
            st.image(link, width=150)
        except Exception:
            st.error("Wystąpił błąd! Proszę spróbować później.")
            bot.send_message_to_telegram("Błąd podczas wyświetlania zdjęcia " + link)

def set_left_photo(category, result, link):
    '''
    Ustawia zdjęcie produktu po lewej stronie
    '''
    col1, col2, = st.columns([1,3])
    with col1:
        show_photo_using_link(link)
    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown(cleaner.remove_punctuation_marks(result.get(category)))

def set_right_photo(category, result, link):
    '''
    Ustawia zdjęcie produktu po prawej stronie
    '''
    col1, col2, = st.columns([3,1])
    with col1:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown(cleaner.remove_punctuation_marks(result.get(category)))
    with col2:
        show_photo_using_link(link)

def clear_product_link(product):
    '''
    Czyści link do produktu
    '''
    global PRODUCTS
    return cleaner.remove_punctuation_marks(str(PRODUCTS.get(cleaner.remove_punctuation_marks(product)))).replace("{","").replace("0: ","").replace("}","")

def does_product_link_exist_in_product_dataset(product):
    '''
    Sprawdza czy link do produktu istnieje w bazie produktów
    '''
    global PRODUCTS
    return PRODUCTS.keys().__contains__(product) or product is not None

def show_only_product_name(category):
    '''
    Wyświetla tylko nazwę produktu
    '''
    st.markdown(cleaner.remove_punctuation_marks(RESULT_SKIN_CARE.get(category)))

def set_photo(category, side):
    '''
    Ustawia zdjęcie produktu
    '''
    global PRODUCTS, PREDICTED_PRODUCT, CHOSEN_PRODUCT_LINK, RESULT_SKIN_CARE
    PREDICTED_PRODUCT = str(RESULT_SKIN_CARE.get(category))

    if does_product_link_exist_in_product_dataset(PREDICTED_PRODUCT):
        CHOSEN_PRODUCT_LINK = clear_product_link(PREDICTED_PRODUCT)
    else:
        CHOSEN_PRODUCT_LINK = "0"

    if CHOSEN_PRODUCT_LINK in ["0", "None", "nan"]:
        show_only_product_name(category)
    elif side == 'left':
        set_left_photo(category, RESULT_SKIN_CARE, CHOSEN_PRODUCT_LINK)
    else:
        set_right_photo(category, RESULT_SKIN_CARE, CHOSEN_PRODUCT_LINK)

    st.caption(f"Dokładność przewidywania: {str(ACCURACY.get(category))}%")
    
def read_accuray_from_file():
    '''
    Odczytuje accuracy z pliku
    '''
    global ACCURACY
    with open('accuracy.json') as json_file:
        ACCURACY = json.load(json_file)

def create_form():
    '''
    Funkcja tworzy formularz do wprowadzania danych
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

def set_left_or_right_photo(name, counter):
    '''
    Ustawia zdjęcie produktu na lewo lub prawo naprzemiennie
    '''
    side = 'left' if counter % 2 == 0 else 'right'
    set_photo(name, side)

def show_important_information():
    '''
    Wyświetla informacje o tym, że aplikacja nie jest lekarzem i nie może diagnozować
    '''
    helpMessage = "1. Jeżeli zaproponowana maseczka składa się z dwóch produktów, oznacza to, że na początku należy nałożyć pierwszy produkt i następnie (bez zmywania) nałożyć maseczkę. "\
    "W przypadku kwasu salicylowego, należy odczekać 15/20 minut przed nałożeniem maseczki. \n2. Jeżeli proponowana maseczka zawiera w sobie glinkę, należy pamiętać, "\
    "że glinka nigdy nie powinna zasychać, dlatego warto dodać do maseczki kilka kropel ulubionego oleju kosmetycznego lub nałożoną maseczkę zwilżać poprzez spryskiwanie "\
    "twarzy wodą."
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption(helpMessage)

def set_configuration_of_page():
    '''
    Ustawia konfigurację strony
    '''
    st.set_page_config(
    page_title="System rekomendacyjny, do tworzenia planów pielęgnacyjnych",
    menu_items={
    'Report a bug': "https://forms.gle/5KV7rdhNi8epigL26",
    'About': "# Praca inżynierska. *s20943*"
    },
    page_icon="skincareIcon.png"
    )

def predict_result_using_input_data(userDataFrame):
    '''
    Funkcja przewidująca wynik na podstawie danych wejściowych
    '''
    global RESULT_SKIN_CARE, DECISION_COLUMN_NAMES
    for singleDecisionColumn in DECISION_COLUMN_NAMES:
        filename = "{column}.sv".format(column = singleDecisionColumn.replace(" ", "_"))
        problemModel = pickle.load(open(filename, 'rb'))
        result = predict_my_object(problemModel, userDataFrame, singleDecisionColumn)
        RESULT_SKIN_CARE[singleDecisionColumn] = result

def show_gui():
    '''
    Funkcja odpowiedzialna za wyświetlenie interfejsu graficznego.
    '''
    global SKIN_TYPE, IS_SENSITIVE, MAIN_PROBLEM, SECOND_PROBLEM, AGE, ACCURACY, PREDICTED_PRODUCT, CHOSEN_PRODUCT_LINK   

    set_configuration_of_page()
    st.title("Kreator planów pielęgnacyjnych")

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
            st.subheader(name)
            try:
                set_left_or_right_photo(name, counter)
                counter += 1
            except Exception:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                bot.send_message_to_telegram(bot.create_message(userData,
                "Błąd podczas wyświetlania produktu - " + name
                + "\nProdukt - " + str(PREDICTED_PRODUCT) 
                + "\nLink - " + str(CHOSEN_PRODUCT_LINK))
                + "\n" + traceback.format_exc())
        show_important_information()

        # userDataFrame = pd.DataFrame(
        #     {"Kategoria": ACCURACY.keys(), 
        #     "Dokładność": ACCURACY.values()}
        #     )
        
        st.stop()

def read_products():
    '''
    Funkcja wczytująca produkty z pliku
    '''
    global PRODUCTS
    PRODUCTS = pd.read_csv("products.csv", sep=';')
    PRODUCTS = PRODUCTS.to_dict()

def main():
    read_products()
    dataset = pd.read_csv("DATASET.csv", encoding='utf-16')
    create_label_encoding(dataset)
    read_accuray_from_file()
    show_gui()

if __name__ == '__main__':
    main()