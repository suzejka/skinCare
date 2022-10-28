from inspect import stack
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import warnings 
import telegramBot as bot
import textCleaner as cleaner
import traceback
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

def clear_text(text):
    text = str(text).replace("'","").replace("[","").replace("]","").replace("\\xa0", " ")
    return text

def predict_my_object(model, objectToPredict, columnName):
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
    global ENCODERS, LABELS_DESCRIPTION
    for categoricalColumn in ALL_CATEGORICAL_COLUMNS:
        ENCODERS[categoricalColumn] = LabelEncoder()
        uniqueValues = list(datasetToEncode[categoricalColumn].unique())
        ENCODERS[categoricalColumn] = ENCODERS[categoricalColumn].fit(uniqueValues)
        datasetToEncode[categoricalColumn] = ENCODERS[categoricalColumn].transform(datasetToEncode[categoricalColumn])
    return datasetToEncode

def set_chosen_product_link(category):
    global CHOSEN_PRODUCT_LINK, RESULT_SKIN_CARE, PREDICTED_PRODUCT, PRODUCTS
    

def set_photo(category, side):
    global PRODUCTS, PREDICTED_PRODUCT, CHOSEN_PRODUCT_LINK, RESULT_SKIN_CARE
    if side == 'left':
        PREDICTED_PRODUCT = str(RESULT_SKIN_CARE.get(category))
        if PRODUCTS.keys().__contains__(PREDICTED_PRODUCT) or PREDICTED_PRODUCT is None:
            CHOSEN_PRODUCT_LINK = "0"
            print(":)")
        else:
            CHOSEN_PRODUCT_LINK = clear_text(str(PRODUCTS.get(clear_text(PREDICTED_PRODUCT)))).replace("{","").replace("0: ","").replace("}","")
        
        if CHOSEN_PRODUCT_LINK != "0" and CHOSEN_PRODUCT_LINK != "None" and CHOSEN_PRODUCT_LINK != "nan":
            col1, col2, = st.columns([1,3])
            with col1:
                if CHOSEN_PRODUCT_LINK != "0":
                    st.image(CHOSEN_PRODUCT_LINK, width=150)
            with col2:
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown(clear_text(RESULT_SKIN_CARE.get(category)))
        else:
            st.markdown(clear_text(RESULT_SKIN_CARE.get(category)))
    else:
        PREDICTED_PRODUCT = str(RESULT_SKIN_CARE.get(category))
        if PRODUCTS.keys().__contains__(PREDICTED_PRODUCT) or PREDICTED_PRODUCT is None:
            CHOSEN_PRODUCT_LINK = "0"
        else:
            CHOSEN_PRODUCT_LINK = clear_text(str(PRODUCTS.get(clear_text(PREDICTED_PRODUCT)))).replace("{","").replace("0: ","").replace("}","")
        if CHOSEN_PRODUCT_LINK != "0" and CHOSEN_PRODUCT_LINK != "None" and CHOSEN_PRODUCT_LINK != "nan":
            col1, col2, = st.columns([3,1])
            with col1:
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown(clear_text(RESULT_SKIN_CARE.get(category)))
            with col2:
                try:
                    st.image(CHOSEN_PRODUCT_LINK, width=150)
                except Exception:
                    st.error("Wystąpił błąd! Proszę spróbować później.")
                    bot.send_message_to_telegram("Błąd podczas wyświetlania zdjęcia " + CHOSEN_PRODUCT_LINK)
        else:
            st.markdown(clear_text(RESULT_SKIN_CARE.get(category)))
           
def show_gui():
    global SKIN_TYPE, IS_SENSITIVE, MAIN_PROBLEM, SECOND_PROBLEM, AGE, ACCURACY, PREDICTED_PRODUCT, CHOSEN_PRODUCT_LINK   

    st.set_page_config(
     page_title="System rekomendacyjny, do tworzenia planów pielęgnacyjnych",
     menu_items={
        'Report a bug': "https://forms.gle/5KV7rdhNi8epigL26",
        'About': "# Praca inżynierska. *s20943*"
        },
    page_icon="skincareIcon.png"
    )

    st.title("Kreator planów pielęgnacyjnych")

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

    if form.form_submit_button("Wyślij"):
        userData = {'Typ cery': SKIN_TYPE,
                    'Główny problem': MAIN_PROBLEM,
                    'Poboczny problem': SECOND_PROBLEM,
                    'Wrażliwa': IS_SENSITIVE,
                    'Wiek': AGE}

        userDataFrame = pd.DataFrame.from_dict([userData])
        st.session_state.accuracy = userDataFrame
        with st.spinner('Tworzę Twój plan pielęgnacyjny...'):
            for singleDecisionColumn in DECISION_COLUMN_NAMES:
                filename = "{column}.sv".format(column = singleDecisionColumn.replace(" ", "_"))
                problemModel = pickle.load(open(filename, 'rb'))
                result = predict_my_object(problemModel, userDataFrame, singleDecisionColumn)
                RESULT_SKIN_CARE[singleDecisionColumn] = result

        st.success('Skończone!')

        st.header('Proponowana pielęgnacja')
        counter = 0
        for name in DECISION_COLUMN_NAMES:
            st.subheader(name)
            try:
                side = 'right'
                if counter % 2 == 0:
                    side = 'left'
                set_photo(name, side)
                counter += 1
            except Exception:
                st.error("Wystąpił błąd! Proszę spróbować później.")
                bot.send_message_to_telegram(bot.create_message(userData,
                "Błąd podczas wyświetlania produktu - " + name
                + "\nProdukt - " + str(PREDICTED_PRODUCT) 
                + "\nLink - " + str(CHOSEN_PRODUCT_LINK))
                + "\n" + traceback.format_exc())
        helpMessage = "1. Przedstawione produkty to tylko i wyłącznie PROPOZYCJA pielęgnacji! Użycie programu nie zastąpi wizyty u specjalisty!\n2. Jeżeli zaproponowana maseczka składa się z dwóch produktów, oznacza to, że na początku należy nałożyć pierwszy produkt i następnie (bez zmywania) nałożyć maseczkę. W przypadku kwasu salicylowego, należy odczekać 15/20 minut przed nałożeniem maseczki. \n3. Jeżeli proponowana maseczka zawiera w sobie glinkę, należy pamiętać, że glinka nigdy nie powinna zasychać, dlatego warto dodać do maseczki kilka kropel ulubionego oleju kosmetycznego lub nałożoną maseczkę zwilżać poprzez spryskiwanie twarzy wodą."
        st.caption("")
        st.caption("")
        st.caption("")
        st.caption(helpMessage)
        userDataFrame = pd.DataFrame(
            {"Kategoria": ACCURACY.keys(), 
            "Dokładność": ACCURACY.values()}
            )
        
        st.stop()

def main():
    global PRODUCTS
    PRODUCTS = pd.read_csv("products.csv", sep=';')
    PRODUCTS = PRODUCTS.to_dict()
    dataset = pd.read_csv("DATASET.csv", encoding='cp1250')
    create_label_encoding(dataset)
    show_gui()

if __name__ == '__main__':
    main()