import streamlit as st

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

def if_SPF_change_name_for_user(name):
    '''
    Funkcja odpowiedzialna za zmianę nazwy produktu SPF na "Krem z filtrem UV"
    '''
    return "Krem z filtrem przeciwsłonecznym" if name == "SPF" else name