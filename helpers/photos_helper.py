import streamlit as st
import telegram_bot_for_messages as bot
import text_cleaner as cleaner

def show_photo_using_link(link):
    '''
    Shows photo using link
    '''
    if link != "0":
        try:
            st.image(link, width=150)
        except Exception:
            st.error("Wystąpił błąd! Proszę spróbować później.")
            bot.send_message_to_telegram(f"Błąd podczas wyświetlania zdjęcia {link}")

def set_left_photo(category, result, link):
    '''
    Sets photo of product on the left side
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
    Sets photo of product on the right side
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

def set_left_or_right_photo(name, counter, PRODUCTS, PREDICTED_PRODUCT, CHOSEN_PRODUCT_LINK, RESULT_SKIN_CARE, ACCURACY):
    '''
    Sets photo of product on the left or right side
    '''
    side = 'left' if counter % 2 == 0 else 'right'
    set_photo(name, side, PRODUCTS, PREDICTED_PRODUCT, CHOSEN_PRODUCT_LINK, RESULT_SKIN_CARE, ACCURACY)

def does_product_link_exist_in_product_dataset(product, PRODUCTS):
    '''
    Checks if product link exists in product dataset
    '''
    return PRODUCTS.keys().__contains__(product) or product is not None

def show_only_product_name(category, RESULT_SKIN_CARE):
    '''
    Shows only product name
    '''
    st.markdown(cleaner.remove_punctuation_marks(RESULT_SKIN_CARE.get(category)))

def set_photo(category, side, PRODUCTS, PREDICTED_PRODUCT, CHOSEN_PRODUCT_LINK, RESULT_SKIN_CARE, ACCURACY):
    '''
    Sets photo of product
    '''
    PREDICTED_PRODUCT = str(RESULT_SKIN_CARE.get(category))

    if does_product_link_exist_in_product_dataset(PREDICTED_PRODUCT, PRODUCTS):
        CHOSEN_PRODUCT_LINK = cleaner.clean_product_link(PREDICTED_PRODUCT, PRODUCTS)
    else:
        CHOSEN_PRODUCT_LINK = "0"

    if CHOSEN_PRODUCT_LINK in ["0", "None", "nan"]:
        show_only_product_name(category, RESULT_SKIN_CARE)
    elif side == 'left':
        set_left_photo(category, RESULT_SKIN_CARE, CHOSEN_PRODUCT_LINK)
    else:
        set_right_photo(category, RESULT_SKIN_CARE, CHOSEN_PRODUCT_LINK)

    st.caption(f"Dokładność przewidywania: {str(round(ACCURACY.get(category) * 100, 2))}%")