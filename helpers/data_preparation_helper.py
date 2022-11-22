import telegram_bot_for_messages as bot

def get_problem_column_index(problemName):
    '''
    Funkcja zwraca indeks kolumny z danym problemem.
    '''
    if problemName == 'Mycie':
        return 5
    elif problemName == 'Serum na dzień':
        return 6
    elif problemName == 'Krem na dzień':
        return  7
    elif problemName == 'SPF' :
        return  8
    elif problemName == 'Serum na noc':
        return  9
    elif problemName == 'Krem na noc':
        return  10
    elif problemName == 'Punktowo':
        return  11
    elif problemName == 'Maseczka':
        return  12
    elif problemName == 'Peeling':
        return  13
    else :
        bot.send_message_to_telegram("Błąd! Nie rozpoznano kategorii produktu.")
        raise ValueError("Nie rozpoznano kategorii produktu.")