import requests

def send_message_to_telegram(message):
    chatId = '5303880405'
    botToken = '5660046213:AAHCSDYbdW7E5rc5MnoL1n8QCY-Qh8M1ZgI'
    url = f"https://api.telegram.org/bot{botToken}/sendMessage?chat_id={chatId}&text={message}"
    requests.get(url)

def create_message(inputDict, message):
    result = message + '\n'
    for i in inputDict.keys():
        result += f"{str(i)}: {str(inputDict[i])}" + '\n'
    return result