#!/usr/bin/python3
import argparse
import telebot, requests, json

BOT_TOKEN='token'

def response(msg):    
    message = msg
    data = '{"message": "' + message + '"}'
    s = requests.Session()
    r = s.post(url = 'http://ip:port/input', data = data, verify=False)
    
    if r.status_code == 200:
        print("Message sent successfully.")
    else:
        print(f"Failed to send message. Status code: {r.status_code}, Response: {r.text}")

def ban(chat_id, user_id):
    s = requests.Session()
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/banChatMember"
    data = {
        'chat_id': chat_id,
        'user_id': user_id,
        'revoke_messages': True
        }

    r = s.post(url, data = data, verify=False)

    if r.status_code == 200:
        success_message = f"User {user_id} has been banned from chat {chat_id}."
        print(success_message)
        response(success_message)
    else:
        error_message = f"Failed to ban user. Status code: {r.status_code}, Response: {r.text}"
        print(error_message)
        response(error_message)

def deletemsg(chat_id, message_id):
    s = requests.Session()
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteMessage"
    data = {
        'chat_id': chat_id,
        'message_id': message_id,
        }

    r = s.post(url, data = data, verify=False)


parser = argparse.ArgumentParser(description='Process user, message and chat IDs.')

    # Добавление аргументов
parser.add_argument('--userid', type=str, required=True, help='User ID')
parser.add_argument('--chatid', type=str, required=True, help='Chat ID')
parser.add_argument('--messageid', type=str, required=True, help='Message ID')
    # Разбор аргументов
args = parser.parse_args()

    # Вывод переданных значений
print(f"User ID: {args.userid}")
print(f"Chat ID: {args.chatid}")

user_id = args.userid
chat_id = args.chatid
message_id = args.messageid

ban(chat_id, user_id)
deletemsg(chat_id, message_id)
