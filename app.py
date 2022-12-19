import telebot
from config import token # Не забудь создать себе токен!
from bert_utils import *

bot=telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет')


@bot.message_handler(content_types='text')
def message_reply(message):
    
    print('Вижу новое сообщение!')
    text, score = get_toxic_score(message.text)
    answer = f'{text} \nПроцент токсичности: {score}'

    bot.send_message(message.chat.id, answer)

    
if __name__ == '__main__':
    print('Bot is staring...')
    bot.infinity_polling() # запускаем бота, чтоб он работал вечно
