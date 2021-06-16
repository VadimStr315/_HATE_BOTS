import nltk
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split

with open('CLEAR_BIG_BOT_CONFIG.json',encoding='utf-8') as f:
    BOT_CONFIG = json.load(f)  # считываем json в переменную


def cleaner(text):  # ???????????????????????????
    cleaned_text = ""
    for ch in text.lower():
        if ch in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz ':
            cleaned_text = cleaned_text + ch
    return cleaned_text


def match(text, example):
    return nltk.edit_distance(text, example) / len(example) < 0.4


x = []
y = []
for intent in BOT_CONFIG['intents']:
    if 'examples' in BOT_CONFIG['intents'][intent]:
        x += BOT_CONFIG['intents'][intent]['examples']
        y += [intent for i in range(len(BOT_CONFIG['intents'][intent]['examples']))]
        # создаем обучающую выборку для ML-модели

vectorizer = CountVectorizer(preprocessor=cleaner, ngram_range=(1, 3), stop_words=['а', 'и'])
# создаем векторайзер - объект для приращения текста в вектора

vectorizer.fit(x)
X_vect = vectorizer.transform(x)
# обучаем векторайзер на нашей выборке

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  # разбиваем выборку на train и test

sgd = SGDClassifier()  # cоздаем модель
sgd.fit(X_vect, y)


def get_intent_by_model(text):  # Функция определяющиая интент текста с помощью ML-модели
    return sgd.predict(vectorizer.transform([text]))[0]


def bot(text):  # функция бота
    intent = get_intent_by_model(text)
    if intent is None:
        intent = get_intent_by_model(text)  # 2. попытаться понять намерение с помощью ML-модели

    return random.choice(BOT_CONFIG['intents'][intent]['responses'])


import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


def start(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Здравствуйте, я-чат бот. \n Я умею только отвечать на вопросы, а точнее давать определение терминам, \n если они есть в моей базе данных, если нет, то вы имеете возможность это исправить. \n Чтобы я дал определение вы должны задать запрос по типу: \n \n Чтобы увидеть чем я могу быть полезен набери help ')


def help_command(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def add_command(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Эта функция на стадии разработки)')


def echo(update: Update, _: CallbackContext) -> None:
    """Echo the user message."""
    text = update.message.text
    print(text)
    reply = bot(text)
    update.message.reply_text(bot(text))


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # updater = Updater("1516415326:AAEyW6qt2KdP9YzSX2-7pJSCX32qzBYqGEQ")
    updater = Updater
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("add", add_command))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


main()


