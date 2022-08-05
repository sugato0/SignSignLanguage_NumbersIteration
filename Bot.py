

import mediapipe as mp

import cv2
from modelNumbers import loaded_model
from ImageProcessing import GetLmListFromImg


# -*- coding: utf-8 -*-
import cv2
import os


import telebot
from telebot import types
import glob
import time


def neuro_guess(pred_arr,message):
  convert_pred = dict()
  print(pred_arr)
  for _ in enumerate(pred_arr[0]):
    convert_pred[_[0]] = _[1]


  asd= sorted(convert_pred.items(), key=lambda item: item[1], reverse=True)

  bot.send_message(message.chat.id, "this is number " + str(asd[0][0])+"\n might be " + str(asd[1][0]) + " or "+ str(asd[2][0]) )


API_hand_detect = '5362306060:AAEXhy-7D5k6553J0DduwDyqAK-i0sNjep4'
bot = telebot.TeleBot(API_hand_detect)


@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Наша команда")

    markup.add(btn1)
    bot.send_message(message.chat.id,
                     text="<--- Привет!Покажи мне цифру от 0 - 9 одной рукой --->".format(
                         message.from_user), reply_markup=markup)


@bot.message_handler(content_types=["text"])
def photo_prediction(message):
    if message.text == "Наша команда":
        bot.send_message(message.chat.id,"Сахаров Данила Сергеевич "
                                         "\n telegramm:@SugatoKavary "
                                         "\n Попова Александра Алексеевна "
                                         "\n telegramm:@zvukii_paniki\n"
                                         " Марьяна Молчанова-Великая Алексеевна"
                                         " \ntelegramm:@kotoylitka "
                                         "\n https://sugato0.github.io/SignSignLanguage_NumbersIteration/ ")
@bot.message_handler(content_types=["photo"])
def photo_prediction(message):


    # photo downloading

    FilePath = bot.get_file(message.photo[len(message.photo) - 1].file_id).file_path
    downloaded_file = bot.download_file(FilePath)
    src = 'DataImages/' + message.photo[1].file_id
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)





    try:
        #function from libriry
        #getting list [x,y]
        data_our_image = GetLmListFromImg(src)

        print(data_our_image)
        #predict and get list with result
        prediction = loaded_model.predict(data_our_image)
        #func for outputing
        neuro_guess(prediction,message)
    except Exception as e:

        bot.send_message(message.chat.id, e)

    for file in glob.glob("DataImages/*"):
        if file == "DataImages/folder_for_images.txt":
            pass
        os.remove(file)


if __name__ == '__main__': # чтобы код выполнялся только при запуске в виде сценария, а не при импорте модуля
    try:
       bot.polling(none_stop=True) # запуск бота
    except Exception as e:
       print(e) # или import traceback; traceback.print_exc() для печати полной инфы
       time.sleep(15)










