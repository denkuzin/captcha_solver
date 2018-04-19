import config
from models import models
import train
import torch
import os.path
import numpy as np
from preprocessing import process_picture
from browser import browser
import user_agent
import urllib.request
from PIL import Image
from bs4 import BeautifulSoup
import telegram
import datetime
import logging
import schedule
import time

isDebug = config.mode

if isDebug == "DEBUG":
    LOGGIN_LEVEL = logging.DEBUG
else:
    LOGGIN_LEVEL = logging.INFO

logging.basicConfig(
    format='%(asctime)s\t%(levelname)s\t(%(name)s)\t%(message)s',
    level=LOGGIN_LEVEL)
logger = logging.getLogger(__name__)
fh = logging.FileHandler(r'log.txt', 'w', 'utf-8')
fh.setLevel(LOGGIN_LEVEL)
formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t(%(name)s)\t%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("mode is {}".format(isDebug))


class WebCrawler:
    def __init__(self):
        self.model = self.load_model()
        self.ua = user_agent.generate_user_agent()
        logger.debug("self.ua = {}".format(self.ua))

    @staticmethod
    def load_model():
        # check do we have trained model
        if not os.path.isfile(config.model_path):
            logger.info("start to train the model")
            train.train()
        else:
            logger.info("using pretrained model '{}'".format(config.model_path))
        model = models.CNN(num_classes=config.num_classes)
        model.load_state_dict(torch.load(config.model_path))
        logger.info("the model is uploaded")
        return model

    def get_image(self):
        # get captcha image link:
        css_path = ('#checkBalanceForm > div.b-check-balance-'
                    'form_field-block.b-check-balance-form_field-'
                    'block__captcha > div.b-check-balance-form_field-wrapper > img')
        logger.debug("css_path = {}".format(css_path))
        el = self.driver.find_element_by_css_selector(css_path)
        src = el.get_attribute('src')
        logger.debug("src = {}".format(src))
        browser.wait(1)

        # upload image to PC
        urllib.request.version = self.ua
        file_name, _ = urllib.request.urlretrieve(src)
        image = Image.open(file_name)
        array = np.asarray(image)
        logger.info("image is uploaded to '{}'".format(file_name))
        logger.info("image shape is {}".format(array.shape))
        return array

    def predict_image(self,image):
        """
        :param image: numpy array
        :return: recognized word
        """
        small_images = process_picture(image)
        result = []
        for img in small_images:
            img = np.expand_dims(img, axis=0)
            tenzor = torch.autograd.Variable(torch.from_numpy(img))
            outputs = self.model.forward(tenzor)
            _, predicted = torch.max(outputs.data, 1)
            char = config.ind2char[predicted.numpy()[0]]
            result.append(char)
        word = "".join(result)
        logger.info("recognized characters are '{}'".format(word))
        return word

    def fill_fields(self):
        # card number
        el = self.driver.find_element_by_css_selector('#ean')
        el.send_keys(config.card_number)
        logger.debug("{} was sent to the window".format(config.card_number))
        browser.wait(2)

        # captcha data
        el = self.driver.find_element_by_css_selector('#captcha')
        el.send_keys(self.recognized_word)
        logger.debug("{} was sent to the window".format(self.recognized_word))
        browser.wait(1)

        # click 'check balance button'
        css_path ='#checkBalanceForm > div.b-check-balance-form_footer > button > span'
        el = self.driver.find_element_by_css_selector(css_path)
        el.click()
        logger.info("the fields are filled")
        browser.wait(1)

    def parse_html(self):
        html_source = self.driver.page_source
        soup = BeautifulSoup(html_source, 'lxml')
        info_balance = soup.find(class_="b-customer-info_balance_value ng-binding")
        balance = info_balance.get_text()
        logger.info("parsed balance is {}".format(balance))
        return balance

    @staticmethod
    def send2telegramm(message):
        bot = telegram.Bot(token=config.telegram_token)
        bot.send_message(chat_id=config.chat_id, text=message)
        logger.info("the message '{}' was sent to telegram".format(message))

    def run(self):
        self.driver = browser.start_browser(browser=config.browser, user_agent=self.ua, isDebug=isDebug)
        logger.debug("start to get {}".format(config.link))
        self.driver.get(config.link)
        logger.debug("the link {} was got".format(config.link))
        image = self.get_image()
        self.recognized_word = self.predict_image(image)  # here can be error (incorrectly recognized image)
        self.fill_fields()
        browser.wait(20)
        balance = self.parse_html()
        date = datetime.date.today().strftime('%B,%d')
        message = "{} ({})".format(balance, date)
        self.send2telegramm(message)


def job():
    try:
        WC = WebCrawler()
        WC.run()
    except Exception as ex:
        template = "main: an exception of type {0} occurred. Arguments: {1!r}"
        message = template.format(type(ex).__name__, str(ex.args))
        logger.error(message)
        try:
            WC = WebCrawler()
            WC.send2telegramm("error")
            WC.run() # repeat
        except:
            pass


if __name__ == '__main__':
    job()
    schedule.every().day.at("05:00").do(job)
    schedule.every().day.at("12:00").do(job)
    schedule.every().day.at("20:00").do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)
