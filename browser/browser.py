from selenium import webdriver
import time
import random
from pyvirtualdisplay import Display
import logging
import os

logger = logging.getLogger()

def wait(mean):
    stddev = mean*0.2
    t = random.normalvariate(mean, stddev)
    mintime = random.uniform(0.05,0.15)
    t = max(t, mintime)
    time.sleep(t)


def init_phantomjs_driver(user_agent=None):
    if not user_agent:
        user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "\
                     "(KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36"
    headers = {
              'Accept': 'text/html,application/xhtml+xml,application'\
                       '/xml;q=0.9,image/webp,*/*;q=0.8',
              'Accept-Language': 'en-US,en;q=0.8,ru;q=0.6',
              'Cache-Control': 'max-age=0',
              'Connection': 'keep-alive'
               }
    for key, value in headers.items():
        big_key = 'phantomjs.page.customHeaders.{}'.format(key)
        webdriver.DesiredCapabilities.PHANTOMJS[big_key] = value
    webdriver.DesiredCapabilities.PHANTOMJS['phantomjs.page.settings.userAgent'] = user_agent
    service_args = ['--ignore-ssl-errors=true', '--ssl-protocol=ANY']
    driver_start = webdriver.PhantomJS(service_args=service_args)
    driver_start.set_window_size(1855, 1056)
    return driver_start


def start_browser(browser="Chrome", user_agent=None, isDebug='PROD'):
    logger.info("launch browser {}".format(browser))

    if browser == "Chrome":
        options = webdriver.ChromeOptions()
        options.add_argument("--window-size=1855x1056")
        if isDebug == 'DEBUG':
            driver_start = webdriver.Chrome(chrome_options=options)
        else:
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            chrome_driver_path = os.path.join(os.getcwd(), "browser/chromedriver")
            driver_start = webdriver.Chrome(chrome_options=options,
                                        executable_path=chrome_driver_path)
    elif browser == "PhantomJS":
        driver_start = init_phantomjs_driver(user_agent=user_agent)
    elif browser == "FireFox":
        display = Display(visible=0, size=(1855, 1056))
        display.start()
        driver_start = webdriver.Firefox()
    else:
        raise RuntimeError("please, specify correct browser")
    wait(5)

    driver_start.set_page_load_timeout(900)
    driver_start.set_script_timeout(900)
    logger.info("browser {} is launched".format(browser))
    return driver_start
