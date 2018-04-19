# WebCrauler with Captcha resolver (using CNN neural network on pyTorch)
Bot for parse domain with Captcha protection

## Usage
Tested on Python 3.6.

For configuration, set the following varibales to config.py file:
```
mode = "PROD"    # "DEBUG" / "PROD"

#info about captcha
- number_chars_per_image = 5
- path_test_data = "data/test"
- target_width, target_height = (115, 45)
- font_path = "data/train_generator/fonts/DINNextRoundedLTPro-Regular.otf"
- model_path = 'models/model.pkl'
- possible_characters = 'abcdefghijklmnopqrstuvwxyz' + '0123456789'
- char2ind = dict((ch, i) for i, ch in enumerate(possible_characters))
- ind2char = dict((i, ch) for i, ch in enumerate(possible_characters))
- num_classes = len(possible_characters)

- batch_size = 32

- num_steps = 500 # num_steps will be splitted into number of learning rates end for
                    each part will be applied corresponded learning rate

- learning_rates = [0.001, 0.0001, 0.00001]

- card_number = "" # number of card to fill requered field on https://restaurantpass.gift-cards.ru/balance

- link = "https://restaurantpass.gift-cards.ru/balance" # link of parsed domain

- browser = "Chrome"  PhantomJS or Chrome or FireFox

- telegram_token = ""  Telegram bot token
                       create your bot, by sending message to @botfather
                       in return you will receive the token

- chat_id = "" Chat id of Telegram channel
               how to get the chat id:
               https://stackoverflow.com/questions/32423837/telegram-bot-how-to-get-a-group-chat-id


```

Run with:
```
python run.py
nohup python run.py &
```
You can also build and use the docker image:
```
docker build -t <docker-image-url:docker-image-tag> .
docker push <docker-image-url:docker-image-tag>
docker run -d --name web-crawler --restart=always <docker-image-url:docker-image-tag>
```


## Depencencies
- numpy==1.13.3
- pandas==0.20.1
- PyVirtualDisplay==0.2.1
- matplotlib==2.1.1
- opencv_python==3.4.0.12
- scipy==0.19.1
- selenium==3.11.0
- python_telegram_bot==8.0
- schedule==0.5.0
- user_agent==0.1.9
- Pillow==5.1.0
- beautifulsoup4==4.6.0
- seaborn==0.8.1
- telegram==0.0.1
- pytorch

install chromium-browser:
`sudo apt-get install chromium-browser`

Install the dependencies via pip: <br> 
`pip3 install -r requirements.txt`.

Install PyTorch: <br>
`pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl`
`pip3 install torchvisio`


Run tests: <br>
`python -m unittest discover tests/`

## License
Licensed under the [Unlicense](http://unlicense.org/).
Do with it whatever you want.

sudo docker build -t web-crawler . && sudo docker tag web-crawler:latest denkuzin/web-crawler && sudo docker push denkuzin/web-crawler
