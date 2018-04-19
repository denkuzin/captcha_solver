mode = "PROD"    # "DEBUG" / "PROD"

number_chars_per_image = 5

path_test_data = "data/test"
target_width, target_height = (115, 45)
font_path = "data/train_generator/fonts/DINNextRoundedLTPro-Regular.otf"
model_path = 'models/model.pkl'

possible_characters = 'abcdefghijklmnopqrstuvwxyz' + '0123456789'
char2ind = dict((ch, i) for i, ch in enumerate(possible_characters))
ind2char = dict((i, ch) for i, ch in enumerate(possible_characters))
num_classes = len(possible_characters)

# Train hyper parameters
batch_size = 32
num_steps = 10000
learning_rates = [0.001, 0.0001, 0.00001]     # num_steps will be splitted into number of learning rates end for
                                              # each part will be applied corresponded learning rate

# info to fill in the site
card_number = 123456789
link = "https://restaurantpass.gift-cards.ru/balance" # site link
browser = "Chrome"  # PhantomJS or Chrome or FireFox

telegram_token = ""
chat_id = ""
