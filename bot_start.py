import argparse
import logging
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from numba import njit, prange
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import psycopg2

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


@njit(parallel=True, fastmath=True)
def l2_distance(index, vector):
    scores = np.zeros((index.shape[0],))
    for i in prange(index.shape[0]):
        scores[i] = np.linalg.norm(index[i] - vector)
    return scores


class BotHandler:
    def __init__(self, token, index_dir):
        self.token = token
        self.index_dir = index_dir
        self.model = None
        self.index = None
        self.product_ids = None

    def start(self):
        self._load_model()
        self._load_index()
        self._create_bot()

    def _load_model(self):
        self.model = self._build_model()

    def _load_index(self):
        data = np.load(os.path.join(self.index_dir, 'index.npz'))
        self.index = data['vectors'].astype(np.float32)
        self.product_ids = data['ids']

    @staticmethod
    def _build_model():
        m = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2", trainable=False)
        ])
        m.build((None, 224, 224, 3))
        return m

    @staticmethod
    def _load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, expand_animations=False, channels=3)
        image = tf.image.resize(image, (224, 224)) / 255
        return tf.expand_dims(image, axis=0)

    def _process_image(self, model, index, product_ids, image):
        vector = model.predict(image, verbose=0)
        scores = l2_distance(index, vector[0])
        n_closest = np.argsort(scores)[:3]
        closest_product_ids = product_ids[n_closest.tolist()]

        conn = psycopg2.connect(
            database="pricehub",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()

        cursor.execute("SELECT id, title, price, url, photo FROM products WHERE id = ANY(%s)",
                       (closest_product_ids.tolist(),))

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return rows

    def _create_bot(self):
        updater = Updater(self.token, use_context=True)
        dispatcher = updater.dispatcher

        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

        start_handler = CommandHandler('start', self._start)
        photo_handler = MessageHandler(Filters.photo, self._handle_photo)

        dispatcher.add_handler(start_handler)
        dispatcher.add_handler(photo_handler)

        updater.start_polling()

    def _start(self, update: Update, context: CallbackContext):
        context.bot.send_message(chat_id=update.effective_chat.id, text='Добро пожаловать в поисковый бот !')

    def _handle_photo(self, update: Update, context: CallbackContext):
        file_id = update.message.photo[-1].file_id
        file = context.bot.get_file(file_id)
        file_path = file.file_path
        image_path = os.path.join('data', os.path.basename(file_path))
        file.download(image_path)

        loading_message = context.bot.send_message(chat_id=update.effective_chat.id, text='Поиск по базе...')
        time.sleep(2)

        image = self._load_image(image_path)
        products = self._process_image(self.model, self.index, self.product_ids, image)
        for product in products:
            product_id, title, price, url, photo_data = product

            context.bot.send_photo(chat_id=update.effective_chat.id, photo=photo_data)

            response = f"{title}\n<b>{price}</b>\n{url}\n\n"
            update.message.reply_html(response)

        context.bot.delete_message(chat_id=update.effective_chat.id, message_id=loading_message.message_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, help='Telegram Bot token',
                        default="6001288764:AAHBfbPIcZUBWDNmFVDb9pBsn8moticRkrg")
    parser.add_argument('--index', type=str, help='Path to the index directory', default="data")
    args = parser.parse_args()

    bot_handler = BotHandler(args.token, args.index)
    bot_handler.start()
