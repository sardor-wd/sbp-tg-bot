import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from io import BytesIO
from telegram import Update
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import psycopg2

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def load_features(file_path):
    data = np.load(file_path)
    vectors = data['vectors']
    product_ids = data['ids']
    return vectors, product_ids


def cosine_search(vectorizer, index, vector):
    v_norm = np.linalg.norm(vector)
    vector = tf.expand_dims(vector, axis=0)
    vector = vectorizer.predict(vector)[0]
    scores = np.dot(index, vector) / (np.linalg.norm(index, axis=1) * v_norm)
    return scores


def start_command(update: Update, context: CallbackContext):
    update.message.reply_text('Bot started. Send an image to find similar products.')


def image_received(update: Update, context: CallbackContext):
    photo = update.message.photo[-1]  # Get the highest resolution photo
    image_file = context.bot.get_file(photo.file_id)
    image_data = image_file.download_as_bytearray()

    image = Image.open(BytesIO(image_data))
    processed_image = preprocess_image(image)

    vectorizer, vectors, product_ids, connection = context.bot_data['features']

    scores = cosine_search(vectorizer, vectors, processed_image)

    n_closest = np.argsort(scores)[::-1][:5]
    closest_product_ids = product_ids[n_closest]

    product_query = f"SELECT id, title, price, url, photo FROM products WHERE id IN ({', '.join(map(str, closest_product_ids))})"
    cursor = connection.cursor()
    cursor.execute(product_query)
    products = cursor.fetchall()
    cursor.close()

    for product in products:
        product_id = product[0]
        title = product[1]
        price = product[2]
        url = product[3]
        photo_data = product[4]

        context.bot.send_photo(chat_id=update.effective_chat.id, photo=photo_data)

        response = f"{title}\n<b>{price}</b>\n{url}\n\n"
        update.message.reply_html(response)

    connection.commit()
    connection.close()


def main():
    token = '6001288764:AAHBfbPIcZUBWDNmFVDb9pBsn8moticRkrg'
    file_path = 'vectorized_features.npz'

    vectorizer = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5", trainable=False)
    ])
    vectorizer.build([None, 256, 256, 3])

    vectors, product_ids = load_features(file_path)

    connection_params = {
        'host': '127.0.0.1',
        'port': '5432',
        'database': 'pricehub',
        'user': 'postgres',
        'password': 'postgres'
    }

    connection = psycopg2.connect(**connection_params)

    updater = Updater(token, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.bot_data['features'] = (vectorizer, vectors, product_ids, connection)

    dispatcher.add_handler(CommandHandler("start", start_command))
    dispatcher.add_handler(MessageHandler(Filters.photo, image_received))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
