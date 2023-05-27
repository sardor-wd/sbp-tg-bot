import os
import urllib.request
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_product_photos():
    connection_params = {
        'host': '127.0.0.1',
        'port': '5432',
        'database': 'pricehub',
        'user': 'postgres',
        'password': 'postgres'
    }

    conn = None
    cursor = None

    try:
        conn = psycopg2.connect(**connection_params)
        cursor = conn.cursor()

        cursor.execute("SELECT id, photo FROM products WHERE photo IS NOT NULL")
        products = cursor.fetchall()

        os.makedirs('photos', exist_ok=True)
        max_workers = 10

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for product in products:
                product_id = product[0]
                image_url = product[1]

                filename = f'product_{product_id}.png'
                save_path = os.path.join('photos', filename)

                future = executor.submit(download_image, image_url, save_path, filename)
                futures.append(future)

            for future in futures:
                future.result()

        logging.info('All photos downloaded successfully.')

    except psycopg2.Error as e:
        logging.error(f'Error connecting to PostgreSQL: {str(e)}')

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def download_image(image_url, save_path, filename):
    try:
        response = urllib.request.urlopen(image_url)
        expected_size = int(response.headers['Content-Length'])
        data = response.read()

        if len(data) != expected_size:
            raise ValueError('Incomplete download')

        with open(save_path, 'wb') as file:
            file.write(data)

        logging.info(f'Saved photo: {filename}')

    except urllib.error.URLError as e:
        if isinstance(e.reason, urllib.error.HTTPError) and e.reason.code == 403:
            logging.warning(f'Skipped photo: {filename} (Forbidden)')
        else:
            logging.error(f'Error downloading photo: {filename}')

    except (ValueError, Exception) as e:
        logging.error(f'Skipped photo: {filename} (Incomplete download)')
        logging.error(f'Error: {str(e)}')


def main():
    start_time = time.time()
    download_product_photos()
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f'Total execution time: {execution_time} seconds')

if __name__ == '__main__':
    main()
