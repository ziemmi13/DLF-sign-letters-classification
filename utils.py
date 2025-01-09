import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def download_asl_images():
    driver = webdriver.Chrome()
    base_url = "https://spreadthesign.com/en.us/alphabet/21/"

    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        try:
            driver.get(base_url)
            time.sleep(3)

            # Find letter element by XPath
            letter_elem = driver.find_element(By.XPATH, f"//div[@class='letter' and text()='{letter}']")
            letter_elem.click()

            time.sleep(3)

            img = driver.find_element(By.XPATH, "//div[contains(@class, 'sign-video')]//img")
            img_url = img.get_attribute('src')

            if img_url:
                response = requests.get(img_url)
                if response.status_code == 200:
                    filename = os.path.join('asl_dataset', f'letter_{letter}', f'{letter}.jpg')
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {letter}")

        except Exception as e:
            print(f"Error with letter {letter}: {e}")

    driver.quit()


if __name__ == '__main__':
    download_asl_images()