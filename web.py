import os
import time
import random
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from shutil import move, rmtree
from PIL import Image

# Configuration
VEGETABLE_CLASSES = ['eggplant', 'broccoli', 'carrot', 'cucumber', 'potato', 'radish', 'tomato']
DOWNLOAD_PATH = "./temp_vegetables_dataset" 
FINAL_PATH = "./vegetables_dataset"  
TOTAL_IMAGES_PER_CLASS = 100 
CHROME_DRIVER_PATH = "C:\\Users\\o\\.wdm\\drivers\\chromedriver\\win64\\130.0.6723.69\\chromedriver-win32\\chromedriver.exe"

# Configuration Selenium
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
service = Service(CHROME_DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

# Créer des répertoires nécessaires
def create_directories(base_path, splits, classes):
    for split in splits:
        for vegetable in classes:
            path = os.path.join(base_path, split, vegetable)
            os.makedirs(path, exist_ok=True)

def download_image(url, filepath):
    """Télécharger une image depuis une URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Image téléchargée : {filepath}")
    except Exception as e:
        print(f"Erreur lors du téléchargement de {url} : {e}")

def is_valid_image(filepath):
    """Vérifier si une image est valide."""
    try:
        with Image.open(filepath) as img:
            img.verify()  # Vérifie si l'image est valide
        return True
    except Exception as e:
        print(f"Image corrompue : {filepath} - {e}")
        return False

def scroll_to_load_more(driver, scroll_count=5):
    """Scroller pour charger plus d'images."""
    for _ in range(scroll_count):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  

def scrape_images():
    """Scraper des images pour chaque classe et les stocker dans un répertoire temporaire."""
    for vegetable in VEGETABLE_CLASSES:
        print(f"Recherche pour la classe : {vegetable}")
        keywords = [
            f"{vegetable} vegetable",
            f"raw {vegetable}",
            f"{vegetable} plant",
            f"fresh {vegetable} vegetable"
        ]
        image_urls = []
        for keyword in keywords:
            driver.get("https://www.bing.com/images/")
            search_box = driver.find_element(By.NAME, "q")
            search_box.clear()
            search_box.send_keys(keyword)
            search_box.send_keys(Keys.RETURN)
            scroll_to_load_more(driver)

            image_elements = driver.find_elements(By.CSS_SELECTOR, "img.mimg")
            for img in image_elements:
                try:
                    url = img.get_attribute("src")
                    if url and url.startswith("http") and url not in image_urls:
                        image_urls.append(url)
                    if len(image_urls) >= TOTAL_IMAGES_PER_CLASS:
                        break
                except Exception as e:
                    print(f"Erreur lors de la récupération de l'URL : {e}")
            if len(image_urls) >= TOTAL_IMAGES_PER_CLASS:
                break

        print(f"{len(image_urls)} URLs récupérées pour {vegetable}.")
        
        temp_class_path = os.path.join(DOWNLOAD_PATH, vegetable)
        os.makedirs(temp_class_path, exist_ok=True)

        for idx, url in enumerate(image_urls):
            filepath = os.path.join(temp_class_path, f"{vegetable}_{idx}.jpg")
            download_image(url, filepath)
            if not is_valid_image(filepath):
                os.remove(filepath)

        if len(os.listdir(temp_class_path)) < TOTAL_IMAGES_PER_CLASS:
            print(f"Nombre insuffisant d'images pour {vegetable}. Réessayez manuellement si nécessaire.")

def distribute_images():
    """Répartir les images téléchargées dans train, validation et test."""
    for vegetable in VEGETABLE_CLASSES:
        temp_class_path = os.path.join(DOWNLOAD_PATH, vegetable)
        images = os.listdir(temp_class_path)
        random.shuffle(images)

        train_images = images[:70]
        test_images = images[70:85]
        val_images = images[85:100]

        for split, image_set in zip(['train', 'test', 'validation'], [train_images, test_images, val_images]):
            for image in image_set:
                src_path = os.path.join(temp_class_path, image)
                dest_path = os.path.join(FINAL_PATH, split, vegetable, image)
                move(src_path, dest_path)

    print("Images réparties avec succès !")

def main():
    create_directories(FINAL_PATH, ['train', 'test', 'validation'], VEGETABLE_CLASSES)
    scrape_images()
    distribute_images()

    # Nettoyage du répertoire temporaire
    if os.path.exists(DOWNLOAD_PATH):
        rmtree(DOWNLOAD_PATH)
    
    driver.quit()
    print("Téléchargement terminé et dataset prêt !")

if __name__ == "__main__":
    main()
