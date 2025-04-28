from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

# Setup driver
url = input("Masukkan URL data BPS: ")
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
driver.get(url)

WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.max-sm\\:hidden button"))
)

buttons = driver.find_elements(By.CSS_SELECTOR, "div.max-sm\\:hidden button")
all_data = []

for btn in buttons:
    tahun = btn.text
    print(f"Scraping data untuk periode: {tahun}")
    
    driver.execute_script("arguments[0].click();", btn)
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("table", {"id": "data-table"})

    # Pastikan tabel ditemukan
    if table:
        rows = table.find("tbody").find_all("tr")
        for row in rows:
            cols = [ele.text.strip() for ele in row.find_all("td")]
            if cols:
                cols.insert(0, tahun)
                all_data.append(cols)
    else:
        print(f"Tabel tidak ditemukan di periode {tahun}")

driver.quit()

# Simpan ke CSV
df = pd.DataFrame(all_data)
df.to_csv("data_ahh_periode.csv", index=False)
print("Scraping selesai dan data disimpan.")