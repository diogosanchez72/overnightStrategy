import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def init_driver():
    options = ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280x800")
    options.add_argument("--no-sandbox")
    
    service = Service(ChromeDriverManager().install())

    return webdriver.Chrome(service=service)


def fetch_options(driver, target_dates):
    fetched_options = {}
    element = driver.find_element("css selector", "#vencimento")

    for date in target_dates:
        element.send_keys(date)
        page_source = BeautifulSoup(driver.page_source, "html.parser")
        rows = page_source.select("#miniGrid tbody tr")
        fetched_options[date] = [(cell[0].text.strip(), cell[1].text.strip(), cell[2].text.strip())
                                 for row in rows if (cell := row.select("td"))]

    return fetched_options


def fetch_call_put(fetched_options):
    calls, puts = {}, {}

    for data in fetched_options.values():
        for _, call, put in data:
            call_url = f"http://opcoes.net.br/{call}"
            call_dataframe = pd.read_html(call_url)[0]
            if not call_dataframe.empty:
                calls[call] = call_dataframe.to_dict()

            put_url = f"http://opcoes.net.br/{put}"
            put_dataframe = pd.read_html(put_url)[0]
            if not put_dataframe.empty:
                puts[put] = put_dataframe.to_dict()

    return calls, puts


def get_options():
    driver = init_driver()
    driver.get("http://opcoes.net.br/JBSST177")
    page_source = BeautifulSoup(driver.page_source, "html.parser")
    target_dates = [option.text for option in page_source.select("#vencimento option")]

    fetched_options = fetch_options(driver, target_dates)
    driver.quit()

    calls, puts = fetch_call_put(fetched_options)

    return {"options": fetched_options, "calls": calls, "puts": puts}
