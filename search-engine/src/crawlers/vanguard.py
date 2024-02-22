import re
from time import sleep

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager




class VanguardInsights:

    def __init__(self, all_insight_link):
        self.all_insight_link = all_insight_link
        self.driver = webdriver.Chrome(ChromeDriverManager().install())

    @staticmethod
    def extract_content_with_id(link):
        response = requests.get(link)
        soup = BeautifulSoup(response.text, features="html.parser")
        content = ' '.join([i.text for i in soup.find_all("div", {"id": "iw_placeholder1585759247342"})]).strip()
        return content

    def get_article_insight(self):

        self.driver.get(self.all_insight_link)
        sleep(10)

        all_pages = []

        for i in range(18):

            title_tags = self.driver.find_elements(By.XPATH, '//*[(@id = "insights-archive-section")]//h3')
            title_contents = self.driver.find_elements(By.XPATH,
                                                       '//*[(@id = "insights-archive-section")]//*[contains(concat( " '
                                                       '", @class, " " ), concat( " ", "p2", " " ))]')
            date_perspectives = self.driver.find_elements(By.XPATH,
                                                          '//*[contains(concat( " ", @class, " " ), concat( " ", '
                                                          '"eyebrow", " " ))]')
            perspectives = self.driver.find_elements(By.XPATH,
                                                     '//*[contains(concat( " ", @class, " " ), concat( " ", "type", '
                                                     '" " ))]')
            # dates = browser.find_elements(By.CSS_SELECTOR, '.content-eyebrow .eyebrow')
            tags = self.driver.find_elements(By.CSS_SELECTOR, "[id='insights-archive'] [class='tags']")
            tags = [i.find_elements(By.CSS_SELECTOR, "[id='insights-archive'] [class='tags'] [class='pill tag']") for i
                    in tags]
            links = self.driver.find_elements(By.CSS_SELECTOR, '.detail-link')

            title_tag = [i.text for i in title_tags]
            title_content = [i.text for i in title_contents]
            date_perspective = [i.text for i in date_perspectives]
            date = [''.join(re.findall(r'[A-Z]+\s\d+,\s\d+', i)) for i in date_perspective[1:]]
            perspective = [i.text for i in perspectives]
            # date = [i.text for i in dates]
            tag = [[j.text for j in i] for i in tags]
            links_list = [link.get_attribute('href') for link in links]
            
            list_of_dict = []
            for tg, tc, d, p, t, l in zip(title_tag, title_content, date, perspective, tag, links_list):
                economy_market_dict = {
                    'company': 'Vanguard',
                    'topic': 'Insight',
                    'tag': ','.join(t),
                    'section': '',
                    'title': tg,
                    'date': d,
                    'link': l,
                    'abstract': tc,
                    'content': self.extract_content_with_id(l),
                }
                list_of_dict.append(economy_market_dict)

            all_pages.append(list_of_dict)
            next_page = self.driver.find_element(By.CSS_SELECTOR,
                                                 '#insights-archive-section > div.pagination > div.pagination__arrows '
                                                 '> span.icon.icon-right')
            next_page.click()

        self.driver.quit()
        vanguard_insight_list_ = []
        for p_8 in all_pages:
            for p in p_8:
                vanguard_insight_list_.append(p)

        return vanguard_insight_list_


if __name__ == "__main__":
    from utils import DataBaseClass   
    # Vanguard
    vanguard_insight_obj = VanguardInsights('https://advisors.vanguard.com/insights/all')
    vanguard_insight_list = vanguard_insight_obj.get_article_insight()
    vanguard_df = pd.DataFrame(vanguard_insight_list)

    # Store data in db

    obj_data_base_class = DataBaseClass('guest', 'Aa12345', 'localhost', 5432, 'insights_db')
    obj_data_base_class.store_data(vanguard_df, sql_table_name='insights_data')