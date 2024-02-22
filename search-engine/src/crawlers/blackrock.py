import pandas as pd
import requests
from bs4 import BeautifulSoup
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm


class BlackRockInsights:

    def __init__(self, all_insight_link):
        self.all_insight_link = all_insight_link
        self.driver = webdriver.Chrome(ChromeDriverManager().install())

    @staticmethod
    def extract_content_with_p(link: str) -> str:
        # send requests and get response
        response = requests.get(link)

        # get content if response is ok
        if response.ok is True:
            soup = BeautifulSoup(response.text)

            content = " ".join([i.text for i in soup.find_all('p')[5:]])
            return content
        else:
            raise Exception(f'Response is: {response}')

    @staticmethod
    def extract_content_with_class(link: str) -> str:
        response = requests.get(link)
        if response.ok is True:
            # send requests
            response = requests.get(link)
            soup = BeautifulSoup(response.text)

            content = ' '.join([i.text for i in soup.find_all(class_='para-content')])
            return content

        else:
            raise Exception(f'Response is: {response}')

    def get_article_insight(self):

        self.driver.get(self.all_insight_link)
        sleep(10)
        elem = self.driver.find_element(By.ID, 'onetrust-reject-all-handler')
        elem.click()
        button = self.driver.find_element(By.XPATH, '//*[@id="c1561158932259"]/div/div[2]/div/div/div[3]/button')
        for i in range(8):
            self.driver.execute_script("arguments[0].click();", button)

        title_list = [i.text for i in self.driver.find_elements(By.XPATH, '//h2')]
        date_list = [i.text for i in self.driver.find_elements(By.XPATH,
                                                               '//a//div//div//span[(((count(preceding-sibling::*) + '
                                                               '1) = 1) and parent::*)]')]
        author_list = [i.text for i in self.driver.find_elements(By.CSS_SELECTOR, 'span:nth-child(4)')]
        abstract_list = [i.text for i in self.driver.find_elements(By.XPATH,
                                                                   '//*[contains(concat( " ", @class, " " ), concat( '
                                                                   '" ", "description", " " ))]')]
        link_list = [i.get_attribute('href') for i in self.driver.find_elements(By.CSS_SELECTOR, '.skip-animation')]
        tag_list = [i.text for i in self.driver.find_elements(By.XPATH, '//*[@class="category-text"]')]

        blackrock_insight_list = []
        for title, date, author, abstract, link, tag in tqdm(
                zip(title_list, date_list, author_list, abstract_list, link_list, tag_list)):
            blackrock_insight_dict = {
                'company': 'BlackRock',
                'topic': 'Insight',
                'tag': tag,
                'section': '',
                'title': title,
                'date': date,
                'link': link,
                'abstract': abstract,
                'content': self.extract_content_with_class(link),
                
            }

            blackrock_insight_list.append(blackrock_insight_dict)

        return blackrock_insight_list


if __name__ == '__main__':
    obj = BlackRockInsights('https://www.blackrock.com/us/individual/insights')
    list_of_article = obj.get_article_insight()
    blackrock_df = pd.DataFrame(list_of_article)
    print(blackrock_df.head())