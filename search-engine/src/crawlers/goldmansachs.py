import requests
from bs4 import BeautifulSoup

from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

import re
import pandas as pd

class GoldmanSachsInsights:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        service = Service(executable_path=ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.implicitly_wait(5)
        
        
    
    def get_link(self):
        self.driver.get('https://www.goldmansachs.com/insights/series/articles/index.html')              
        link_list = [i.get_attribute('href') for i in self.driver\
                     .find_elements(By.CSS_SELECTOR,'[class="title-link-hover"]')]
        
        return link_list
    
    @staticmethod
    def goldmansachs_insight(link):
    
        response = requests.get(link)
        soup = BeautifulSoup(response.text, features='html.parser')
        
        # title
        title = soup.find_all(class_='article-content-page__title article-margin')
        if bool(title) is True:
            title = title[0].text
        else:
            title = 'N/A'
            
        # date
        date = soup.find_all(class_='article-content-page__date')
        if bool(date) is True:
            date = date[0].text
            
        else:
            date = 'N/A'
        
        # tag
        tag = soup.find_all(class_='article-content-page__topics')
        if bool(tag) is True:
            tag = re.sub('\s',' ',tag[0].text).replace('TOPIC:', '').strip()       
            
        else:
            tag = 'N/A'    

        # content
        content = soup.find_all(class_='article-content-page__content')
        if bool(content) is True:
            content = re.sub('\s',' ',content[0].text).strip()        
            
        else:
            content = 'N/A'  
                  
        return {         
            'company': 'Goldman Sachs',
            'topic': 'Insight',
            'tag': tag,
            'section': '',
            'title': title,
            'date': date,
            'link': link,
            'abstract': '',
            'content': content,
            }
    
    def get_all_insights(self):
        links = self.get_link()
        article_dict_list = [self.goldmansachs_insight(link) for link in links]
        return article_dict_list
        
        
    def get_df_insights(self):
        article_dict = self.get_all_insights()
        df = pd.DataFrame(article_dict)
        return df



if __name__ == '__main__':

    obj = GoldmanSachsInsight()
    df = obj.get_df_insights()
    print(df.head())
    

