from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path

WORDCLOUD_IMAGES_PATH = Path('.').absolute() / 'src/wordcloud_images/'

def create_wordcloud(text, by_name):
    wcd = WordCloud().generate(text)
    wcd.to_file(f'{WORDCLOUD_IMAGES_PATH}/{by_name}.png')
    

if __name__ == '__main__':
    create_wordcloud('this is test ', 'example')
    # print(WORDCLOUD_IMAGES_PATH)
    

