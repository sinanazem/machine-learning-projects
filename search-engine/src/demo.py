import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud
from wordcloud_article import create_wordcloud, WORDCLOUD_IMAGES_PATH
from PIL import Image 

from search import most_similar_article
from text_preprocessing import preprocessing


st.header('Insight Recommendation')  # header for webapp
st.image('https://builtin.com/sites/www.builtin.com/files/styles/og/public/recommendation-system-machine-learning_1.jpg')
query = st.text_input(':mag: **Ask your query:**')
df = most_similar_article(query=query)
df['content'] = df['content'].apply(lambda x: preprocessing(x))

# st.dataframe(df)


for i in list(df.index):

    st.markdown(f'### {df.loc[i, "company"]}')
    st.info(f'**{df.loc[i, "title"]}**')

    # word_image = create_wordcloud(f'{df.loc[i, "content"]}', str(i))
    # image = Image.open(f'{WORDCLOUD_IMAGES_PATH}/{i}.png')
    # st.image(image)

    wc = WordCloud().generate(f'{df.loc[i, "content"]}')
    fig, ax = plt.subplots()  
    ax.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)
        

    st.write(f'{df.loc[i, "content"]}')
    st.markdown(f'**Article Link:** {df.loc[i, "link"]}')

    
