import streamlit as st
kpmg_col1, kpmg_col2 = st.columns([1, 9])
with kpmg_col2:
    st.markdown("**| KPMG Innovation Project**")
with kpmg_col1:
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/KPMG_logo.svg/1280px-KPMG_logo.svg.png',width=70)

st.header('Financial Insights Text Analysis')
st.image("https://images.dowjones.com/wp-content/uploads/sites/193/2021/07/12163514/Newswires_Run_Analytics_blog_1440x420.png")



st.markdown("### Project Overview")
st.write("The goal of this project was to gather financial insights and articles from four investment management firms (PIMCO, Capital Group, Vanguard, and JPMorgan) and store them for use in a natural language processing (NLP) app. The app would allow users to interact with the insights and articles in a variety of ways, such as personalized recommendations, topic search, or sentiment analysis.")

st.markdown("### Data Gathering")
st.markdown(""" To gather the relevant insights and articles from each firm,
         we first identified the web pages that contained the information we were interested in.
         For each firm, we crawled a set of pages that included market commentaries,
         research reports, and other types of financial analysis.
         We used a combination of web scraping tools and custom scripts to extract
         the text and metadata from the pages, and stored the results in a database for further processing.""")



col1, col2 = st.columns(2)

with col1:
    # st.checkbox("Disable selectbox widget", key="disabled")
    st.image('https://www.scopeexplorer.com/files/download/?name=funds.LogoFile/bytes/filename/mimetype/CG_Horizontal_w_TM_rgb.png', width=130)
    st.info('This is a box with a blue background color')
    st.markdown("<hr>", unsafe_allow_html=True)
    st.image('https://cdn.freebiesupply.com/logos/large/2x/pimco-funds-logo-png-transparent.png',width=70)
    st.info('This is a box with a blue background color')
with col2:
    st.image('https://logos-world.net/wp-content/uploads/2021/03/Vanguard-Emblem.png',width=100)
    st.info('This is a box with a blue background color')
    st.markdown("<hr>", unsafe_allow_html=True)
    st.image('https://logos-world.net/wp-content/uploads/2021/02/JP-Morgan-Chase-Emblem.png',width=100)
    st.info('This is a box with a blue background color')



st.markdown("### Data Cleaning")
st.markdown(""" After gathering the data, we pre-processed it to remove any irrelevant
            or redundant information. This included removing boilerplate text
            (such as copyright notices or legal disclaimers), and filtering out pages that
            did not contain any substantive financial analysis.
            We also performed some basic text normalization tasks,
            such as removing stop words and stemming the remaining
            words to reduce the dimensionality of the data.""")


st.markdown("### Data Storage")
st.markdown(""" After gathering the data, we pre-processed it to remove any irrelevant
            or redundant information. This included removing boilerplate text
            (such as copyright notices or legal disclaimers), and filtering out pages that
            did not contain any substantive financial analysis.
            We also performed some basic text normalization tasks,
            such as removing stop words and stemming the remaining
            words to reduce the dimensionality of the data.""")


st.markdown("### NLP Analysis")
st.markdown(""" With the data stored in a database, we were able to apply a range
            of NLP techniques to extract insights and trends from the articles.
            This included text classification, sentiment analysis, and topic modeling.
            We used pre-built libraries and tools for most of the NLP tasks,
            such as scikit-learn for classification and gensim for topic modeling.
            We also experimented with different techniques and parameter settings
            to optimize the performance of the NLP models.""")

st.markdown("### App Development")
st.markdown(""" The final stage of the project was to build the NLP app itself.
            We chose to use the Streamlit framework, which allowed us to quickly
            develop a user-friendly and interactive app that could be accessed through a web browser.
            The app included features such as personalized recommendations based on
            user preferences, sentiment analysis of individual articles, and topic search.
            We also included visualizations to help users better understand the data,
            such as word clouds and bar charts.""")

