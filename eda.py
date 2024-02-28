import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi layout webpage
st.set_page_config(
    page_title='Emotional Text Analysis',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    # Title and Sub-header
    st.title('Emotional Text Analysis')
    st.subheader('Visualizing Median Sentence Length by Emotional State')

    # Description
    st.write('This webpage is designed to analyze and visualize the median sentence length for various emotional states...')

    
    df = pd.read_csv("train.txt",
                 delimiter=';', header=None, names=['sentence','label']) 
    st.dataframe(df.head())  # Display the first few rows of the dataset
    
    # Assuming the dataframe has columns 'sentence' and 'label'
    df['sentence_length'] = df['sentence'].apply(len)
    
    # Visualizing Label Distribution
    label_distribution = df['label'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_distribution.index, y=label_distribution.values)
    plt.title('Label Distribution in Training Data')
    plt.xlabel('Emotion Label')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    st.pyplot(plt.gcf())
    
    # Plotting Sentence Length Distribution by Label
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='label', y='sentence_length', data=df)
    plt.title('Sentence Length Distribution by Label')
    plt.xlabel('Emotion Label')
    plt.ylabel('Sentence Length')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    st.pyplot(plt.gcf())

    # Display sentence length stats
    sentence_length_stats = df['sentence_length'].describe()
    st.write(sentence_length_stats)

    # Additional insights
    st.write("""
    The visualization of median sentence lengths by emotional state reveals the following insights:
    ...
    - Label Distribution:
The label distribution in the training data shows the following

- proportions of emotional states:

 - Joy is the most common label, comprising approximately 33.5% of the dataset.
 - Sadness follows with about 29.2%.
 - The other labels are distributed as Anger (13.5%), Fear (12.1%), Love (8.15%), and Surprise (3.58%).

Sentence Length Distribution
The average sentence length across all messages is about 97 characters.
There's a wide range of sentence lengths, with a minimum of 7 characters and a maximum of 300 characters.
The distribution has a standard deviation of approximately 56 characters, indicating variability in sentence length.

- Visual Analysis:
 The Label Distribution plot visually confirms the numerical proportions, highlighting the imbalance in the dataset, with "joy" and "sadness" being more prevalent.
 The Sentence Length Distribution by Label plot reveals variability in sentence lengths across different labels. Some emotional states might lead to longer or shorter sentences, but the overall distributions overlap considerably, suggesting that sentence length alone may not be a strong discriminative feature for emotional state classification.
             .""")

if __name__ == '__main__':
    run()
