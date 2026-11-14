
# Google Play Store Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

apps = pd.read_csv('play_store_apps_dataset.csv')
reviews = pd.read_csv('play_store_reviews_dataset.csv')
print(apps.shape, reviews.shape)
apps['Installs_num'] = apps['Installs'].str.replace(',','',regex=False).str.replace('+','',regex=False).astype(float)
apps['Size_M'] = apps['Size'].str.extract(r'(\d+)').astype(float)
apps['Rating'] = pd.to_numeric(apps['Rating'],errors='coerce')
apps = apps.drop_duplicates(subset='App')

apps['Category'].value_counts().plot(kind='bar', figsize=(12,5), title='Apps by Category')
plt.xticks(rotation=45); plt.tight_layout(); plt.show()
apps['Rating'].hist(bins=20); plt.title('Rating Distribution'); plt.show()
apps.groupby('Category')['Rating'].mean().sort_values().plot(kind='barh', title='Average Rating by Category'); plt.show()
sns.scatterplot(data=apps,x='Size_M',y='Installs_num'); plt.title('App Size vs Installs'); plt.show()
apps['Type'].value_counts().plot(kind='pie',autopct='%1.1f%%'); plt.title('Free vs Paid Apps'); plt.ylabel(''); plt.show()
paid=apps[apps['Type']=='Paid']
paid['Price'].hist(bins=10); plt.title('Paid App Price Distribution'); plt.show()
apps['Estimated_Revenue'] = apps['Price'] * apps['Installs_num']
apps.groupby('Category')['Estimated_Revenue'].sum().sort_values().plot(kind='barh',title='Estimated Paid App Revenue by Category'); plt.show()

# Dataset includes a sentiment label for educational purposes; also calculate TextBlob polarity.
reviews['Polarity'] = reviews['Review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
reviews['Polarity_Sentiment'] = reviews['Polarity'].apply(lambda x: 'Positive' if x>0.1 else ('Negative' if x<-0.1 else 'Neutral'))
reviews['Sentiment'].value_counts().plot(kind='bar',title='Review Sentiment Distribution'); plt.show()
reviews.groupby('App')['Polarity'].mean().sort_values().head(10).plot(kind='barh',title='10 Apps with Lowest Average Review Polarity'); plt.show()

# Developer insights:
# 1. Investigate categories with high saturation before launching.
# 2. Rating and review counts can help identify user expectations.
# 3. Free apps dominate the sample; consider ads/freemium monetization.
