import numpy as np
import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"

response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', {'class': 'wikitable'})

df = pd.read_html(str(table))[0]

df.columns = [' '.join(col).strip() for col in df.columns.values]


df.rename(columns={'United Nations[14] Estimate': '2023'}, inplace=True)
df.rename(columns={'IMF[1][12] Forecast': '2025'}, inplace=True)
df.rename(columns={'World Bank[13] Estimate': '2024'}, inplace=True)
df.rename(columns={'Country/Territory Country/Territory': 'Country'}, inplace=True)
df.drop(columns=['IMF[1][12] Year', 'World Bank[13] Year', 'United Nations[14] Year'], inplace=True )
df.to_csv("gdp.csv", index=False)

df = pd.read_csv('gdp.csv').replace('—', np.nan)
years = ['2023', '2024', '2025']
df[years] = df[years].apply(pd.to_numeric, errors='coerce')

df.interpolate(inplace=True)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


data = []
for _, row in df.iterrows():
    for i, year in enumerate(years):
        if not pd.isna(row[year]):
            data.append({
                'Country': row['Country'],
                'Year': i,
                'GDP': row[year]
            })
gdp_df = pd.DataFrame(data)

predictions = []
for country in df['Country']:
    country_data = gdp_df[gdp_df['Country'] == country]

    X = country_data[['Year']].values
    y = np.log(country_data['GDP'].values)

    model = LinearRegression()

    model.fit(X, y)
    pred = model.predict([[3]])[0]
    predictions.append(np.exp(pred))

df['2026'] = predictions
df.to_csv('gdp_predictions_sklearn.csv', index=False)
print(df[['Country','2023','2024' ,'2025', '2026']])

from ydata_profiling import ProfileReport
rp = ProfileReport(df)
rp.to_notebook_iframe()