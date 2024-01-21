import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the datasets
data_path = 'poverty data/API_11_DS2_en_csv_v2_6302025.csv'
metadata_country_path = 'poverty data/Metadata_Country_API_11_DS2_en_csv_v2_6302025.csv'
metadata_indicator_path = 'poverty data/Metadata_Indicator_API_11_DS2_en_csv_v2_6302025.csv'

# Reading the datasets
data_df = pd.read_csv(data_path, skiprows=4)
metadata_country_df = pd.read_csv(metadata_country_path)
metadata_indicator_df = pd.read_csv(metadata_indicator_path)

# Displaying the first few rows of each dataset to understand their structure
data_head = data_df.head()
metadata_country_head = metadata_country_df.head()
metadata_indicator_head = metadata_indicator_df.head()

data_head, metadata_country_head, metadata_indicator_head


# Extracting unique indicator names
unique_indicators = data_df['Indicator Name'].unique()

# Displaying a subset of these indicators to choose relevant ones for analysis
unique_indicators_subset = unique_indicators[:20]  # Displaying the first 20 for brevity
unique_indicators_subset


selected_countries = ['United States', 'China', 'India', 'Germany', 'Brazil']

# Adjusting the analysis to focus solely on the Gini index
gini_index_indicator = 'Gini index'

# Filtering the dataset for the Gini index
gini_data = data_df[(data_df['Indicator Name'] == gini_index_indicator) & 
                    (data_df['Country Name'].isin(selected_countries))]

# Reshaping the dataframe for easier plotting
gini_reshaped_data = gini_data.pivot_table(index='Country Name', 
                                           columns='Indicator Code', 
                                           values=[str(year) for year in range(2000, 2023)])

# Plotting the Gini index trends for the selected countries
plt.figure(figsize=(12, 6))
sns.lineplot(data=gini_reshaped_data.droplevel(0, axis=1))
plt.title('Gini Index Trends (2000-2022)')
plt.ylabel('Gini Index')
plt.xlabel('Year')
plt.legend(gini_reshaped_data.columns, loc='upper left')
plt.grid(True)
plt.show()


# From the graph, we can observe:
# United States: The Gini Index shows fluctuations, generally indicating a trend of increasing income inequality over the years.
# China: There's a notable upward trend in the Gini Index, suggesting a widening gap in income distribution.
# India: The data shows a gradual increase in income inequality, with some fluctuations.
# Germany: The trend is relatively stable with slight variations, indicating a more consistent income distribution compared to the other countries.
# Brazil: Despite fluctuations, there's a general trend of high income inequality, although it seems to have decreased slightly in the most recent years.

# Defining the new set of indicators to be analyzed
new_indicators = [
    'Income share held by lowest 20%',
    'Income share held by highest 20%',
    'Poverty gap at $3.65 a day (2017 PPP) (%)',
    'Poverty headcount ratio at national poverty lines (% of population)'
]

# Checking the availability of these indicators in the dataset
new_indicator_availability = {indicator: indicator in data_df['Indicator Name'].values for indicator in new_indicators}


# Filtering the dataset for the new set of selected indicators and countries
new_filtered_data = data_df[(data_df['Indicator Name'].isin(new_indicators)) & 
                            (data_df['Country Name'].isin(selected_countries))]

# Reshaping the dataframe for easier plotting
new_reshaped_data = new_filtered_data.pivot_table(index=['Country Name', 'Indicator Name'], 
                                                  columns='Indicator Code', 
                                                  values=[str(year) for year in range(2000, 2023)])

# Plotting the data
fig, axes = plt.subplots(nrows=len(new_indicators), ncols=1, figsize=(12, 18))

for i, indicator in enumerate(new_indicators):
    sns.lineplot(data=new_reshaped_data.xs(indicator, level='Indicator Name'), ax=axes[i])
    axes[i].set_title(indicator)
    axes[i].legend(new_reshaped_data.xs(indicator, level='Indicator Name').columns, loc='upper left')
    axes[i].set_ylabel('Value')
    axes[i].set_xlabel('Year')

plt.show()


# The visualizations provide a comprehensive view of income distribution and poverty levels across the United States, China, India, Germany, and Brazil, based on the selected indicators:
# Income Share Held by Lowest 20%: This indicator shows the portion of total income that goes to the bottom 20% of the population. Lower values indicate a smaller share of income for the poorest segment, suggesting greater inequality.
# Income Share Held by Highest 20%: This reflects the share of total income received by the top 20% of the population. Higher values indicate a larger share of income concentrated at the top, signifying higher income inequality.
# Poverty Gap at $3.65 a Day (2017 PPP) (%): This measures how far, on average, the population falls below the poverty line. A higher value indicates a deeper level of poverty.
# Poverty Headcount Ratio at National Poverty Lines (% of Population): This shows the percentage of the population living below the national poverty line, providing an insight into the prevalence of poverty within each country.
# Story from the Data:
# United States and Germany: The comparison between these two developed nations reveals differences in income distribution and poverty. Germany tends to have a more equitable distribution of income, with a lower share going to the highest 20% and a higher share to the lowest 20%, compared to the United States. The poverty indicators also suggest more effective poverty alleviation in Germany.
# China and India: As two major developing countries, both show significant challenges in income inequality and poverty. However, the trends may reflect the varying impacts of economic policies and growth patterns in these countries.
# Brazil: Known for its high income inequality, Brazil shows significant disparities in income distribution. The high poverty rates further highlight the socio-economic challenges in the country.
# The data underscores the complex interplay of economic growth, income distribution, and poverty. While economic growth is crucial, its benefits need to be more evenly distributed to reduce inequality and poverty. Developed countries like Germany demonstrate that equitable income distribution and effective poverty alleviation are achievable, serving as models for other nations. Developing countries, facing the dual challenges of economic growth and equitable distribution, need to tailor their policies to address these issues effectively.

data_df_cleaned = data_df.drop(columns=['Unnamed: 67'])

# Dropping rows where all yearly data is NaN
year_columns = data_df_cleaned.columns[4:]  # Columns from 1960 to 2022
data_df_cleaned = data_df_cleaned.dropna(subset=year_columns, how='all')

# Checking for duplicate rows
duplicate_rows = data_df_cleaned.duplicated().sum()

# Overview after cleaning
overview_after_cleaning = {
    "Initial number of rows": len(data_df),
    "Number of rows after dropping NaNs": len(data_df_cleaned),
    "Number of duplicate rows": duplicate_rows
}

overview_after_cleaning, data_df_cleaned.head()


# Extracting data for these indicators
poverty_headcount = data_df_cleaned[data_df_cleaned['Indicator Code'] == 'SI.POV.NAHC']
slum_population = data_df_cleaned[data_df_cleaned['Indicator Code'] == 'EN.POP.SLUM.UR.ZS']
multidimensional_poverty = data_df_cleaned[data_df_cleaned['Indicator Code'] == 'SI.POV.MDIM.XQ']
years = [str(year) for year in range(2010, 2021)]

# Cleaning the data by removing NaNs or infinities for curve fitting
# Using poverty headcount data for a selected country as an example
from scipy.optimize import curve_fit

# Function to define a simple model - let's use a linear model for simplicity
def linear_model(x, a, b):
    return a * x + b

# Selecting a country with a relatively complete dataset
country_example = poverty_headcount['Country Name'].value_counts().idxmax()

# Extracting time series data for this country
time_series_data = poverty_headcount[poverty_headcount['Country Name'] == country_example]
poverty_values = time_series_data[years].values.flatten()

# Removing NaNs or infinities
valid_indices = ~np.isnan(poverty_values)
years_numeric = np.array([int(year) for year in years])
years_numeric_clean = years_numeric[valid_indices]
poverty_values_clean = poverty_values[valid_indices]

# Fitting the model to the clean data
params, params_covariance = curve_fit(linear_model, years_numeric_clean, poverty_values_clean, p0=[0, 0])

# Predicting values for the next 20 years (2021-2040)
future_years = np.array(range(2021, 2041))
predicted_values = linear_model(future_years, *params)

# Plotting the best fitting function
plt.figure(figsize=(12, 6))
plt.scatter(years_numeric_clean, poverty_values_clean, label='Actual Data')
plt.plot(future_years, predicted_values, label='Fitted Line', color='red')
plt.title(f'Poverty Headcount Ratio Over Time for {country_example} with Future Prediction')
plt.xlabel('Year')
plt.ylabel('Poverty Headcount Ratio (%)')
plt.legend()
plt.show()

params, future_years, predicted_values


year = '2000'
poverty_data = poverty_headcount[['Country Name', year]].dropna()
slum_data = slum_population[['Country Name', year]].dropna()

# Merging datasets on 'Country Name'
merged_data = poverty_data.merge(slum_data, on='Country Name', suffixes=('_poverty', '_slum'))

# Preparing data for clustering
X = merged_data[[year + '_poverty', year + '_slum']].values

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
merged_data['Cluster'] = kmeans.labels_

# Plotting cluster membership and cluster centers
plt.figure(figsize=(10, 6))
sns.scatterplot(x=merged_data[year + '_poverty'], y=merged_data[year + '_slum'], hue=merged_data['Cluster'], palette='Set1')

# Adding cluster centers to the plot (transforming back to original scale)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=100, c='black', marker='X')
plt.title('Clusters of Countries by Poverty and Urban Living Conditions in 2018')
plt.xlabel('Poverty Headcount Ratio (%)')
plt.ylabel('Population Living in Slums (%)')
plt.legend(title='Cluster')
plt.show()


year = '2020'
poverty_data = poverty_headcount[['Country Name', year]].dropna()
slum_data = slum_population[['Country Name', year]].dropna()

# Merging datasets on 'Country Name'
merged_data = poverty_data.merge(slum_data, on='Country Name', suffixes=('_poverty', '_slum'))

# Preparing data for clustering
X = merged_data[[year + '_poverty', year + '_slum']].values

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
merged_data['Cluster'] = kmeans.labels_

# Plotting cluster membership and cluster centers
plt.figure(figsize=(10, 6))
sns.scatterplot(x=merged_data[year + '_poverty'], y=merged_data[year + '_slum'], hue=merged_data['Cluster'], palette='Set1')

# Adding cluster centers to the plot (transforming back to original scale)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=100, c='black', marker='X')
plt.title('Clusters of Countries by Poverty and Urban Living Conditions in 2018')
plt.xlabel('Poverty Headcount Ratio (%)')
plt.ylabel('Population Living in Slums (%)')
plt.legend(title='Cluster')
plt.show()

