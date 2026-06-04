# Rental Prices in the Czech Republic
## Data Analysis in Python Final Project
### Authors:
Matyáš Tvrz (70038431@fsv.cuni.cz)
Jonathan Eugenio Gaeta (48207837@fsv.cuni.cz)

This is an interactive environment for the Data Processing in Python final project created by Matyáš Tvrz and Jonathan Eugenio Gaeta. We scrape data from sreality.cz and bezrealitky.cz on rental properties in the Czech Republic. In the first part, we present an interactive heatmap (choropleth) based on the median rental prices per meter squared in each region. The map includes popups with information about specific properties including a link to the public listing. The user can filter the properties based on characteristics and price. In the second part, we analyze the data. We allow the user to filter by cities, flat types, and area. First, we present basic summary statistics of the rental prices, grouped by districts and flat types. Next, we present exploratory plots, which visually illustrate some of the features of the data. Finally, we estimate two simple regressions: a level OLS on log-level OLS. We allow for the user to toggle controls for districts and flat types, and display the results, along with diagnostic plots and a fixed effects plot for districts.

### Installation of required packages:

```bash
pip install -r requirements.txt
```

## Run the interactive Streamlit app:

```bash
streamlit run streamlit.py
```

# Or continue in the notebook...
