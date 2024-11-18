## Readme for Rome Weather Analysis Dashboard

This Streamlit app provides a comprehensive analysis of historical weather data for Rome, Italy, spanning from 1950 to 2022. The app utilizes various analytical and visualization techniques to explore climate patterns, trends, and potential impacts of climate change on Rome's weather.

### **Project Overview**

*   **Data Source:** The app utilizes historical weather records from Rome Ciampino, IT  from 1950 to 2022.
*   **Key Variables:** The analysis focuses on key meteorological variables such as:
    *   Average, maximum, and minimum temperatures (TAVG, TMAX, TMIN) 
    *   Precipitation (PRCP)

### **Analysis Techniques:**

The dashboard employs a variety of analytical techniques, including:

*   **Time Series Analysis:** Visualizes temperature and precipitation trends over time, highlighting seasonal patterns and long-term changes.
*   **Seasonal Decomposition:** Breaks down average temperature data into trend, seasonal, and residual components to understand underlying patterns and anomalies.
*   **Correlation Analysis:** Explores the relationships between different weather variables through a correlation heatmap.
*   **Climate Change Indicators:** Calculates and visualizes temperature anomalies to assess the impact of climate change on Rome's climate.
*   **Precipitation Patterns:**  Analyzes monthly and yearly precipitation patterns, revealing seasonal variations and interannual variability.
*   **Statistical Tests:** Performs the Mann-Kendall test for trend significance and the Shapiro-Wilk test for normality of temperature data.
*   **Machine Learning Models:** Utilizes the Random Forest Regressor to predict average temperature based on other weather variables, demonstrating predictability in Rome's climate.
*   **Data Assimilation:**  Illustrates the concept of data assimilation by combining observed temperatures with a hypothetical model forecast using the Optimal Interpolation method, showcasing its importance in weather forecasting and climate modeling.

### **Dashboard Features**

The dashboard offers several interactive features:

*   **Time Series Plot:** Shows the historical trends of average, maximum, and minimum temperatures, and precipitation over time.
*   **Seasonal Decomposition Plot:** Displays the trend, seasonality, and residual components of average temperature.
*   **Correlation Heatmap:**  Visualizes the correlations between temperature variables and precipitation.
*   **Climate Change Plot:** Charts the temperature anomaly over time, highlighting the warming trend.
*   **Precipitation Plots:** Presents average monthly precipitation and yearly precipitation totals.
*   **Trend Analysis Plot:**  Shows short-term and long-term temperature trends using moving averages.
*   **Statistical Test Results:** Displays the results of the Mann-Kendall and Shapiro-Wilk tests, providing insights into trend significance and data distribution.
*   **Machine Learning Model Performance:** Presents the performance metrics of the Random Forest Regressor for average temperature prediction.
*   **Data Assimilation Visualization:**  Shows the results of combining observed temperatures with a model forecast.

### **Running the App**

*   **Requirements:** The app requires several Python libraries. You can find a list of them in the previously generated `requirements.txt` file. 
*   **Execution:**  The Streamlit app is launched by running the `create_streamlit_dashboard()` function. 

### **Conclusion**

The Rome Weather Analysis Dashboard offers valuable insights into Rome's climate patterns and the potential impacts of climate change. The findings highlighted in the dashboard can inform urban planning, agriculture, water resource management, and climate adaptation strategies in Rome and other regions with similar Mediterranean climates.
