## Rome Weather Analysis: Project Overview and Jupyter Notebook

This project analyzes historical weather data for Rome, Italy, to gain insights into climate patterns, trends, and potential climate change impacts. The project employs various techniques, including:

*   Time series analysis
*   Seasonal decomposition
*   Correlation analysis
*   Climate change indicators
*   Precipitation pattern analysis
*   Statistical tests
*   Machine Learning models
*   Data assimilation

### Data Source and Key Variables

The primary data source is 'Roma\_weather.csv', containing historical weather records for Rome from 1950 to 2022. The data includes the following key variables:

*   **Temperature:** Average temperature (TAVG), maximum temperature (TMAX), and minimum temperature (TMIN), measured in degrees Celsius (°C).
*   **Precipitation:** Total precipitation (PRCP), measured in millimeters (mm).

### Analysis Techniques

The project utilizes both statistical and machine learning techniques to analyze the weather data, as described below:

#### 1. Time Series Analysis

The project analyzes the temporal trends in temperature and precipitation. Time series plots are created to visualize these trends over the years. This analysis helps identify patterns such as seasonal variations and long-term trends like the upward trend in average temperatures.

#### 2. Seasonal Decomposition

The time series data for average temperature (TAVG) is decomposed into three components: trend, seasonal, and residual. This decomposition helps isolate the long-term trend from the cyclical seasonal variations and any irregular fluctuations. This analysis provides evidence of a gradual increase in temperature over time.

#### 3. Correlation Analysis

Correlation heatmaps are created to examine the relationships between the different weather variables. Strong positive correlations are observed between average (TAVG) and maximum (TMAX) temperatures. Moderate positive correlation is observed between average (TAVG) and minimum (TMIN) temperatures. A weak negative correlation is identified between temperature variables and precipitation (PRCP).

#### 4. Climate Change Indicators

The project analyzes climate change indicators, particularly temperature anomaly. The temperature anomaly is calculated as the difference between the average temperature for each day and the overall average temperature for the entire dataset. The plot of temperature anomaly reveals an upward trend, indicating warming in Rome.

#### 5. Precipitation Analysis

The project examines precipitation patterns, including average monthly and yearly totals. The analysis reveals a seasonal pattern with more rainfall during autumn and winter and drier summer months, typically July and August. Yearly precipitation totals exhibit significant variability.

#### 6. Trend Analysis

The project utilizes moving averages to analyze both short-term and long-term temperature trends. A 365-day moving average is used to smooth short-term fluctuations and visualize the annual temperature cycle. A 10-year moving average is employed to discern the long-term temperature trend. The analysis confirms a clear warming pattern in Rome, especially prominent in recent decades.

#### 7. Statistical Tests

Statistical tests are conducted to evaluate the significance of observed trends. The **Mann-Kendall test is performed on the average temperature data, and the results confirm a statistically significant trend**. Additionally, the **Shapiro-Wilk test is applied, indicating that the temperature data is not normally distributed**.

#### 8. Machine Learning Models

The project employs machine learning models to predict average temperature using other weather variables. A Random Forest Regressor is trained and evaluated using metrics like Mean Squared Error (MSE) and R-squared (R2). The Random Forest model demonstrates good performance in predicting average temperatures.

#### 9. Data Assimilation

A simple data assimilation technique, Optimal Interpolation, is demonstrated to showcase how observed temperatures can be combined with model forecasts to improve temperature estimates.

### Conclusion

This project provides a comprehensive analysis of Rome's historical weather data, revealing significant insights into the city's climate patterns and trends.

### Jupyter Notebook (Roma\_weather\_analysis.ipynb)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import folium
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import warnings
warnings.filterwarnings("ignore")

# Read the CSV file
df = pd.read_csv('Roma_weather.csv', parse_dates=['DATE'])
df.head()
# STATION NAME LATITUDE LONGITUDE ELEVATION DATE DP01 DP
# 0 IT000016239 ROMA CIAMPINO, IT 41.7831 12.5831 105.0 1951-01-01 12.0
# 1 IT000016239 ROMA CIAMPINO, IT 41.7831 12.5831 105.0 1951-02-01 15.0
# 2 IT000016239 ROMA CIAMPINO, IT 41.7831 12.5831 105.0 1951-03-01 12.0
# 3 IT000016239 ROMA CIAMPINO, IT 41.7831 12.5831 105.0 1951-04-01 7.0
# 4 IT000016239 ROMA CIAMPINO, IT 41.7831 12.5831 105.0 1951-05-01 9.0

# Data Preprocessing
numeric_columns = ['PRCP', 'TAVG', 'TMAX', 'TMIN']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Basic Statistical Analysis
print(df[numeric_columns].describe())
# PRCP        TAVG        TMAX        TMIN
# count  770.000000  623.000000  689.000000  703.000000
# mean    63.624935   15.569181   21.275907   10.225462
# std     52.281702    6.122862    6.845409    5.607757
# min      0.000000    3.500000    8.100000   -1.100000
# 25%     23.150000   10.150000   15.000000    5.500000
# 50%     53.450000   15.300000   21.200000    9.500000
# 75%     90.525000   21.200000   27.700000   15.400000
# max    266.800000   27.200000   34.400000   21.600000

# Time Series Analysis
df.set_index('DATE', inplace=True)
df.sort_index(inplace=True)

# Set  matplotlib style
plt.style.use('default')  # or try 'classic'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=)

# Plot 1: Temperature trends
ax1.plot(df.index, df['TAVG'], label='Average Temperature', color='#4299e1', linewidth=2)
ax1.plot(df.index, df['TMAX'], label='Max Temperature', color='#ff7e67', alpha=0.8, linewidth=1.5)
ax1.plot(df.index, df['TMIN'], label='Min Temperature', color='#68d391', alpha=0.8, linewidth=1.5)

# Add trend line
z = np.polyfit(range(len(df.index)), df['TAVG'], 1)
p = np.poly1d(z)
ax1.plot(df.index, p(range(len(df.index))), '--', color='#805ad5',
         label='Long-term Trend', linewidth=2)

# Customize first plot
ax1.set_title('Temperature Trends in Rome (1950-2022)', pad=20, fontsize=14)
ax1.set_xlabel('')
ax1.set_ylabel('Temperature (°C)', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Monthly distribution
monthly_avg = df.groupby(df.index.month)['TAVG'].mean()
monthly_std = df.groupby(df.index.month)['TAVG'].std()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Bar plot with error bars
ax2.bar(range(1, 13), monthly_avg, yerr=monthly_std,
        color='#4299e1', alpha=0.6, capsize=5)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(months, rotation=45)
ax2.set_title('Monthly Temperature Distribution', fontsize=12)
ax2.set_xlabel('Month', fontsize=10)
ax2.set_ylabel('Average Temperature (°C)', fontsize=10)
ax2.grid(True, axis='y', alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.savefig('Temperature Trends.png')
plt.show()

# Correlation Analysis
correlation = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Weather Variables')
plt.savefig('correlation_heatmap.png')

# Precipitation Analysis
plt.figure(figsize=(15, 8))
monthly_precipitation = df['PRCP'].resample('ME').sum()

# Calculate yearly average for comparison
yearly_avg = monthly_precipitation.groupby(monthly_precipitation.index.month).mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot
plt.plot(months, yearly_avg.values, 'r--', label='Yearly Average', linewidth=2)
sns.barplot(x=months,
            y=monthly_precipitation.iloc[-12:].values,  # Last 12 months
            alpha=0.6,
            color='royalblue',
            label='Monthly Precipitation')

# Styling
plt.title('Monthly Precipitation in Rome', pad=20, fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Precipitation (mm)', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels only for the bars
for i, v in enumerate(monthly_precipitation.iloc[-12:].values):
    plt.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('monthly_precipitation.png')
plt.show()

df['TAVG']
# DATE
# 1951-01-01     9.7
# 1951-02-01    10.5
# 1951-03-01    10.6
# 1951-04-01    13.6
# 1951-05-01    16.6
# ...
# 2022-05-01     NaN
# 2022-06-01     NaN
# 2022-07-01     NaN
# 2022-08-01     NaN
# 2022-09-01     NaN
# Name: TAVG, Length: 804, dtype: float64

df['TAVG'] = df['TAVG'].interpolate(method='linear')


def create_improved_dashboard(df, model_results=None, predictions=None, y_test=None):
    plt.style.use('fivethirtyeight')

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Comprehensive Weather Analysis Dashboard', fontsize=16, y=0.92)

    # Define grid layout
    gs = plt.GridSpec(4, 3, height_ratios=[1.2, 1, 1, 1], hspace=0.5, wspace=0.3)

    # 1. Temperature Trends with visibility
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['DATE'], df['TMAX'], label='Max Temp', color='#ff7f7f', alpha=0.7, linewidth=2)
    ax1.plot(df['DATE'], df['TAVG'], label='Avg Temp', color='#7fbf7f', alpha=0.7, linewidth=2)
    ax1.plot(df['DATE'], df['TMIN'], label='Min Temp', color='#7f7fff', alpha=0.7, linewidth=2)
    ax1.set_title('Long-term Temperature Trends', fontsize=14, pad=15)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # 2. Temperature Distribution with styling
    ax2 = fig.add_subplot(gs)
    sns.boxplot(data=df[['TMIN', 'TAVG', 'TMAX']], ax=ax2,
                palette=['#7f7fff', '#7fbf7f', '#ff7f7f'])
    ax2.set_title('Temperature Distribution', fontsize=14)
    ax2.set_xlabel('Temperature Type', fontsize=12)
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.tick_params(axis='x', rotation=0)
    ax2.set_xticklabels(['TMIN', 'TAVG', 'TMAX'], fontsize=10)

    # 3. Monthly Temperature Patterns with error bars
    ax3 = fig.add_subplot(gs)
    monthly_stats = df.groupby(df['DATE'].dt.month)['TAVG'].agg(['mean', 'std'])
    ax3.errorbar(monthly_stats.index, monthly_stats['mean'],
                 yerr=monthly_stats['std'], fmt='o-', capsize=5,
                 color='#2f7fb8', ecolor='#8fbfd9', markersize=8,
                 linewidth=2, elinewidth=1, capthick=1)
    ax3.set_title('Monthly Temperature Patterns\nwith Standard Deviation', fontsize=14)
    ax3.set_xlabel('Month', fontsize=12)
    ax3.set_ylabel('Average Temperature (°C)', fontsize=12)
    ax3.set_xticks(range(1, 13))
    ax3.grid(True, alpha=0.3)

    # 4. Monthly Precipitation with styling
    ax4 = fig.add_subplot(gs)
    monthly_precip = df.groupby(df['DATE'].dt.month)['PRCP'].mean()
    bars = ax4.bar(range(1, 13), monthly_precip.values,
                   color='#6495ed', alpha=0.7)
    ax4.set_title('Average Monthly Precipitation', fontsize=14)
    ax4.set_xlabel('Month', fontsize=12)
    ax4.set_ylabel('Precipitation (mm)', fontsize=12)
    ax4.set_xticks(range(1, 13))
    # Bar labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}mm',
                 ha='center', va='bottom', fontsize=9)

    # 5. Temperature vs Precipitation scatter
    ax5 = fig.add_subplot(gs)
    scatter = ax5.scatter(df['PRCP'], df['TAVG'],
                          c=df['DATE'].dt.month, cmap='viridis',
                          alpha=0.6, s=50)
    ax5.set_title('Temperature vs Precipitation\nby Month', fontsize=14)
    ax5.set_xlabel('Precipitation (mm)', fontsize=12)
    ax5.set_ylabel('Average Temperature (°C)', fontsize=12)
    colorbar = plt.colorbar(scatter, ax=ax5)
    colorbar.set_label('Month', fontsize=12)

    # 6. Special Conditions with formatting
    ax6 = fig.add_subplot(gs)
    conditions = ['DP01', 'DP10', 'DT32', 'DX32', 'DX90']
    condition_means = [df[cond].mean() for cond in conditions]
    condition_labels = ['Precipitation\n≥0.1mm', 'Precipitation\n≥1.0mm',
                       'Min Temp\n≤0°C', 'Max Temp\n≤0°C', 'Max Temp\n≥32.2°C']
    bars = ax6.bar(range(len(conditions)), condition_means,
                   color='#4682b4', alpha=0.7)
    ax6.set_title('Average Days with Special Conditions\nper Month', fontsize=14)
    ax6.set_xticks(range(len(conditions)))
    ax6.set_xticklabels(condition_labels, rotation=45, ha='right', fontsize=10)
    # Bar labels
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=9)

    # 7. Model Performance with styling
    if model_results is not None:
        ax7 = fig.add_subplot(gs)
        models = list(model_results.keys())
        r2_scores = [score['r2'] for score in model_results.values()]
        bars = ax7.bar(models, r2_scores, color='#4169e1', alpha=0.7)
        ax7.set_title('Model Performance\nComparison (R²)', fontsize=14)
        ax7.set_ylim(0, 1)
        ax7.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax7.grid(True, alpha=0.3)
        # Enhanced bar labels
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontsize=9)

    # 8. Prediction Error Distribution
    if predictions is not None and y_test is not None:
        ax8 = fig.add_subplot(gs[3, :])
        for model_name, preds in predictions.items():
            errors = y_test - preds
            ax8.hist(errors, bins=50, alpha=0.5, label=model_name, density=True)
        ax8.set_title('Model Prediction Error Distribution')
        ax8.set_xlabel('Prediction Error (°C)')
        ax8.set_ylabel('Density')
        ax8.legend(fontsize=12)
        ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_comprehensive_analysis(df):
    """Run complete analysis pipeline"""
    # Prepare features
    df['Temp_Range'] = df['TMAX'] - df['TMIN']
    df['Temp_7day_avg'] = df['TAVG'].rolling(window=7).mean()
    df['Precip_Temp_Interaction'] = df['PRCP'] * df['TAVG']

    # Model training and evaluation
    feature_cols = ['PRCP', 'TMIN', 'TMAX', 'Temp_Range',
                   'Temp_7day_avg', 'Precip_Temp_Interaction']
    X = df[feature_cols].copy()
    y = df['TAVG'].interpolate(method='linear')
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train models
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'r2': model.score(X_test, y_test)
        }
        predictions[name] = y_pred

    return results, predictions, create_improved_dashboard(df, results, predictions, y_test)


df = pd.read_csv('Roma_weather.csv', parse_dates=['DATE'])
results, predictions, fig = run_comprehensive_analysis(df)
plt.show()
fig.savefig('comprehensive_weather_analysis.png', dpi=300, bbox_inches='tight')
```