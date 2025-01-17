import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb+srv://admin:9491263@cluster0.nwcjgox.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['shufersal_data']
collection = db['combined_DB']
pd.set_option('display.max_columns', None)

def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S', errors='raise')
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise')
        except (ValueError, TypeError):
            return pd.NaT

def run_seasonal_trends(df, user_period):
    df.set_index('PriceUpdateDate', inplace=True)
    period = user_period
    stl = STL(df['ItemPrice'], period=period)
    result = stl.fit()

    # Plot decomposition components
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax1.plot(df.index, df['ItemPrice'], label='Original')
    ax1.legend(loc='upper left')

    ax2.plot(df.index, result.trend, label='Trend')
    ax2.legend(loc='upper left')

    ax3.plot(df.index, result.seasonal, label='Seasonal')
    ax3.legend(loc='upper left')

    ax4.plot(df.index, result.resid, label='Residual')
    ax4.legend(loc='upper left')

    plt.suptitle('Seasonal Decomposition of Item Prices')
    plt.xlabel('Date')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'../F-E/static/plots/seasonalTrends.png')

    return result

def seasonalTrends(item_code, user_period=12):
    query = {
        'ItemCode': item_code,
        'PriceUpdateDate': {"$gt": '2021-01-01'}
    }

    data = list(collection.find(query))
    df = pd.DataFrame(data)

    # Apply the custom function to convert dates
    df['PriceUpdateDate'] = df['PriceUpdateDate'].apply(parse_date)

    # Extract only the date part
    df['PriceUpdateDate'] = df['PriceUpdateDate'].dt.date
    df['PriceUpdateDate'] = pd.to_datetime(df['PriceUpdateDate'], errors='coerce')
    df.sort_values(by='PriceUpdateDate', inplace=True)

    df['ItemPrice'] = pd.to_numeric(df['ItemPrice'], errors='coerce')
    df_filtered = df[['PriceUpdateDate', 'ItemCode', 'ItemPrice']]

    result = run_seasonal_trends(df_filtered, user_period)

    # Insights and conclusions based on STL decomposition
    trend_change = result.trend[-1] - result.trend[0]
    seasonal_variation = result.seasonal.max() - result.seasonal.min()
    residual_variance = result.resid.var()

    # Analyze seasonal component
    df_filtered['Seasonal'] = result.seasonal
    df_filtered['Month'] = df_filtered.index.month

    seasonal_means = df_filtered.groupby('Month')['Seasonal'].mean()
    max_season = seasonal_means.idxmax()
    min_season = seasonal_means.idxmin()

    seasons = {
        1: 'חורף',
        2: 'חורף',
        3: 'אביב',
        4: 'אביב',
        5: 'אביב',
        6: 'קיץ',
        7: 'קיץ',
        8: 'קיץ',
        9: 'סתיו',
        10: 'סתיו',
        11: 'סתיו',
        12: 'חורף'
    }

    conclusion = "השפעות עונתיות:\n"

    if seasonal_variation > 0:
        conclusion += f"   - נמצאו שינויים עונתיים במחיר עם טווח עונתי של {seasonal_variation:.2f} ש\"ח.\n"
        conclusion += f"   - המחירים הגבוהים ביותר נרשמים ב{seasons[max_season]}.\n"
        conclusion += f"   - המחירים הנמוכים ביותר נרשמים ב{seasons[min_season]}.\n"
    else:
        conclusion += "   - לא נמצאו דפוסים עונתיים משמעותיים במחיר.\n"

    conclusion += "סיכום:\n"
    if trend_change > 0 and seasonal_variation > 0:
        conclusion += "   - המחירים במגמת עלייה עם השפעות עונתיות ברורות. כדאי להתחשב בהשפעות עונתיות בזמן רכישה.\n"
    elif trend_change < 0 and seasonal_variation > 0:
        conclusion += "   - המחירים במגמת ירידה אך עם השפעות עונתיות. אולי כדאי לנצל את התקופות העונתיות הנמוכות לרכישה.\n"
    elif trend_change > 0 and seasonal_variation == 0:
        conclusion += "   - המחירים במגמת עלייה ללא השפעות עונתיות ברורות.\n"
    else:
        conclusion += "   - המחירים יציבים יחסית ללא מגמות ברורות או השפעות עונתיות משמעותיות.\n"

    # print(conclusion)
    return ('/static/plots/seasonalTrends.png', conclusion)

# Example usage
#seasonal_trends('P_7290000965031')
