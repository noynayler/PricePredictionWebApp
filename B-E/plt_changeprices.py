import pandas as pd
import matplotlib.pyplot as plt
import holidays
import mplcursors
from pymongo import MongoClient

client = MongoClient('mongodb+srv://admin:9491263@cluster0.nwcjgox.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['shufersal_data']
collection = db['combined_DB']

def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S', errors='raise')
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise')
        except (ValueError, TypeError):
            return pd.NaT

# Mapping of English months to Hebrew months
month_translation = {
    'January': 'ינואר',
    'February': 'פברואר',
    'March': 'מרץ',
    'April': 'אפריל',
    'May': 'מאי',
    'June': 'יוני',
    'July': 'יולי',
    'August': 'אוגוסט',
    'September': 'ספטמבר',
    'October': 'אוקטובר',
    'November': 'נובמבר',
    'December': 'דצמבר'
}
def generate_conclusion(df):
    df['PriceUpdateDate'] = pd.to_datetime(df['PriceUpdateDate'])
    df = df.sort_values(by='PriceUpdateDate').copy()

    # Calculate price differences
    df['PriceDiff'] = df['ItemPrice'].diff()

    # Find the dates with the maximum price increase and decrease
    max_increase_date = df.iloc[df['PriceDiff'].idxmax()]['PriceUpdateDate']
    max_decrease_date = df.iloc[df['PriceDiff'].idxmin()]['PriceUpdateDate']

    # Extract the months of maximum increase and decrease
    max_increase_month = month_translation[max_increase_date.strftime('%B')]
    max_decrease_month = month_translation[max_decrease_date.strftime('%B')]

    # Generate the advice
    if df['PriceDiff'].max() > 0:
        advice = (f"בהתבסס על נתוני העבר, המחירים נוטים לעלות בעיקר בחודש {max_increase_month}. "
                  f"לכן, כדאי לשקול לרכוש את המוצר לפני חודש זה. "
                  f"מנגד, המחירים נוטים לרדת בעיקר בחודש {max_decrease_month}, "
                  f"לכן, זהו הזמן האידיאלי לרכוש את המוצר במחיר נמוך יותר.")
    else:
        advice = "לא נמצאו תנודות משמעותיות במחירים לאורך השנה, ולכן ניתן לרכוש את המוצר בכל עת."

    return advice

def item_prices_with_holidays(data, show_holidays=True):
    df = pd.DataFrame(data)
    df['PriceUpdateDate'] = pd.to_datetime(df['PriceUpdateDate'])
    df = df.sort_values(by='PriceUpdateDate').copy()

    plt.figure(figsize=(12, 8))

    item_code_to_name = dict(zip(df['ItemCode'], df['ItemName']))

    holiday_lines = {}

    for item_code, group in df.groupby('ItemCode'):
        item_name = item_code_to_name[item_code]
        line, = plt.plot(group['PriceUpdateDate'], group['ItemPrice'], marker='o', linestyle='-', label=item_name)
        holiday_lines[item_name] = line

    israel_holidays = holidays.Israel(years=[2023, 2024])
    holiday_dates = pd.to_datetime(list(israel_holidays.keys()))

    holiday_colors = {
        'ראש השנה': 'red',
        'יום כיפור': 'blue',
        'סוכות': 'green',
        'שמחת תורה/שמיני עצרת': 'purple',
        'פסח': 'orange',
        'שביעי של פסח': 'yellow',
        'יום העצמאות': 'cyan',
        'שבועות': 'magenta',
    }

    if show_holidays:
        for holiday_date, holiday_name in israel_holidays.items():
            color = holiday_colors.get(holiday_name, 'black')  # Default to black if holiday name not in dictionary
            plt.axvline(holiday_date, color=color, linestyle='--', linewidth=0.8, label=holiday_name)

    plt.title('Price changes ' + (' (Holidays included)' if show_holidays else ''))
    plt.xlabel('Date')
    plt.ylabel('Product Price')

    handles, labels = plt.gca().get_legend_handles_labels()
    legend_labels = [f'\u200F{label[::-1]}' for label in labels]  # Right-to-left text
    plt.legend(handles, legend_labels, title='שם מוצר', loc='center left', bbox_to_anchor=(1, 0.5))

    plt.grid(True)
    plt.tight_layout()

    cursor = mplcursors.cursor(hover=True)

    for item_name, line in holiday_lines.items():
        cursor.connect('add', lambda sel: sel.annotation.set_text(f"{item_name}\nתאריך: {sel.artist.get_xdata()[sel.ind[0]]}\nמחיר: {sel.artist.get_ydata()[sel.ind[0]]}"))

    def toggle_visibility(holiday_name):
        for item_name, line in holiday_lines.items():
            if item_name == holiday_name:
                line.set_visible(not line.get_visible())
        plt.legend()

    plt.connect('key_press_event', lambda event: toggle_visibility(event.key))
    #plt.show()
    plt.savefig(f'../F-E/static/plots/priceChanges.png')

def plt_change_prices(item_code):
    query = {
        'ItemCode': item_code,
        'PriceUpdateDate': {"$gt": '2023-01-01'}
    }

    data = list(collection.find(query))
    df = pd.DataFrame(data)
    # Apply the custom function to convert dates
    df['PriceUpdateDate'] = df['PriceUpdateDate'].apply(parse_date)

    # Extract only the date part
    df['PriceUpdateDate'] = df['PriceUpdateDate'].dt.date
    df['PriceUpdateDate'] = pd.to_datetime(df['PriceUpdateDate'], errors='coerce')
    df['ItemPrice'] = pd.to_numeric(df['ItemPrice'], errors='coerce')
    df_filtered = df[['PriceUpdateDate', 'ItemCode', 'ItemPrice']]
    #print(df_filtered)

    # Generate and print the conclusion
    conclusion =  generate_conclusion(df)

    #print(conclusion)

    item_prices_with_holidays(df, show_holidays=True)
    return ('/static/plots/priceChanges.png',conclusion)

# Example usage:
#plt_change_prices('P_7290000000091')