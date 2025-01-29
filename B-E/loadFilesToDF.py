import os
import pandas as pd
import gzip
import xml.etree.ElementTree as ET
import pandas as pd

def load_data_from_gz(folder_path):
    data = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.gz')]
    for file in files:
        print(file)
        file_path = os.path.join(folder_path, file)
        try:
            with gzip.open(file_path, 'rt') as f:
                xml_data = f.read()
                root = ET.fromstring(xml_data)
                for item in root.find('Items').findall('Item'):
                    item_data = {
                        'PriceUpdateDate': item.find('PriceUpdateDate').text,
                        'ItemCode': f"P_{item.find('ItemCode').text}",
                        #'ItemType': item.find('ItemType').text,
                        'ItemName': item.find('ItemName').text,
                       # 'ManufacturerName': item.find('ManufacturerName').text,
                       # 'ManufactureCountry': item.find('ManufactureCountry').text,
                       # 'ManufacturerItemDescription': item.find('ManufacturerItemDescription').text,
                        'UnitQty': item.find('UnitQty').text,
                        'Quantity': item.find('Quantity').text,
                       # 'bIsWeighted': item.find('bIsWeighted').text,
                        'UnitOfMeasure': item.find('UnitOfMeasure').text,
                        'QtyInPackage': item.find('QtyInPackage').text,
                        'ItemPrice': item.find('ItemPrice').text,
                        'UnitOfMeasurePrice': item.find('UnitOfMeasurePrice').text,
                        #'AllowDiscount': item.find('AllowDiscount').text,
                        #'ItemStatus': item.find('ItemStatus').text
                    }
                    data.append(item_data)
        except gzip.BadGzipFile:
            print(f"Skipping file {file_path}: Not a valid gzip file.")
        except PermissionError as e:
            print(f"Permission error: {e}")
    return data


folder_path = '/Users/shirgeorge/Documents/files'
data = load_data_from_gz(folder_path)
df = pd.DataFrame(data)
df = df.drop_duplicates()
print(df)

df.to_csv('data2.csv', index=False)  # Export to a CSV file

import pandas as pd
from pymongo import MongoClient

# MongoDB connection
client = MongoClient(os.getenv("MONGODB_URI")) # configure in .env
db = client['shufersal_data']
collection = db['productsCSV']

# Read the CSV file
file_path = 'FilePath' # edit your path
data = pd.read_csv(file_path)

# Convert the DataFrame to a list of dictionaries
data_dict = data.to_dict('records')

# Insert data into MongoDB
collection.insert_many(data_dict)

print(f'{len(data_dict)} records inserted successfully.')
