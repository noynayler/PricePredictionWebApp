from flask import jsonify
from pymongo import MongoClient
from datetime import datetime

# MongoDB connection
client = MongoClient('mongodb+srv://admin:9491263@cluster0.nwcjgox.mongodb.net/')
db = client['shufersal_data']
collection = db['products_labels']

def format_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str  # Return the original string if it cannot be parsed

def get_all_unique_items():
    pipeline = [
        {
            "$sort": {"PriceUpdateDate": -1}
        },
        {
            "$group": {
                "_id": "$ItemCode",
                "BrandName": {"$first": "$BrandName"},
                "ItemName": {"$first": "$ItemName"},
                "ItemPrice": {"$first": "$ItemPrice"},
                "PriceUpdateDate": {"$first": "$PriceUpdateDate"}
            }
        },
        {
            "$limit": 5
        }
    ]
    results = list(collection.aggregate(pipeline))
    response = [
        {
            "BrandName": item["BrandName"],
            "ItemName": item["ItemName"],
            "ItemPrice": item["ItemPrice"],
            "ItemCode": item["_id"],
            "PriceUpdateDate": format_date(item["PriceUpdateDate"])
        } for item in results
    ]
    return jsonify(response)

def search_suggestions(query):
    if len(query) < 1:
        return get_all_unique_items()

    regex_query = {'ItemName': {'$regex': f'^{query}', '$options': 'i'}}
    results = collection.find(regex_query).sort('PriceUpdateDate', -1).limit(5)

    response = [
        {
            'BrandName': item['BrandName'],
            'ItemName': item['ItemName'],
            'ItemPrice': item['ItemPrice'],
            'ItemCode': item['ItemCode'],
            'PriceUpdateDate': format_date(item['PriceUpdateDate'])
        } for item in results
    ]
    return jsonify(response)

def search_products(query):
    if len(query) < 1:
        return jsonify([])  # Return empty list if query is too short

    regex_query = {'ItemName': {'$regex': query, '$options': 'i'}}
    results = collection.find(regex_query).sort('PriceUpdateDate', -1)

    # Prepare the response
    response = [
        {
            'BrandName': item['BrandName'],
            'ItemName': item['ItemName'],
            'ItemPrice': item['ItemPrice'],
            'ItemCode': item['ItemCode'],
            'PriceUpdateDate': format_date(item['PriceUpdateDate'])
        } for item in results
    ]

    return jsonify(response)
