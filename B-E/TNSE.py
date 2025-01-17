import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pymongo import MongoClient

def reverse_text(text):
    return text[::-1]

def plot_tsne_for_product(ItemCode, n_neighbors=10, use_pca=True, n_pca_components=50):
    # Load data from MongoDB
    client = MongoClient('mongodb+srv://admin:9491263@cluster0.nwcjgox.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
    db = client['shufersal_data']
    collection = db['combined_DB']

    query = {'PriceUpdateDate': {"$gt": '2023-01-01'}}
    data = list(collection.find(query))
    df = pd.DataFrame(data)

    # Preprocess data with flexible date parsing
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S', errors='raise')
        except (ValueError, TypeError):
            try:
                return pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise')
            except (ValueError, TypeError):
                return pd.to_datetime(date_str, errors='coerce')

    df['PriceUpdateDate'] = df['PriceUpdateDate'].apply(parse_date)

    # Convert ItemPrice to numeric, coercing any errors
    df['ItemPrice'] = pd.to_numeric(df['ItemPrice'], errors='coerce')

    # Drop any rows where ItemPrice could not be converted to numeric
    df = df.dropna(subset=['ItemPrice'])

    # Create the pivot table using both ItemCode and ItemName as the multi-index columns
    df_pivot = df.pivot_table(index='PriceUpdateDate', columns=['ItemCode', 'ItemName'], values='ItemPrice').fillna(0)

    # Standardize data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_pivot.T)

    # Apply PCA before t-SNE if use_pca is True
    if use_pca:
        pca = PCA(n_components=n_pca_components)
        df_scaled = pca.fit_transform(df_scaled)

    # Perform t-SNE
    tsne = TSNE(n_components=2, n_iter=500, perplexity=30, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(df_scaled)

    # Create a DataFrame for the t-SNE results
    tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['ItemCode'] = df_pivot.columns.get_level_values('ItemCode')
    tsne_df['ItemName'] = df_pivot.columns.get_level_values('ItemName')

    # Find the position of the selected product
    selected_product = tsne_df[tsne_df['ItemCode'] == ItemCode].iloc[0]

    # Calculate distances to the selected product
    tsne_df['Distance'] = np.sqrt(
        (tsne_df['Dimension 1'] - selected_product['Dimension 1'])**2 +
        (tsne_df['Dimension 2'] - selected_product['Dimension 2'])**2
    )

    # Get the n most similar products
    similar_products = tsne_df.nsmallest(n_neighbors + 1, 'Distance')[1:]  # Exclude the selected product itself

    # Assign different colors to each similar product
    colors = plt.cm.get_cmap('tab10', n_neighbors).colors

    # Plot t-SNE results
    plt.figure(figsize=(10, 7))

    # Highlight the selected product
    plt.scatter(selected_product['Dimension 1'], selected_product['Dimension 2'], color='red', label=reverse_text(f'Selected Product: {selected_product["ItemName"]}'), s=100, edgecolors='black')

    # Plot each similar product with a different color
    for idx, (_, row) in enumerate(similar_products.iterrows()):
        plt.scatter(row['Dimension 1'], row['Dimension 2'], color=colors[idx], s=80, edgecolors='black')

    item_name = reverse_text(selected_product["ItemName"])
    plt.title(f't-SNE Visualization of Products Similar to {item_name}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Create a legend with product names and corresponding colors, and place it outside the plot
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=reverse_text(f'{row["ItemName"]}'),
                          markerfacecolor=colors[idx], markersize=10, markeredgecolor='black')
               for idx, (_, row) in enumerate(similar_products.iterrows())]

    handles.insert(0, plt.Line2D([0], [0], marker='o', color='w', label='Selected Product:'+reverse_text(f'{selected_product["ItemName"]}'),
                                 markerfacecolor='red', markersize=10, markeredgecolor='black'))

    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)  # Adjusted to move outside the plot

    plt.tight_layout()
    #plt.show()

    plt.savefig(f'../F-E/static/plots/TNSE.png')
    analysis_text = f" מוצרים עם טרנדים דומים: המוצר   '{selected_product['ItemName']}' דומה ל- {n_neighbors} מוצרים נוספים."
    return ('/static/plots/TNSE.png',analysis_text)

    # Return the list of similar products
    #return similar_products[['ItemCode', 'ItemName', 'Distance']]
# Example usage
#plot_tsne_for_product('P_7290000965031')
#print("Similar Products:")
#print(similar_products)
