import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pymongo import MongoClient
import joblib  # For saving and loading the model

# MongoDB connection details
MONGO_URI = 'mongodb+srv://admin:9491263@cluster0.nwcjgox.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
DB_NAME = 'shufersal_data'
COLLECTION_NAME = 'combined_DB'

# Function to reverse text (for visualizations)
def reverse_text(text):
    return text[::-1]

# Function to preprocess data with flexible date parsing
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S', errors='raise')
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise')
        except (ValueError, TypeError):
            return pd.to_datetime(date_str, errors='coerce')

# Function to fetch all products from the MongoDB database
def fetch_all_products():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    query = {}  # Fetch all documents
    data = list(collection.find(query))

    df = pd.DataFrame(data)
    return df

# Function to train and save the t-SNE model on all products
def train_and_save_tsne_model(use_pca=True, n_pca_components=50, model_filename="tsne_model.joblib"):
    # Fetch all products from the database
    df = fetch_all_products()

    # Preprocess data
    df['PriceUpdateDate'] = df['PriceUpdateDate'].apply(parse_date)
    df['ItemPrice'] = pd.to_numeric(df['ItemPrice'], errors='coerce')
    df = df.dropna(subset=['ItemPrice'])

    # Create the pivot table
    df_pivot = df.pivot_table(index='PriceUpdateDate', columns=['ItemCode', 'ItemName'], values='ItemPrice').fillna(0)

    # Standardize data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_pivot.T)

    # Apply PCA if specified
    if use_pca:
        pca = PCA(n_components=n_pca_components)
        df_scaled = pca.fit_transform(df_scaled)

    # Perform t-SNE
    tsne = TSNE(n_components=2, n_iter=500, perplexity=30, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(df_scaled)

    # Save the model and necessary components (scaler, PCA if used, and t-SNE results)
    joblib.dump((scaler, pca if use_pca else None, tsne_results, df_pivot.columns), model_filename)

# Function to load the model and make predictions for a specific item
# Function to load the model and make predictions for a specific item
def plot_tsne_for_product_live(item_code, n_neighbors=10, model_filename="tsne_model.joblib"):
    # Load the model and components
    scaler, pca, tsne_results, columns = joblib.load(model_filename)

    # Create DataFrame for t-SNE results
    tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['ItemCode'] = columns.get_level_values('ItemCode')
    tsne_df['ItemName'] = columns.get_level_values('ItemName')

    # Find the position of the selected product
    selected_product = tsne_df[tsne_df['ItemCode'] == item_code].iloc[0]

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
    plt.scatter(selected_product['Dimension 1'], selected_product['Dimension 2'], color='red',
                label=reverse_text(f'Selected Product: {selected_product["ItemName"]}'), s=100, edgecolors='black')

    # Variables to control the size of the lines
    horizontal_offset = 0.003  # Increased horizontal length of the line
    vertical_offset = 0.003    # Increased vertical length of the line

    # Plot each similar product with a different color and label direction
    directions = ['top right', 'top left', 'bottom right', 'bottom left']
    for idx, (_, row) in enumerate(similar_products.iterrows()):
        plt.scatter(row['Dimension 1'], row['Dimension 2'], color=colors[idx], s=80, edgecolors='black')

        # Alternate label positions based on index
        direction = directions[idx % len(directions)]
        if direction == 'top right':
            label_pos = (row['Dimension 1'] + horizontal_offset, row['Dimension 2'] + vertical_offset)
        elif direction == 'top left':
            label_pos = (row['Dimension 1'] - horizontal_offset, row['Dimension 2'] + vertical_offset)
        elif direction == 'bottom right':
            label_pos = (row['Dimension 1'] + horizontal_offset, row['Dimension 2'] - vertical_offset)
        elif direction == 'bottom left':
            label_pos = (row['Dimension 1'] - horizontal_offset, row['Dimension 2'] - vertical_offset)

        plt.annotate(
            reverse_text(row['ItemName']),
            xy=(row['Dimension 1'], row['Dimension 2']),
            xytext=label_pos,
            textcoords='data',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=colors[idx], facecolor='white')
        )

        # Create an "r"-shaped line (right angle)
        plt.plot([row['Dimension 1'], row['Dimension 1'], label_pos[0]],
                 [row['Dimension 2'], label_pos[1], label_pos[1]],
                 color=colors[idx], lw=1.5)

    # Annotate the selected product
    label_pos = (selected_product['Dimension 1'] + horizontal_offset, selected_product['Dimension 2'] + vertical_offset)

    plt.annotate(
        reverse_text(selected_product['ItemName']),
        xy=(selected_product['Dimension 1'], selected_product['Dimension 2']),
        xytext=label_pos,
        textcoords='data',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white')
    )

    # Create an "r"-shaped line for the selected product
    plt.plot([selected_product['Dimension 1'], selected_product['Dimension 1'], label_pos[0]],
             [selected_product['Dimension 2'], label_pos[1], label_pos[1]],
             color='red', lw=1.5)

    plt.title(f't-SNE Visualization of Products Similar to {reverse_text(selected_product["ItemName"])}')
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
    plt.savefig(f'../F-E/static/plots/TNSE.png')
    analysis_text = f" מוצרים עם טרנדים דומים: המוצר   '{selected_product['ItemName']}' דומה ל- {n_neighbors} מוצרים נוספים."
    return ('/static/plots/TNSE.png', analysis_text)


# train_and_save_tsne_model()

# Step 2: Use the trained model for real-time predictions
# plot_tsne_for_product_live('P_7290000000022')
