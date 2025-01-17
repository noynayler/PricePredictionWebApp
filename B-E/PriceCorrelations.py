
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pymongo import MongoClient

client = MongoClient('mongodb+srv://admin:9491263@cluster0.nwcjgox.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['shufersal_data']
collection = db['combined_DB']
pd.set_option('display.max_columns', None)

def analyze_strong_correlations(item_code_input, start_year, end_year ):
    strong_threshold=0.5
    print(f"connecting to DB...")
    data = list(collection.find())
    df = pd.DataFrame(data)

    # Convert 'PriceUpdateDate' to datetime if needed
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S', errors='raise')
        except (ValueError, TypeError):
            try:
                return pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise')
            except (ValueError, TypeError):
                return pd.NaT

    # Apply the custom function to convert dates
    df['PriceUpdateDate'] = df['PriceUpdateDate'].apply(parse_date)

    # Extract only the date part
    df['PriceUpdateDate'] = df['PriceUpdateDate'].dt.date
    df['PriceUpdateDate'] = pd.to_datetime(df['PriceUpdateDate'], errors='coerce')
    df['ItemPrice'] = pd.to_numeric(df['ItemPrice'], errors='coerce')

    df_filtered = df[(df['PriceUpdateDate'].dt.year >= start_year) & (df['PriceUpdateDate'].dt.year <= end_year)]

    # Filter relevant columns
    df_filtered = df_filtered[['PriceUpdateDate', 'ItemCode', 'ItemPrice']]
    print(df_filtered)
    # Create pivot table
    pivot_table = df_filtered.pivot_table(index='PriceUpdateDate', columns='ItemCode', values='ItemPrice')

    print(f"Searching for strong correlations for {item_code_input}...")
    # Compute correlation matrix
    correlation_matrix = pivot_table.corr()

    item_code_to_name = dict(zip(df['ItemCode'], df['ItemName']))

    # Get pairs of strongly correlated items
    strong_correlations = (correlation_matrix > strong_threshold) | (correlation_matrix < -strong_threshold)
    correlated_items = [correlated_item for correlated_item in strong_correlations.index
                        if item_code_input != correlated_item and strong_correlations.loc[correlated_item, item_code_input]]
    print(len(correlated_items))
    # Print results
    if correlated_items:
        item_name = item_code_to_name.get(item_code_input, f"Item {item_code_input}")
        correlated_item_names = [item_code_to_name.get(corr_item, f"Item {corr_item}") for corr_item in correlated_items]
        print(f"{item_name} ({item_code_input}) is strongly correlated with:")
        for corr_item_name, corr_item_code in zip(correlated_item_names, correlated_items):
            print(f"- {corr_item_name} ({corr_item_code})")
    else:
        print(f"No strong correlations found for {item_code_input}")

    # Visualize the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Item Prices')
    plt.show()

    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(df_filtered['ItemCode'])

    # Add edges for strong correlations
    for item_code, correlated_items in strong_correlations.iterrows():
        for corr_item in correlated_items.index:
            if correlated_items[corr_item]:
                G.add_edge(item_code, corr_item)

    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout for the nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_color='black')
    plt.title('Network Graph of Item Price Correlations')
    plt.show()

    sns.clustermap(correlation_matrix, cmap='coolwarm', linewidths=.75)
    plt.title('Clustered Heatmap of Item Price Correlations')
    plt.show()

    sns.pairplot(df_filtered[['ItemCode', 'ItemPrice']], hue='ItemCode', diag_kind='kde')
    plt.title('Pairplot of Item Prices')
    plt.show()


analyze_strong_correlations('P_7290000041445',2021,2024)
