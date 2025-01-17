from flask import Flask, render_template, send_from_directory, request
from flask import jsonify
from flask_cors import CORS

from SeasonalTrends import seasonalTrends
# from TNSE import plot_tsne_for_product
from TNSE_RT import plot_tsne_for_product_live
from plt_changeprices import plt_change_prices
from predictionTransformer import prediction
from search import search_products, get_all_unique_items, search_suggestions

app = Flask(__name__, template_folder='../F-E/templates', static_folder='../F-E/static')
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return render_template('price-fluctuations.html')

@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    return search_products(query)

@app.route('/api/suggestions', methods=['GET'])
def suggestions():
    query = request.args.get('q', '')
    return search_suggestions(query)

@app.route('/api/unique_items', methods=['GET'])
def unique_items():
    return get_all_unique_items()



@app.route('/api/predict_price', methods=['GET'])
def predict_price():
    item_code = request.args.get('item_code', '')

    try:
        # Unpack the returned image paths
        bar_chart_img_path, prediction_img_path, analysis_text = prediction(item_code)

        # Return both image paths in the response
        return jsonify({
            'bar_chart_url': bar_chart_img_path,
            'plot_url': prediction_img_path,
            'analysis': analysis_text
        })

    except Exception as e:
        return jsonify({'error': f'Prediction plot generation error: {str(e)}'}), 500

@app.route('/api/price_changes', methods=['GET'])
def price_changes():
    item_code = request.args.get('item_code', '')

    try:
        # Generate the price changes graph and analysis text
        price_changes_img_path, analysis_text = plt_change_prices(item_code)

        if price_changes_img_path:
            return jsonify({
                'price_changes_url': price_changes_img_path,
                'analysis': analysis_text
            })
        else:
            return jsonify({'error': 'Price changes plot generation failed.'}), 500
    except Exception as e:
        return jsonify({'error': f'Price changes plot generation error: {str(e)}'}), 500@app.route('/api/seasonal_trends', methods=['GET'])

@app.route('/api/seasonal_trends', methods=['GET'])
def seasonal_trends():
    item_code = request.args.get('item_code', '')

    try:
        # Generate the seasonal trends graph and analysis text
        seasonal_trends_img_path, analysis_text = seasonalTrends(item_code)

        if seasonal_trends_img_path:
            return jsonify({
                'seasonal_trends_url': seasonal_trends_img_path,
                'analysis': analysis_text
            })
        else:
            return jsonify({'error': 'Seasonal trends plot generation failed.'}), 500
    except Exception as e:
        return jsonify({'error': f'Seasonal trends plot generation error: {str(e)}'}), 500
@app.route('/api/tsne_plot', methods=['GET'])
def tsne_plot():
    item_code = request.args.get('item_code', '')

    try:
        tsne_img_path, analysis_text = plot_tsne_for_product_live(item_code)  # Ensure analysis_text is returned here
        if tsne_img_path:
            return jsonify({
                'tsne_plot_url': tsne_img_path,
                'analysis': analysis_text  # Return the analysis text
            })
        else:
            return jsonify({'error': 't-SNE plot generation failed.'}), 500
    except Exception as e:
        print(f"Error in tsne_plot route: {e}")
        return jsonify({'error': f't-SNE plot generation error: {str(e)}'}), 500
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True)
