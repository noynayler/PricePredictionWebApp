# Smarket – The Smart Market

## Overview

**Smarket – The Smart Market** is an innovative platform designed to transform grocery market analysis and price forecasting. By leveraging advanced machine learning models like Prophet, STL, and t-SNE, it provides users with actionable insights into market trends, price changes, and seasonal fluctuations. The platform caters to individual consumers, investors, and industries by enabling informed purchasing decisions.

## Features

- **Price Prediction**: Accurate short-term price forecasting using the Prophet model.
- **Seasonal Trend Analysis**: Insights into long-term market behavior through STL decomposition.
- **Product Correlation Visualization**: Dimensionality reduction with t-SNE to identify relationships between products.
- **Real-Time Data Updates**: Weekly scraping of market data ensures up-to-date predictions.
- **User-Friendly Interface**: Accessible insights presented through intuitive graphs and statistical summaries.

## System Architecture

The project follows a three-tier architecture:
1. **Frontend**: Built with JavaScript, HTML, and CSS to provide an interactive user experience.
2. **Backend**: Developed in Python using Flask, handling data processing and model predictions.
3. **Database**: MongoDB for flexible and efficient storage of market data.

## Key Technologies

- **Programming Languages**: Python, JavaScript, HTML, CSS.
- **Frameworks**: Flask, Selenium.
- **Libraries**: NumPy, Matplotlib, Prophet, t-SNE.
- **Database**: MongoDB.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SmarketGroup/Smarket.git
   cd Smarket
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Backend**:
   ```bash
   python app.py
   ```









