document.addEventListener('DOMContentLoaded', function() {
    const SERVER_URL = 'http://localhost:5000'; // Update this URL as needed

    const searchForm = document.getElementById('searchForm');
    const productInput = document.getElementById('productInput');
    const suggestionsDiv = document.getElementById('suggestions');
    const toolbar = document.getElementById('toolbar');
    const resultsDiv = document.getElementById('results');
    const basketContents = document.getElementById('basketContents');
    const calculateBasket = document.getElementById('calculateBasket');
    let selectedOption = null;
    let basket = [];

    productInput.addEventListener('focus', fetchInitialSuggestions);
    productInput.addEventListener('input', fetchSuggestions);
    searchForm.addEventListener('submit', searchProducts);

    // Fetch initial suggestions
    function fetchInitialSuggestions() {
        fetch(`${SERVER_URL}/api/unique_items`)
            .then(response => response.json())
            .then(data => {
                toolbar.innerHTML = '';
                data.forEach(item => {
                    const suggestion = document.createElement('div');
                    suggestion.classList.add('suggestion-item');
                    suggestion.textContent = item.ItemName;
                    suggestion.addEventListener('click', function() {
                        fetchPriceChanges(item.ItemCode);
                        toolbar.innerHTML = ''; // Clear the toolbar
                    });
                    toolbar.appendChild(suggestion);
                });
            })
            .catch(error => console.error('Error:', error));
    }

    // Fetch suggestions based on input
    function fetchSuggestions() {
        const query = productInput.value;
        if (query.length > 1) {
            fetch(`${SERVER_URL}/api/suggestions?q=${query}`)
                .then(response => response.json())
                .then(data => {
                    toolbar.innerHTML = '';
                    data.forEach(item => {
                        const suggestion = document.createElement('div');
                        suggestion.classList.add('suggestion-item');
                        suggestion.textContent = item.ItemName;
                        suggestion.addEventListener('click', function() {
                            fetchPriceChanges(item.ItemCode);
                            toolbar.innerHTML = ''; // Clear the toolbar
                        });
                        toolbar.appendChild(suggestion);
                    });
                })
                .catch(error => console.error('Error:', error));
        } else {
            fetchInitialSuggestions();
        }
    }

    // Search products based on query
    function searchProducts(event) {
        event.preventDefault();
        const query = productInput.value;
        fetch(`${SERVER_URL}/api/search?q=${query}`)
            .then(response => response.json())
            .then(data => {
                resultsDiv.innerHTML = '';
                data.forEach(item => {
                    const resultCard = document.createElement('div');
                    resultCard.classList.add('result-card');
                    resultCard.innerHTML = `
                        <strong>${item.BrandName}</strong><br>
                        <span>${item.ItemName}</span><br>
                        <span>${item.ItemPrice}</span><br>
                        <button class="add-to-cart-btn" data-item='${JSON.stringify(item)}'>Add to Cart</button>
                        <button class="predict-price-btn" data-item-code='${item.ItemCode}'>Price Prediction</button>
                    `;
                    resultsDiv.appendChild(resultCard);
                });

                // Add event listeners to "Add to Cart" buttons
                const addToCartButtons = document.querySelectorAll('.add-to-cart-btn');
                addToCartButtons.forEach(button => {
                    button.addEventListener('click', addToCart);
                });

                // Add event listeners to "Price Prediction" buttons
                const predictPriceButtons = document.querySelectorAll('.predict-price-btn');
                predictPriceButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        const itemCode = button.getAttribute('data-item-code');
                        fetchPriceChanges(itemCode);
                    });
                });
            })
            .catch(error => console.error('Error:', error));
    }

    // Fetch price changes for a specific item
    function fetchPriceChanges(itemCode) {
        const chartContainer = document.querySelector('.chart-container');
        chartContainer.innerHTML = ''; // Clear previous content
        fetch(`${SERVER_URL}/api/price_changes?item_code=${itemCode}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error('Error:', data.error);
                    return;
                }
                const imgUrl = `${SERVER_URL}/static/plots/predicted_prices_bar_chart.png`;
                chartContainer.innerHTML = `
                    <p>Price Prediction</p>
                    <img src="${imgUrl}" alt="Price Changes Chart">
                `;
            })
            .catch(error => console.error('Error:', error));
    }

    // Add item to cart
    function addToCart(event) {
        const item = JSON.parse(event.target.dataset.item);
        const quantity = prompt('בחר כמות ממוצר זה שתרצה להוסיף לעגלה שלך', '0');

        if (quantity !== null && !isNaN(quantity) && quantity > 0) {
            item.quantity = quantity;
            basket.push(item);
            updateBasketContents();
        }
    }

    // Update the basket contents display
    function updateBasketContents() {
        basketContents.innerHTML = '';
        basket.forEach(item => {
            const basketItem = document.createElement('div');
            basketItem.classList.add('basket-item');
            basketItem.innerHTML = `
                <strong>${item.BrandName}</strong><br>
                <span>${item.ItemName}</span><br>
                <span>Price: ${item.ItemPrice}</span><br>
                <span>Quantity: ${item.quantity}</span>
            `;
            basketContents.appendChild(basketItem);
        });
    }

    // Add event listeners to "Add to Cart" buttons
    const addToCartButtons = document.querySelectorAll('.add-to-cart-btn');
    addToCartButtons.forEach(button => {
        button.addEventListener('click', addToCart);
    });


    // Toggle visibility of options based on selected radio button
    function toggleOption(optionValue, elementId) {
        const element = document.getElementById(elementId);
        const radio = document.querySelector(`input[value="${optionValue}"]`);

        radio.addEventListener('click', function() {
            if (selectedOption === optionValue) {
                radio.checked = false;
                selectedOption = null;
                element.style.display = 'none';
            } else {
                if (selectedOption) {
                    document.querySelector(`input[value="${selectedOption}"]`).checked = false;
                    const previousElement = document.getElementById(selectedOption + 'Options');
                    if (previousElement) previousElement.style.display = 'none';
                }
                selectedOption = optionValue;
                element.style.display = 'block';
            }
        });
    }

    toggleOption('favoriteBrand', 'brandOptions');
    toggleOption('preferredSupermarket', 'supermarketOptions');

    // Show club options if club member is selected
    document.querySelector('input[name="clubMember"]').addEventListener('change', function() {
        document.getElementById('clubOptions').style.display = this.checked ? 'block' : 'none';
    });

    // Show coupon options if use coupon is selected
    document.querySelector('input[name="useCoupon"]').addEventListener('change', function() {
        document.getElementById('couponOptions').style.display = this.checked ? 'block' : 'none';
    });
});
