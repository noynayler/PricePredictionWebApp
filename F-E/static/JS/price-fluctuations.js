function openTab(evt, tabName) {
    var i, tabcontent, tablinks;

    // Get all elements with class="tabcontent" and hide them
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    // Get all elements with class="tablinks" and remove the class "active"
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab, and add an "active" class to the button that opened the tab
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

document.addEventListener('DOMContentLoaded', function() {
    const SERVER_URL = 'http://localhost:5000';

    const searchForm = document.getElementById('searchForm');
    const productInput = document.getElementById('productInput');
    const resultsDiv = document.getElementById('results');

    const modal = document.getElementById('predictionModal');
    const modalProductName = document.getElementById('modalProductName');
    const productDetails = document.getElementById('productDetails');
    const chartContainer = document.getElementById('chartContainer');
    const analysisText = document.getElementById('analysisText');

    // A modal for displaying larger images
    const largeImageModal = document.createElement('div');
    largeImageModal.classList.add('modal');
    largeImageModal.style.display = 'none';
    largeImageModal.innerHTML = `
        <div class="modal-content">
            <span class="close">&times;</span>
            <img id="largeImage" src="" alt="Large View" style="max-width: 100%; height: auto;">
        </div>
    `;
    document.body.appendChild(largeImageModal);

    const largeImage = document.getElementById('largeImage');
    const closeLargeImageModal = largeImageModal.querySelector('.close');

    closeLargeImageModal.onclick = function() {
        largeImageModal.style.display = 'none';
    };

    // Function to handle search and display results in grid cards
    async function searchProducts(event) {
        event.preventDefault();
        const query = productInput.value;

        try {
            const response = await fetch(`${SERVER_URL}/api/search?q=${query}`);
            const data = await response.json();
            resultsDiv.innerHTML = ''; // Clear previous results

            // Create a container for the grid
            const gridContainer = document.createElement('div');
            gridContainer.classList.add('grid-container');

            // Add each result to the grid
            data.forEach(item => {
                const resultCard = document.createElement('div');
                resultCard.classList.add('result-card');
                resultCard.innerHTML = `
                    <strong>${item.BrandName}</strong><br>
                    <span>${item.ItemName}</span><br>
                    ₪ <span>${item.ItemPrice} </span>
                `;
                resultCard.addEventListener('click', function() {
                    openModal(item);
                });
                gridContainer.appendChild(resultCard);
            });

            // Append the grid container to the resultsDiv
            resultsDiv.appendChild(gridContainer);
        } catch (error) {
            console.error('Error:', error);
        }
    }

    // Function to open the modal and assign data to tabs
    async function openModal(item) {
        modalProductName.textContent = item.ItemName;
        productDetails.textContent = `Brand: ${item.BrandName}, Price: ${item.ItemPrice} ₪`;
        modal.style.display = 'block';

        // Display "Loading statistics" message for each graph placeholder
        chartContainer.innerHTML = `
            <p id="predictPriceLoading">Loading prediction graph...</p>
            <p id="priceChangesLoading">Loading price changes graph...</p>
            <p id="seasonalTrendsLoading">Loading seasonal trends graph...</p>
            <p id="tsneLoading">Loading t-SNE graph...</p>
        `;
        analysisText.textContent = 'Loading analysis...'; // Placeholder for analysis

        // Run each function sequentially
        try {
            await runAndDisplay(fetchPredictPriceGraph, item.ItemCode, 'predictPriceLoading');
            await runAndDisplay(fetchPriceChangesGraph, item.ItemCode, 'priceChangesLoading');
            await runAndDisplay(fetchSeasonalTrendsGraph, item.ItemCode, 'seasonalTrendsLoading');
            await runAndDisplay(fetchTsnePlot, item.ItemCode, 'tsneLoading');
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    }

    // Function to run each fetch, display graph and analysis, then move to the next function
    async function runAndDisplay(fetchFunction, itemCode, loadingElementId) {
        const data = await fetchFunction(itemCode);

        if (!data) {
            document.getElementById(loadingElementId).textContent = 'Failed to load graph.';
            return;
        }

        // Remove loading message
        document.getElementById(loadingElementId).remove();

        // Display graph in Statistics tab
        if (data.plot_url) {
            createThumbnail(`${SERVER_URL}${data.plot_url}`, 'Graph');
        }
        if (data.bar_chart_url) {
            createThumbnail(`${SERVER_URL}${data.bar_chart_url}`, 'Bar Chart');
        }
        if (data.tsne_plot_url) {
            createThumbnail(`${SERVER_URL}${data.tsne_plot_url}`, 't-SNE Plot');
        }
        if (data.price_changes_url) {
            createThumbnail(`${SERVER_URL}${data.price_changes_url}`, 'Price Changes Plot');
        }
        if (data.seasonal_trends_url) {
            createThumbnail(`${SERVER_URL}${data.seasonal_trends_url}`, 'Seasonal Trends Plot');
        }

        // Append analysis text to Graph Analysis tab
        if (data.analysis) {
            appendAnalysisText(data.analysis);
        }
    }

    // Function to fetch and display the t-SNE graph
    async function fetchTsnePlot(itemCode) {
        try {
            const response = await fetch(`${SERVER_URL}/api/tsne_plot?item_code=${itemCode}`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching t-SNE graph:', error);
            return null;
        }
    }

    // Fetch predict_price graphs and analysis
    async function fetchPredictPriceGraph(itemCode) {
        try {
            const response = await fetch(`${SERVER_URL}/api/predict_price?item_code=${itemCode}`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching predict_price graphs:', error);
            return null;
        }
    }

    // Fetch price changes graphs and analysis
    async function fetchPriceChangesGraph(itemCode) {
        try {
            const response = await fetch(`${SERVER_URL}/api/price_changes?item_code=${itemCode}`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching price_changes graph:', error);
            return null;
        }
    }

    // Fetch seasonal trends graphs and analysis
    async function fetchSeasonalTrendsGraph(itemCode) {
        try {
            const response = await fetch(`${SERVER_URL}/api/seasonal_trends?item_code=${itemCode}`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching seasonal_trends graph:', error);
            return null;
        }
    }

    // Display graph data in Statistics tab
    function displayGraphData(data, loadingElementId) {
        if (!data) {
            document.getElementById(loadingElementId).textContent = 'Failed to load graph.';
            return;
        }

        document.getElementById(loadingElementId).remove(); // Remove loading message

        if (data.plot_url) {
            createThumbnail(`${SERVER_URL}${data.plot_url}`, 'Graph');
        }

        if (data.bar_chart_url) {
            createThumbnail(`${SERVER_URL}${data.bar_chart_url}`, 'Bar Chart');
        }
    }

    // Append analysis text to the Graph Analysis tab
        function appendAnalysisText(text) {
            // Ensure the #analysisText container has RTL direction
            analysisText.style.direction = 'rtl';
            analysisText.style.textAlign = 'right';
            // If "Loading analysis..." is still present, remove it
            if (analysisText.textContent.includes('Loading analysis...')) {
                analysisText.textContent = ''; // Clear the loading text
            }

            // Define the default icon
            let imageIcon = '';

            // Check the content of the text to decide which icon to use
            if (text.includes('בהתבסס על נתוני העבר, המחירים') || text.includes('לא נמצאו תנודות משמעותיות במחירים')) {
                imageIcon = '<img src="../static/images/price-changes.png" alt="Price Changes Icon" class="icon-margin">';
            } else if (text.includes('המחיר הממוצע')) {
                imageIcon = '<img src="../static/images/price-prediction.png" alt="Price Prediction Icon" class="icon-margin">';
            } else if (text.includes('השפעות עונתיות') || text.toLowerCase().includes('seasonal')) {
                imageIcon = '<img src="../static/images/seasonal-trends.png" alt="Seasonal Effect Icon" class="icon-margin">';
            } else if (text.toLowerCase().includes('מוצרים עם טרנדים דומים')) {
                imageIcon = '<img src="../static/images/similar_products.png" alt="Similar Products Icon" class="icon-margin">';
            }



            const formattedText = `${imageIcon} ${text.split('\n').join('<br>')}`; // Add the image and format the text for line breaks
            analysisText.innerHTML += `${formattedText}<br><br>`; // Add extra space between items
        }


    // Create thumbnail for small square image in Statistics tab
    function createThumbnail(imageUrl, altText) {
        const thumbnail = document.createElement('img');
        thumbnail.src = imageUrl;
        thumbnail.alt = altText;
        thumbnail.classList.add('thumbnail');
        thumbnail.style.width = '100px';
        thumbnail.style.height = '100px';
        thumbnail.style.objectFit = 'cover';
        thumbnail.style.margin = '10px';
        thumbnail.style.cursor = 'pointer';

        // Event listener to open the image in a larger view
        thumbnail.addEventListener('click', function() {
            largeImage.src = imageUrl;
            largeImageModal.style.display = 'block';
        });

        chartContainer.appendChild(thumbnail);
    }

    // Close the modal when clicking on the close button
    const closeModal = document.getElementsByClassName('close')[0];
    closeModal.onclick = function() {
        modal.style.display = 'none';

        // Clear previous charts and analysis when closing the modal
        chartContainer.innerHTML = '';
        analysisText.innerHTML = '';
    };

    // Attach the search function to the form submission
    searchForm.addEventListener('submit', searchProducts);

    // Simulate opening the first tab by default
    document.getElementsByClassName("tablinks")[0].click();
});
