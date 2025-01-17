document.addEventListener('DOMContentLoaded', function() {
    const resultsData = [
        { productName: 'Shopping Cart 1', superMarket: 'Tiv Taam', price: '$19.99', dayIndex: -1, timeIndex: -1, selectedOptionIndex: -1 },
        { productName: 'Shopping Cart 2', superMarket: 'Super Yehuda', price: '$29.99', dayIndex: -1, timeIndex: -1, selectedOptionIndex: -1 },
        { productName: 'Shopping Cart 3', superMarket: 'AmPm', price: '$32.5', dayIndex: -1, timeIndex: -1, selectedOptionIndex: -1 },
        // Sample data, replace with actual logic to fetch results
        // Sample data, replace with actual logic to fetch results
    ];

    const resultsTable = document.getElementById('resultsTable').getElementsByTagName('tbody')[0];

    function renderResults() {
        resultsTable.innerHTML = '';
        resultsData.forEach((item, index) => {
            let row = document.createElement('tr');
            row.innerHTML = `
                <td><a href="#" class="cart-details-link">${item.productName}</a></td>
                <td>${item.superMarket}</td>
                <td>${item.price}</td>
                <td class="delivery-options">
                    <div class="delivery-day">
                        <select class="day-select">
                            <option value="0">Sunday</option>
                            <option value="1">Monday</option>
                            <option value="2">Tuesday</option>
                            <option value="3">Wednesday</option>
                            <option value="4">Thursday</option>
                            <option value="5">Friday</option>
                            <option value="6">Saturday</option>
                        </select>
                    </div>
                    <div class="delivery-time">
                        <select class="time-select">
                            <option value="0">09:00-12:00</option>
                            <option value="1">12:00-15:00</option>
                            <option value="2">15:00-18:00</option>
                            <option value="3">18:00-21:00</option>
                        </select>
                    </div>
                </td>
                <td><input type="radio" name="selectOption" data-index="${index}"></td>
            `;
            resultsTable.appendChild(row);

            // Set initial selected values for day and time
            const daySelect = row.querySelector('.day-select');
            const timeSelect = row.querySelector('.time-select');
            daySelect.selectedIndex = item.dayIndex;
            timeSelect.selectedIndex = item.timeIndex;

            // Event listeners for day and time selection
            daySelect.addEventListener('change', function() {
                item.dayIndex = daySelect.selectedIndex;
                updateDeliveryOptions(index);
            });

            timeSelect.addEventListener('change', function() {
                item.timeIndex = timeSelect.selectedIndex;
                updateDeliveryOptions(index);
            });
        });
    }

    function updateDeliveryOptions(rowIndex) {
        // Logic to update delivery options based on day and time selection
        // Replace with your specific logic if needed
        console.log(`Updating delivery options for row ${rowIndex}`);
    }

    renderResults();

    // Event listener for cart details link
    resultsTable.addEventListener('click', function(event) {
        if (event.target.classList.contains('cart-details-link')) {
            event.preventDefault();
            showCartDetails(event.target.innerText);
        }
    });

    // Function to show detailed cart information (dummy function)
    function showCartDetails(productName) {
        alert(`Showing details (Product Name,quantity and price) for ${productName}`);
        // Replace with logic to display detailed cart information
    }
});

