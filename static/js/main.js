/**
 * Main JavaScript file for Energy Demand Predictor
 */

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    
    // Auto-hide flash messages after 5 seconds
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(function(message) {
        setTimeout(function() {
            message.style.transition = 'opacity 0.5s';
            message.style.opacity = '0';
            setTimeout(function() {
                message.remove();
            }, 500);
        }, 5000);
    });
    
    // Add active class to current nav item
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    navLinks.forEach(function(link) {
        const linkPath = link.getAttribute('href');
        if (linkPath && currentLocation.includes(linkPath) && linkPath !== '/') {
            link.classList.add('active');
        } else if (linkPath === '/' && currentLocation === '/') {
            link.classList.add('active');
        }
    });
    
    // Format numbers in tables
    formatTableNumbers();
});

/**
 * Format numbers in tables with 2 decimal places
 */
function formatTableNumbers() {
    const tables = document.querySelectorAll('.table');
    tables.forEach(function(table) {
        const cells = table.querySelectorAll('td');
        cells.forEach(function(cell) {
            // Check if cell contains a number
            const text = cell.textContent.trim();
            if (!isNaN(text) && text !== '') {
                const num = parseFloat(text);
                if (Number.isInteger(num)) {
                    cell.textContent = num;
                } else {
                    cell.textContent = num.toFixed(2);
                }
            }
        });
    });
}

/**
 * Show loading spinner on button click
 * @param {HTMLElement} button - The button element
 */
function showLoading(button) {
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Loading...';
    
    // Store original text to restore if needed
    button.dataset.originalText = originalText;
}

/**
 * Hide loading spinner
 * @param {HTMLElement} button - The button element
 */
function hideLoading(button) {
    button.disabled = false;
    if (button.dataset.originalText) {
        button.innerHTML = button.dataset.originalText;
    }
}

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showToast('Copied to clipboard!', 'success');
    }).catch(function(err) {
        showToast('Failed to copy', 'danger');
    });
}

/**
 * Show toast notification
 * @param {string} message - Message to display
 * @param {string} type - Type of toast (success, danger, warning, info)
 */
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    // Initialize and show toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: 3000 });
    toast.show();
    
    // Remove toast after hidden
    toastElement.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

/**
 * Validate number input
 * @param {HTMLInputElement} input - Input element
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 */
function validateNumberInput(input, min, max) {
    const value = parseFloat(input.value);
    if (isNaN(value)) {
        input.classList.add('is-invalid');
        return false;
    }
    if (min !== undefined && value < min) {
        input.classList.add('is-invalid');
        return false;
    }
    if (max !== undefined && value > max) {
        input.classList.add('is-invalid');
        return false;
    }
    input.classList.remove('is-invalid');
    input.classList.add('is-valid');
    return true;
}

// Export functions for use in HTML
window.showLoading = showLoading;
window.hideLoading = hideLoading;
window.copyToClipboard = copyToClipboard;
window.showToast = showToast;
window.validateNumberInput = validateNumberInput;