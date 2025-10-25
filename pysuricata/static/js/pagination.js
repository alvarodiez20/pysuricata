/**
 * Minimalistic Variables Pagination
 * Simple, modern pagination for the variables section
 */

(function() {
    'use strict';

    // Configuration
    const CARDS_PER_PAGE = 10;
    const SEARCH_DEBOUNCE = 300;

    // State
    let currentPage = 1;
    let currentFilter = 'all';
    let searchTerm = '';
    let allCards = [];
    let filteredCards = [];

    // Initialize when DOM is ready
    function init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setup);
        } else {
            setup();
        }
    }

    function setup() {
        // Get all cards
        allCards = Array.from(document.querySelectorAll('#cards-grid .var-card'));

        if (allCards.length <= CARDS_PER_PAGE) {
            // Hide pagination controls if not needed
            const controls = document.querySelector('.vars-controls');
            const pagination = document.querySelector('.pagination');
            if (controls) controls.style.display = 'none';
            if (pagination) pagination.style.display = 'none';
            return;
        }

        setupSearch();
        setupFilters();
        setupPagination();
        applyFilters();
    }

    function setupSearch() {
        const searchInput = document.getElementById('search-input');

        if (!searchInput) return;

        let timeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => {
                searchTerm = e.target.value.toLowerCase();
                applyFilters();
            }, SEARCH_DEBOUNCE);
        });
    }

    function setupFilters() {
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                e.target.classList.add('active');
                currentFilter = e.target.dataset.filter;
                applyFilters();
            });
        });
    }

    function setupPagination() {
        document.getElementById('prev-btn').addEventListener('click', () => goToPage(currentPage - 1));
        document.getElementById('next-btn').addEventListener('click', () => goToPage(currentPage + 1));
    }

    function applyFilters() {
        // Filter cards
        filteredCards = allCards.filter(card => {
            const cardType = card.dataset.type;
            const cardName = card.dataset.name.toLowerCase();

            const typeMatch = currentFilter === 'all' || cardType === currentFilter;
            const searchMatch = !searchTerm || cardName.includes(searchTerm);

            return typeMatch && searchMatch;
        });

        // Reset page if needed
        currentPage = 1;
        updateDisplay();
        updatePagination();
    }

    function updateDisplay() {
        // Hide all cards
        allCards.forEach(card => {
            card.style.display = 'none';
        });

        // Show filtered cards for current page
        const startIndex = (currentPage - 1) * CARDS_PER_PAGE;
        const endIndex = startIndex + CARDS_PER_PAGE;
        const visibleCards = filteredCards.slice(startIndex, endIndex);

        if (visibleCards.length === 0) {
            showNoResults();
        } else {
            hideNoResults();
            visibleCards.forEach(card => {
                card.style.display = 'block';
            });
        }

        // Update info
        const info = document.getElementById('pagination-info');
        if (visibleCards.length > 0) {
            info.textContent = `Showing ${startIndex + 1}-${startIndex + visibleCards.length} of ${filteredCards.length}`;
        } else {
            info.textContent = 'No columns found';
        }
    }

    function showNoResults() {
        let noResults = document.getElementById('no-results');
        if (!noResults) {
            noResults = document.createElement('div');
            noResults.id = 'no-results';
            noResults.className = 'no-results';
            noResults.innerHTML = `
                <div class="icon">🔍</div>
                <div class="message">No columns found</div>
                <div class="suggestion">Try adjusting your search or filter</div>
            `;
            document.getElementById('cards-grid').appendChild(noResults);
        }
    }

    function hideNoResults() {
        const noResults = document.getElementById('no-results');
        if (noResults) {
            noResults.remove();
        }
    }

    function updatePagination() {
        const totalPages = Math.ceil(filteredCards.length / CARDS_PER_PAGE);

        document.getElementById('prev-btn').disabled = currentPage <= 1;
        document.getElementById('next-btn').disabled = currentPage >= totalPages;

        // Generate page numbers
        const pageNumbers = document.getElementById('page-numbers');
        let html = '';

        for (let i = 1; i <= totalPages; i++) {
            const active = i === currentPage ? 'active' : '';
            html += `<span class="page-number ${active}" data-page="${i}">${i}</span>`;
        }

        pageNumbers.innerHTML = html;

        // Add click listeners
        pageNumbers.querySelectorAll('.page-number').forEach(btn => {
            btn.addEventListener('click', (e) => {
                goToPage(parseInt(e.target.dataset.page));
            });
        });
    }

    function goToPage(page) {
        const totalPages = Math.ceil(filteredCards.length / CARDS_PER_PAGE);
        if (page >= 1 && page <= totalPages) {
            currentPage = page;
            updateDisplay();
            updatePagination();
        }
    }

    // Initialize
    init();

})();
