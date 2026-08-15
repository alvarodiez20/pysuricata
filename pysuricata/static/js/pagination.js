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
    // Set by clicking a quality chip in the "needs attention" block. The chips
    // were already computed per column; this is what makes them navigation
    // rather than decoration.
    let currentFlag = null;
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

        // Only the page buttons are pointless on a single page. Search, the
        // type tabs and the chip filter are not -- returning early here left
        // them wired to nothing on any report with ten columns or fewer, which
        // is most of them.
        if (allCards.length <= CARDS_PER_PAGE) {
            const pagination = document.querySelector('.pagination');
            if (pagination) pagination.style.display = 'none';
        }

        setupSearch();
        setupFilters();
        setupFlagFilters();
        if (allCards.length > CARDS_PER_PAGE) setupPagination();
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
                // A type tab is a different question from a chip, so choosing
                // one clears the other. "All" therefore restores source order
                // in a single click, which is the point of keeping it.
                clearFlagFilter();
                applyFilters();
            });
        });
    }

    function clearFlagFilter() {
        currentFlag = null;
        document.querySelectorAll('.needs-attention .flag.selected')
            .forEach(chip => chip.classList.remove('selected'));
        const banner = document.getElementById('flag-filter-banner');
        if (banner) banner.remove();
    }

    function setupFlagFilters() {
        const block = document.querySelector('.needs-attention');
        if (!block) return;
        block.addEventListener('click', (e) => {
            const chip = e.target.closest('.flag[data-flag]');
            if (!chip) return;
            const flag = chip.dataset.flag;
            const wasSelected = currentFlag === flag;
            clearFlagFilter();
            if (!wasSelected) {
                currentFlag = flag;
                block.querySelectorAll(`.flag[data-flag="${flag}"]`)
                    .forEach(c => c.classList.add('selected'));
                showFlagBanner(chip.textContent.trim());
            }
            applyFilters();
            const grid = document.getElementById('cards-grid');
            if (grid) grid.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    }

    function showFlagBanner(label) {
        const controls = document.querySelector('.vars-controls');
        if (!controls) return;
        const banner = document.createElement('div');
        banner.id = 'flag-filter-banner';
        banner.className = 'flag-filter-banner';
        banner.innerHTML = `Showing only columns flagged <strong></strong> ` +
            `<button type="button" class="clear-flag">clear</button>`;
        banner.querySelector('strong').textContent = label;
        banner.querySelector('.clear-flag').addEventListener('click', () => {
            clearFlagFilter();
            applyFilters();
        });
        controls.appendChild(banner);
    }

    function setupPagination() {
        const prev = document.getElementById('prev-btn');
        const next = document.getElementById('next-btn');
        if (prev) prev.addEventListener('click', () => goToPage(currentPage - 1));
        if (next) next.addEventListener('click', () => goToPage(currentPage + 1));
    }

    function applyFilters() {
        // Filter cards
        filteredCards = allCards.filter(card => {
            const cardType = card.dataset.type;
            const cardName = card.dataset.name.toLowerCase();

            const cardFlags = (card.dataset.flags || '').split(' ');

            const typeMatch = currentFilter === 'all' || cardType === currentFilter;
            const searchMatch = !searchTerm || cardName.includes(searchTerm);
            const flagMatch = !currentFlag || cardFlags.indexOf(currentFlag) !== -1;

            return typeMatch && searchMatch && flagMatch;
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
