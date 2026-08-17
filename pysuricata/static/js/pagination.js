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
        // After applyFilters, which builds the list revealCard indexes into.
        setupDeepLinks();
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
            // An explicit `behavior` beats CSS `scroll-behavior`, so the
            // reduced-motion rule in _01-base.css cannot reach this one.
            const reduce = window.matchMedia &&
                window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            if (grid) grid.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
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

        // A button, not a span with a click listener. The span version could
        // not be reached by keyboard, announced no role, and had "2" as its
        // whole accessible name.
        for (let i = 1; i <= totalPages; i++) {
            const active = i === currentPage ? 'active' : '';
            const current = i === currentPage ? ' aria-current="page"' : '';
            html += `<button type="button" class="page-number ${active}" data-page="${i}" aria-label="Go to page ${i}"${current}>${i}</button>`;
        }

        pageNumbers.innerHTML = html;

        // Add click listeners
        pageNumbers.querySelectorAll('.page-number').forEach(btn => {
            btn.addEventListener('click', (e) => {
                // currentTarget, not target: a click can land on a text node
                // inside the button once it is a real button rather than a span.
                goToPage(parseInt(e.currentTarget.dataset.page));
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

    /* Take a #col_<name> link to the card it names, wherever that card is.
     *
     * Off-page cards are hidden with `display: none`, so a fragment link to one
     * used to do nothing at all: the browser finds no rendered target and stays
     * put. Every link in the needs-attention block is one of these, and the
     * block exists precisely to be clicked -- so the report's own navigation
     * silently failed for any column past the first page.
     *
     * A filter or a search that excludes the target is cleared on the way. A
     * deep link is an explicit request for one column and should outrank a
     * control the reader left set; landing on "No columns found" because a type
     * tab was still on `numeric` would be the same failure with extra steps.
     */
    function revealCard(id) {
        if (!id) return false;
        const card = document.getElementById(id);
        if (!card || !allCards.includes(card)) return false;

        if (!filteredCards.includes(card)) {
            currentFilter = 'all';
            currentFlag = null;
            searchTerm = '';
            const searchInput = document.getElementById('search-input');
            if (searchInput) searchInput.value = '';
            // `.tab`, which is what html.py emits. This read `.filter-tab`
            // first -- a plausible name that appears in no report -- and the
            // tabs would have kept showing a filter that was no longer applied.
            // `test_every_class_it_selects_on_exists` caught it.
            document.querySelectorAll('.tab[data-filter]').forEach((tab) => {
                tab.classList.toggle('active', tab.dataset.filter === 'all');
            });
            applyFilters();
        }

        const index = filteredCards.indexOf(card);
        if (index === -1) return false;
        goToPage(Math.floor(index / CARDS_PER_PAGE) + 1);
        // After updateDisplay, so the card has a box to scroll to.
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return true;
    }

    function setupDeepLinks() {
        // `hashchange` covers the back button and a pasted URL. It does *not*
        // fire when the clicked link's fragment is already the current hash, so
        // the click is handled as well -- otherwise the second click on the same
        // column reads as a dead link.
        window.addEventListener('hashchange', () => {
            revealCard(decodeURIComponent(location.hash.slice(1)));
        });
        document.addEventListener('click', (event) => {
            const link = event.target.closest('a[href^="#col_"]');
            if (!link) return;
            const id = decodeURIComponent(link.getAttribute('href').slice(1));
            if (revealCard(id)) event.preventDefault();
        });
        // A report opened straight at #col_x.
        if (location.hash.startsWith('#col_')) {
            revealCard(decodeURIComponent(location.hash.slice(1)));
        }
    }

    // Initialize
    init();

})();
