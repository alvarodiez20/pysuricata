/**
 * Minimalistic Variables Pagination
 * Simple, modern pagination for the variables section
 */

(function() {
    'use strict';

    // Configuration
    //
    // Not a page size. The first EXPANDED_LIMIT matching columns render in
    // full; every column after that keeps a row in the document with its body
    // folded (design 15d). A hidden card is a card a browser find cannot
    // match, an anchor cannot land on and a printer will not print -- and the
    // primary action in a profiling report is finding a column by name.
    const EXPANDED_LIMIT = 10;
    const SEARCH_DEBOUNCE = 300;

    // State
    // Columns the reader has opened by hand, by card id, plus the flag set by
    // "expand all". Both survive a filter change: having opened a column is a
    // decision, and re-collapsing it because a search ran is undoing it.
    let opened = new Set();
    let expandAll = false;
    // Dataset order is the default and stays it: a reader working alongside
    // their dataframe expects the report's columns in the frame's order, and
    // any other default silently disagrees with the thing on their other
    // screen. The grid's DOM order is that order, so it is the baseline every
    // other sort is applied against.
    let sortBy = 'dataset';
    let datasetOrder = [];
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
        datasetOrder = allCards.slice();

        setupSearch();
        setupFilters();
        setupFlagFilters();
        setupExpansion();
        setupSort();
        setupClearFilter();
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

    function setupExpansion() {
        const grid = document.getElementById('cards-grid');
        const all = document.getElementById('expand-all');

        // Delegated, so a card that is collapsed after a filter change is still
        // clickable without rebinding. Only the header opens a card: a click
        // inside an expanded body belongs to whatever it landed on.
        if (grid) {
            grid.addEventListener('click', (e) => {
                const card = e.target.closest('.var-card.is-collapsed');
                if (!card) return;
                // A link in a collapsed header still navigates.
                if (e.target.closest('a[href]')) return;
                opened.add(card.id);
                updateDisplay();
            });
            grid.addEventListener('keydown', (e) => {
                if (e.key !== 'Enter' && e.key !== ' ') return;
                const card = e.target.closest('.var-card.is-collapsed');
                if (!card) return;
                e.preventDefault();
                opened.add(card.id);
                updateDisplay();
            });
        }

        if (all) {
            all.addEventListener('click', () => {
                expandAll = !expandAll;
                if (!expandAll) opened.clear();
                updateDisplay();
            });
        }
    }


    function setupSort() {
        const select = document.getElementById('sort-select');
        if (!select) return;
        select.addEventListener('change', () => {
            sortBy = select.value;
            applySort();
            applyFilters();
        });
    }

    /* Reorder the grid itself, not a copy.
     *
     * The cards are the document -- moving them is what makes the order true
     * for a browser find, for print and for anyone reading the page top to
     * bottom, rather than only for this script's own bookkeeping.
     */
    function applySort() {
        const grid = document.getElementById('cards-grid');
        if (!grid) return;
        const order = datasetOrder.slice();
        const missing = c => parseFloat(c.dataset.missingPct || '0');
        const flags = c => (c.dataset.flags || '').split(' ').filter(Boolean).length;

        if (sortBy === 'missing') {
            order.sort((a, b) => missing(b) - missing(a));
        } else if (sortBy === 'flagged') {
            order.sort((a, b) => flags(b) - flags(a));
        } else if (sortBy === 'name') {
            order.sort((a, b) => a.dataset.name.localeCompare(b.dataset.name));
        }
        // `dataset` needs no comparator -- `order` is already that order, which
        // is why the original list is kept rather than re-read from the DOM.
        for (const card of order) grid.appendChild(card);
        allCards = order;
    }

    function setupClearFilter() {
        const button = document.getElementById('clear-filter');
        if (!button) return;
        button.addEventListener('click', () => {
            currentFilter = 'all';
            searchTerm = '';
            const search = document.getElementById('search-input');
            if (search) search.value = '';
            document.querySelectorAll('.tab[data-filter]').forEach((tab) => {
                tab.classList.toggle('active', tab.dataset.filter === 'all');
            });
            clearFlagFilter();
            applyFilters();
        });
    }

    function applyFilters() {
        filteredCards = allCards.filter(card => {
            const cardType = card.dataset.type;
            const cardName = card.dataset.name.toLowerCase();

            const cardFlags = (card.dataset.flags || '').split(' ');

            const typeMatch = currentFilter === 'all' || cardType === currentFilter;
            const searchMatch = !searchTerm || cardName.includes(searchTerm);
            const flagMatch = !currentFlag || cardFlags.indexOf(currentFlag) !== -1;

            return typeMatch && searchMatch && flagMatch;
        });

        updateDisplay();
    }

    /* Three states, not two.
     *
     * A card is *out of the filter* (removed from the flow), *collapsed* (in
     * the document, header only), or *expanded*. Only the first uses
     * `display: none`, and only for cards the reader has actively filtered
     * away -- which is the one case where not finding them is the intent.
     */
    function updateDisplay() {
        let shown = 0;
        let collapsed = 0;

        allCards.forEach(card => {
            if (!filteredCards.includes(card)) {
                card.hidden = true;
                card.classList.remove('is-collapsed');
                card.removeAttribute('tabindex');
                card.removeAttribute('aria-expanded');
                return;
            }
            card.hidden = false;
            shown += 1;
            const fold = !expandAll
                && !opened.has(card.id)
                && shown > EXPANDED_LIMIT;
            card.classList.toggle('is-collapsed', fold);
            if (fold) {
                collapsed += 1;
                card.setAttribute('tabindex', '0');
                card.setAttribute('role', 'button');
            } else {
                card.removeAttribute('tabindex');
                card.removeAttribute('role');
            }
            card.setAttribute('aria-expanded', String(!fold));
        });

        if (shown === 0) showNoResults(); else hideNoResults();
        updateRail(shown, collapsed);
    }

    function updateRail(shown, collapsed) {
        const rail = document.getElementById('collapsed-rail');
        const count = document.getElementById('collapsed-count');
        const all = document.getElementById('expand-all');
        const info = document.getElementById('pagination-info');

        // One line for all three mechanisms. `Showing 1-10 of 12` described a
        // page, and could not say that a search and a type filter were also
        // narrowing the list (design 15c).
        if (info) {
            const filtered = currentFilter !== 'all' || searchTerm || currentFlag;
            const noun = currentFilter === 'all' ? 'columns' : `${currentFilter} columns`;
            info.textContent = shown === 0
                ? 'No columns match'
                : filtered
                    ? `${shown} ${noun} of ${allCards.length} · ` +
                      `${shown - collapsed} expanded, ${collapsed} collapsed`
                    : `${shown} columns · ${shown - collapsed} expanded, ` +
                      `${collapsed} collapsed`;
        }
        const clear = document.getElementById('clear-filter');
        if (clear) {
            clear.hidden = !(currentFilter !== 'all' || searchTerm || currentFlag);
        }
        if (!rail || !count || !all) return;

        // The rail is about collapsed rows, so it says nothing when there are
        // none -- which is every report of ten columns or fewer.
        if (collapsed === 0 && !expandAll) {
            rail.hidden = true;
            return;
        }
        rail.hidden = false;
        const noun = collapsed === 1 ? 'row' : 'rows';
        count.textContent = expandAll
            ? `All ${shown} columns expanded`
            : `${collapsed} collapsed ${noun}`;
        all.textContent = expandAll ? 'collapse again' : `expand all ${collapsed}`;
        all.setAttribute('aria-pressed', String(expandAll));
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
        if (noResults) noResults.remove();
    }

    /* Take a #col_<name> link to the card it names, and open it.
     *
     * Every link in the needs-attention block is one of these, and the block
     * exists to be clicked. A collapsed card is already in the document, so
     * the anchor lands on its own -- what this adds is expanding it, since
     * arriving at a folded header answers less than the reader asked for.
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

        if (!filteredCards.includes(card)) return false;
        opened.add(card.id);
        updateDisplay();
        // After updateDisplay, so the card has its full box to scroll to.
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
