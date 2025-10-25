/**
 * Tooltip functionality for PySuricata quality flags and numeric chips.
 *
 * Provides hover explanations for data quality indicators with comprehensive
 * definitions, accessibility support, and theme-aware styling.
 *
 * @fileoverview Quality flag tooltip system for PySuricata reports
 * @version 1.0.0
 * @author PySuricata Team
 */

(function() {
  'use strict';

  const ROOT_ID = 'pysuricata-report';

  /**
   * Header tooltip definitions for metadata chips.
   */
  const HEADER_TOOLTIP_DEFINITIONS = {
    'date': {
      title: 'Report Generation Time',
      getDescription: (value) => `This report was generated on <strong>${value}</strong>. The timestamp indicates when the profiling analysis was completed.`,
      category: 'metadata'
    },
    'duration': {
      title: 'Processing Duration',
      getDescription: (value) => `Time elapsed to generate this report: <strong>${value}</strong>. This includes data processing, statistical analysis, and report rendering.`,
      category: 'performance'
    },
    'version': {
      title: 'PySuricata Version',
      getDescription: (value) => `Generated with <strong>pysuricata v${value}</strong>. Click to visit the GitHub repository for documentation, updates, and support.`,
      category: 'metadata'
    }
  };

  /**
   * Comprehensive tooltip definitions for all quality flags.
   * Each definition includes title, description, and severity level.
   */
  const TOOLTIP_DEFINITIONS = {
    // Missing data flags
    'missing': {
      title: 'Missing Data',
      description: 'This column contains missing or null values. Critical severity (red) when >20%, warning (orange) when ≤20%. High percentages may indicate data quality issues that need attention.',
      severity: 'warn',
      category: 'data-quality'
    },

    // Infinite values
    'has-∞': {
      title: 'Infinite Values',
      description: 'This column contains infinite values (∞ or -∞). Critical severity triggered by any infinite value. This typically indicates calculation overflow, division by zero, or data corruption.',
      severity: 'bad',
      category: 'data-quality'
    },

    // Negative values
    'has-negatives': {
      title: 'Negative Values',
      description: 'This column contains negative values. Warning severity when >10% negative. Consider if this is expected for your data type (e.g., temperatures, deltas) and analysis requirements.',
      severity: 'info',
      category: 'data-characteristics'
    },

    // Zero inflation
    'zero‑inflated': {
      title: 'Zero Inflation',
      description: 'This column has a high percentage of zero values. Critical severity (red) when ≥50%, warning (orange) when significant but <50%. May indicate sparse dataset, special zero category, or data quality issues.',
      severity: 'warn',
      category: 'distribution'
    },

    // Positive only
    'positive‑only': {
      title: 'Positive Only',
      description: 'All values in this column are positive. This is beneficial for certain types of analysis and transformations.',
      severity: 'good',
      category: 'data-characteristics'
    },

    // Skewness indicators
    'skewed-right': {
      title: 'Right Skewed',
      description: 'The data is skewed to the right (positive skew, typically >1). Mean > median, with a long right tail. Consider log transformation or other methods to normalize the distribution for analysis.',
      severity: 'warn',
      category: 'distribution'
    },
    'skewed-left': {
      title: 'Left Skewed',
      description: 'The data is skewed to the left (negative skew, typically <-1). Mean < median, with a long left tail. Consider transformation methods to normalize the distribution for analysis.',
      severity: 'warn',
      category: 'distribution'
    },

    // Heavy tails
    'heavy‑tailed': {
      title: 'Heavy Tailed',
      description: 'The data has heavy tails (high kurtosis, typically |kurtosis| > 3). More extreme values than a normal distribution. This may indicate outliers or non-normal distribution that could affect statistical tests.',
      severity: 'bad',
      category: 'distribution'
    },

    // Normal distribution
    '≈-normal-(jb)': {
      title: 'Approximately Normal',
      description: 'The data appears to follow a normal distribution based on the Jarque-Bera test (JB χ² < 5.99 at 5% significance). This is ideal for many parametric statistical analyses.',
      severity: 'good',
      category: 'distribution'
    },

    // Discrete data
    'discrete': {
      title: 'Discrete Data',
      description: 'This column contains discrete (integer-like) values with low unique count relative to sample size. May behave more like categorical than continuous. Consider appropriate statistical methods for discrete data.',
      severity: 'warn',
      category: 'data-type'
    },

    // Heaping patterns
    'heaping': {
      title: 'Heaping',
      description: 'The data shows heaping patterns, where values cluster at round numbers (detected via digit analysis). This may indicate measurement precision issues, rounding in data collection, or self-reporting.',
      severity: 'info',
      category: 'data-quality'
    },

    // Bimodal distribution
    'possibly-bimodal': {
      title: 'Possibly Bimodal',
      description: 'The data may have two distinct peaks, suggesting multiple underlying distributions or subpopulations in your data.',
      severity: 'warn',
      category: 'distribution'
    },

    // Log scale suggestion
    'log-scale-suggested': {
      title: 'Log Scale Suggested',
      description: 'The data distribution suggests that a logarithmic scale might be more appropriate for visualization and analysis.',
      severity: 'info',
      category: 'visualization'
    },

    // High cardinality
    'high-cardinality': {
      title: 'High Cardinality',
      description: 'This column has many unique values relative to its size. Consider if this is expected and whether it affects your analysis.',
      severity: 'warn',
      category: 'data-characteristics'
    },

    // Dominant category
    'dominant-category': {
      title: 'Dominant Category',
      description: 'One category dominates the data, which may indicate imbalanced classes. Consider stratified sampling for analysis.',
      severity: 'warn',
      category: 'distribution'
    },

    // Many rare levels
    'many-rare-levels': {
      title: 'Many Rare Levels',
      description: 'This column has many categories with very few occurrences. Consider grouping rare categories or using appropriate statistical methods.',
      severity: 'warn',
      category: 'data-characteristics'
    },

    // Case variants
    'case-variants': {
      title: 'Case Variants',
      description: 'This column has values that differ only in case (e.g., "Apple" vs "apple"). Consider standardizing case for consistency.',
      severity: 'info',
      category: 'data-quality'
    },

    // Trim variants
    'trim-variants': {
      title: 'Trim Variants',
      description: 'This column has values that differ only in leading/trailing whitespace. Consider trimming whitespace for consistency.',
      severity: 'info',
      category: 'data-quality'
    },

    // Empty strings
    'empty-strings': {
      title: 'Empty Strings',
      description: 'This column contains empty string values, which may be different from missing values. Consider how to handle these in your analysis.',
      severity: 'info',
      category: 'data-quality'
    },

    // Many outliers
    'many-outliers': {
      title: 'Many Outliers',
      description: 'This column has a high number of outliers (>1%) that deviate significantly from the main distribution using IQR method. Critical severity. Consider investigating these values and their impact on your analysis.',
      severity: 'bad',
      category: 'distribution'
    },

    // Some outliers
    'some-outliers': {
      title: 'Some Outliers',
      description: 'This column contains some outliers (0.3% to 1%) detected using IQR or MAD methods. Warning severity. Review these values to ensure they are valid and consider their impact on statistical analysis.',
      severity: 'warn',
      category: 'distribution'
    },

    // Constant values
    'constant': {
      title: 'Constant Values',
      description: 'All values in this column are identical (only 1 unique value). This column provides no variation and may not be useful for analysis. Critical severity.',
      severity: 'bad',
      category: 'data-characteristics'
    },

    // Quasi-constant values
    'quasi-constant': {
      title: 'Quasi-Constant',
      description: 'This column has very little variation, with one value dominating (very low cardinality). Warning severity. Consider if this column is useful for analysis.',
      severity: 'warn',
      category: 'data-characteristics'
    },

    // Monotonic increasing
    'monotonic-↑': {
      title: 'Monotonic Increasing',
      description: 'Values in this column are strictly increasing. This may indicate a time series or ordered sequence.',
      severity: 'good',
      category: 'data-characteristics'
    },

    // Monotonic decreasing
    'monotonic-↓': {
      title: 'Monotonic Decreasing',
      description: 'Values in this column are strictly decreasing. This may indicate a time series or ordered sequence.',
      severity: 'good',
      category: 'data-characteristics'
    },

    // Log scale suggested
    'log-scale?': {
      title: 'Log Scale Suggested',
      description: 'The data distribution suggests that a logarithmic scale might be more appropriate for visualization and analysis.',
      severity: 'info',
      category: 'visualization'
    }
  };

  /**
   * Tooltip manager class for handling tooltip creation, positioning, and display.
   */
  class TooltipManager {
    constructor() {
      this.tooltip = null;
      this.currentElement = null;
      this.isVisible = false;
      this.init();
    }

    /**
     * Initialize the tooltip manager.
     */
    init() {
      this.createTooltip();
      this.bindEvents();
    }

    /**
     * Create the tooltip DOM element.
     */
    createTooltip() {
      const root = document.getElementById(ROOT_ID);
      if (!root) return;

      this.tooltip = document.createElement('div');
      this.tooltip.className = 'quality-tooltip';
      this.tooltip.setAttribute('role', 'tooltip');
      this.tooltip.setAttribute('aria-hidden', 'true');
      this.tooltip.setAttribute('aria-live', 'polite');
      root.appendChild(this.tooltip);
    }

    /**
     * Get tooltip content for a flag.
     * @param {string} flagText - The text content of the flag
     * @returns {Object} Tooltip content object
     */
    getTooltipContent(flagText) {
      // Normalize flag text for lookup
      const normalizedText = flagText.toLowerCase()
        .replace(/\s+/g, '-')
        .replace(/[^\w‑-]/g, ''); // Keep special dash character (‑) and regular dash (-)

      return TOOLTIP_DEFINITIONS[normalizedText] || {
        title: 'Quality Flag',
        description: 'This is a data quality indicator for this column.',
        severity: 'info',
        category: 'general'
      };
    }

    /**
     * Show tooltip for an element.
     * @param {HTMLElement} element - The element to show tooltip for
     * @param {string} flagText - The flag text
     * @param {Object} headerTooltipData - Optional header tooltip data
     */
    showTooltip(element, flagText, headerTooltipData = null) {
      if (!this.tooltip || !element) return;

      let content;
      let isHeaderTooltip = false;
      let isBarFillTooltip = false;

      if (headerTooltipData) {
        // Header tooltip
        isHeaderTooltip = true;
        const def = HEADER_TOOLTIP_DEFINITIONS[headerTooltipData.type];
        content = {
          title: def.title,
          description: def.getDescription(headerTooltipData.value),
          category: def.category
        };
      } else if (element.classList.contains('bar-fill')) {
        // Bar-fill tooltip (simple text display)
        isBarFillTooltip = true;
        content = {
          title: flagText,
          description: '',
          category: 'data-completeness'
        };
      } else {
        // Quality flag tooltip
        content = this.getTooltipContent(flagText);

        // Read threshold and value from data attributes
        const threshold = element.getAttribute('data-threshold');
        const value = element.getAttribute('data-value');

        if (threshold && value) {
          content.threshold = threshold;
          content.value = value;
        }
      }

      this.currentElement = element;

      // Build tooltip HTML
      if (isHeaderTooltip) {
        this.tooltip.innerHTML = `
          <div class="tooltip-header">
            <span class="tooltip-title">${content.title}</span>
          </div>
          <div class="tooltip-description">${content.description}</div>
          <div class="tooltip-footer">
            <span class="tooltip-category">📋 ${this.escapeHtml(content.category)}</span>
          </div>
        `;
      } else if (isBarFillTooltip) {
        // Simple tooltip for bar-fill elements
        this.tooltip.innerHTML = `
          <div class="tooltip-header">
            <span class="tooltip-title">${this.escapeHtml(content.title)}</span>
          </div>
        `;
      } else {
        // Build threshold info HTML if available
        let thresholdHtml = '';
        if (content.threshold && content.value) {
          thresholdHtml = `
            <div class="tooltip-threshold">
              <span class="threshold-label">Threshold:</span> <strong>${this.escapeHtml(content.threshold)}</strong>
              <span class="threshold-separator">|</span>
              <span class="value-label">Current:</span> <strong>${this.escapeHtml(content.value)}</strong>
            </div>
          `;
        }

        this.tooltip.innerHTML = `
          <div class="tooltip-header">
            <span class="tooltip-title">${this.escapeHtml(content.title)}</span>
            <span class="tooltip-severity ${content.severity}" aria-label="Severity: ${content.severity}">
              ${content.severity}
            </span>
          </div>
          ${thresholdHtml}
          <div class="tooltip-description">${this.escapeHtml(content.description)}</div>
          <div class="tooltip-category">Category: ${this.escapeHtml(content.category)}</div>
        `;
      }

      // Position tooltip
      this.positionTooltip(element);

      // Show tooltip
      this.tooltip.style.display = 'block';
      this.tooltip.setAttribute('aria-hidden', 'false');
      this.isVisible = true;

      // Announce to screen readers
      this.announceTooltip(content.title, content.description);
    }

    /**
     * Hide the tooltip.
     */
    hideTooltip() {
      if (!this.tooltip || !this.isVisible) return;

      this.tooltip.style.display = 'none';
      this.tooltip.setAttribute('aria-hidden', 'true');
      this.isVisible = false;
      this.currentElement = null;
    }

    /**
     * Position tooltip relative to element.
     * @param {HTMLElement} element - The target element
     */
    positionTooltip(element) {
      const root = document.getElementById(ROOT_ID);
      if (!root || !this.tooltip) return;

      const rect = element.getBoundingClientRect();
      const rootRect = root.getBoundingClientRect();
      const tooltipRect = this.tooltip.getBoundingClientRect();

      // Calculate initial position (centered below element)
      let left = rect.left - rootRect.left + (rect.width / 2) - (tooltipRect.width / 2);
      let top = rect.bottom - rootRect.top + 10;

      // Adjust if tooltip would go off screen
      const margin = 10;
      const maxLeft = rootRect.width - tooltipRect.width - margin;
      const maxTop = rootRect.height - tooltipRect.height - margin;

      if (left < margin) {
        left = margin;
      } else if (left > maxLeft) {
        left = maxLeft;
      }

      if (top > maxTop) {
        // Position above element if no space below
        top = rect.top - rootRect.top - tooltipRect.height - 10;
      } else if (top < margin) {
        top = margin;
      }

      this.tooltip.style.left = left + 'px';
      this.tooltip.style.top = top + 'px';
    }

    /**
     * Escape HTML to prevent XSS.
     * @param {string} text - Text to escape
     * @returns {string} Escaped HTML
     */
    escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    /**
     * Announce tooltip content to screen readers.
     * @param {string} title - Tooltip title
     * @param {string} description - Tooltip description
     */
    announceTooltip(title, description) {
      // Create a temporary element for screen reader announcement
      const announcement = document.createElement('div');
      announcement.setAttribute('aria-live', 'polite');
      announcement.setAttribute('aria-atomic', 'true');
      announcement.className = 'sr-only';
      announcement.textContent = `${title}: ${description}`;

      const root = document.getElementById(ROOT_ID);
      if (root) {
        root.appendChild(announcement);
        // Remove after announcement
        setTimeout(() => {
          if (announcement.parentNode) {
            announcement.parentNode.removeChild(announcement);
          }
        }, 1000);
      }
    }

    /**
     * Bind event listeners.
     */
    bindEvents() {
      const root = document.getElementById(ROOT_ID);
      if (!root) return;

      // Handle mouse enter on quality flags
      root.addEventListener('mouseenter', (e) => {
        const flag = e.target.closest('.quality-flags .flag');
        if (flag) {
          const flagText = flag.textContent.trim();
          this.showTooltip(flag, flagText);
          return;
        }

        // Handle bar-fill tooltips
        const barFill = e.target.closest('.completeness-bar-dual .bar-fill');
        if (barFill) {
          const tooltipText = barFill.getAttribute('title');
          if (tooltipText) {
            // Remove title to prevent native tooltip
            barFill.removeAttribute('title');
            barFill.setAttribute('data-original-title', tooltipText);
            this.showTooltip(barFill, tooltipText);
          }
          return;
        }

        // Handle header tooltips
        const headerChip = e.target.closest('.header-tooltip');
        if (headerChip) {
          const tooltipType = headerChip.getAttribute('data-tooltip-type');
          const tooltipValue = headerChip.getAttribute('data-tooltip-value');
          if (tooltipType && tooltipValue) {
            this.showTooltip(headerChip, null, { type: tooltipType, value: tooltipValue });
          }
        }
      }, true);

      // Handle mouse leave on quality flags
      root.addEventListener('mouseleave', (e) => {
        const flag = e.target.closest('.quality-flags .flag');
        if (flag) {
          this.hideTooltip();
          return;
        }

        const barFill = e.target.closest('.completeness-bar-dual .bar-fill');
        if (barFill) {
          const originalTitle = barFill.getAttribute('data-original-title');
          if (originalTitle) {
            barFill.setAttribute('title', originalTitle);
          }
          this.hideTooltip();
          return;
        }

        const headerChip = e.target.closest('.header-tooltip');
        if (headerChip) {
          this.hideTooltip();
        }
      }, true);

      // Hide tooltip when mouse leaves the root element
      root.addEventListener('mouseleave', (e) => {
        if (!root.contains(e.relatedTarget)) {
          this.hideTooltip();
        }
      }, true);

      // Handle keyboard navigation
      root.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && this.isVisible) {
          this.hideTooltip();
        }
      }, true);

      // Handle focus events for accessibility
      root.addEventListener('focusin', (e) => {
        const flag = e.target.closest('.quality-flags .flag');
        if (flag) {
          const flagText = flag.textContent.trim();
          this.showTooltip(flag, flagText);
          return;
        }

        // Handle header tooltips on focus
        const headerChip = e.target.closest('.header-tooltip');
        if (headerChip) {
          const tooltipType = headerChip.getAttribute('data-tooltip-type');
          const tooltipValue = headerChip.getAttribute('data-tooltip-value');
          if (tooltipType && tooltipValue) {
            this.showTooltip(headerChip, null, { type: tooltipType, value: tooltipValue });
          }
        }
      }, true);

      root.addEventListener('focusout', (e) => {
        const flag = e.target.closest('.quality-flags .flag');
        if (flag && !root.contains(e.relatedTarget)) {
          this.hideTooltip();
          return;
        }

        const headerChip = e.target.closest('.header-tooltip');
        if (headerChip && !root.contains(e.relatedTarget)) {
          this.hideTooltip();
        }
      }, true);
    }
  }

  /**
   * Align completeness bars to equal length based on longest column name.
   */
  function alignCompletenessBars() {
    const rows = document.querySelectorAll('.compact-row');
    if (rows.length === 0) return;

    // Find the maximum width of column names
    let maxWidth = 0;
    rows.forEach(row => {
      const colName = row.querySelector('.col-name');
      if (colName) {
        // Temporarily remove max-width to measure full content
        const originalMaxWidth = colName.style.maxWidth;
        colName.style.maxWidth = 'none';
        const width = colName.getBoundingClientRect().width;
        colName.style.maxWidth = originalMaxWidth;
        maxWidth = Math.max(maxWidth, width);
      }
    });

    // Apply the maximum width to all column names
    rows.forEach(row => {
      const colName = row.querySelector('.col-name');
      if (colName) {
        colName.style.width = maxWidth + 'px';
        colName.style.maxWidth = maxWidth + 'px';
      }
    });
  }

  /**
   * Initialize tooltips when DOM is ready.
   */
  function initializeTooltips() {
    const root = document.getElementById(ROOT_ID);
    if (!root) return;

    // Create tooltip manager
    const tooltipManager = new TooltipManager();

    // Make tooltip manager globally accessible for debugging
    if (typeof window !== 'undefined') {
      window.pysuricataTooltips = tooltipManager;
    }

    // Align completeness bars
    alignCompletenessBars();
  }

  /**
   * Re-initialize tooltips when new content is added.
   */
  function setupContentObserver() {
    const root = document.getElementById(ROOT_ID);
    if (!root) return;

    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
          // Check if any added nodes contain quality flags or missing values sections
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              if (node.classList && (node.classList.contains('quality-flags') ||
                  node.classList.contains('missing-values-section-redesign'))) {
                // Re-initialize tooltips and align bars for new content
                setTimeout(() => {
                  initializeTooltips();
                  alignCompletenessBars();
                }, 100);
              }
            }
          });
        }
      });
    });

    observer.observe(root, {
      childList: true,
      subtree: true
    });
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initializeTooltips();
      setupContentObserver();
    });
  } else {
    initializeTooltips();
    setupContentObserver();
  }

  // Export for module systems if available
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { TooltipManager, TOOLTIP_DEFINITIONS, HEADER_TOOLTIP_DEFINITIONS };
  }
})();
