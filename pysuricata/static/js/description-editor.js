/**
 * Description Editor with Markdown Support
 * Handles interactive editing and markdown rendering
 */
(function () {
  'use strict';

  const ROOT_ID = 'pysuricata-report';
  const STORAGE_KEY_PREFIX = 'pysuricata-description-';
  const PLACEHOLDER_TEXT = 'Click to add description...';

  // ===== Private Helper Functions =====

  function getDescriptionContainer() {
    return document.querySelector(`#${ROOT_ID} .description-value`);
  }

  function getStorageKey() {
    const container = getDescriptionContainer();
    const reportId = container?.getAttribute('data-report-id') || 'default';
    return STORAGE_KEY_PREFIX + reportId;
  }

  function getContentElement() {
    const container = getDescriptionContainer();
    return container?.querySelector('.description-content');
  }

  function getMarkdownSource(container) {
    // Get markdown from data attribute or content
    return container?.getAttribute('data-original-markdown') || '';
  }

  function setMarkdownSource(container, markdown) {
    if (container) {
      container.setAttribute('data-original-markdown', markdown);
    }
  }

  // ===== Storage Functions =====

  function saveToStorage(markdownText) {
    try {
      const key = getStorageKey();
      if (markdownText.trim()) {
        localStorage.setItem(key, markdownText);
      } else {
        localStorage.removeItem(key);
      }
    } catch (e) {
      console.warn('Failed to save description:', e);
    }
  }

  function loadFromStorage() {
    try {
      const key = getStorageKey();
      return localStorage.getItem(key) || '';
    } catch (e) {
      console.warn('Failed to load description:', e);
      return '';
    }
  }

  // ===== Rendering Functions =====

  function renderMarkdownToHtml(markdown) {
    // Simple client-side markdown rendering (basic support)
    if (!markdown || !markdown.trim()) {
      return `<span class="placeholder">${PLACEHOLDER_TEXT}</span>`;
    }

    // Escape HTML for security
    let html = markdown
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');

    // Basic markdown patterns
    html = html
      // Headers
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^# (.+)$/gm, '<h1>$1</h1>')
      // Bold and italic
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      // Lists
      .replace(/^- (.+)$/gm, '<li>$1</li>')
      .replace(/^• (.+)$/gm, '<li>$1</li>')
      // Line breaks
      .replace(/\n/g, '<br>');

    // Wrap lists
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

    return html;
  }

  // ===== Init on Page Load =====

  function initializeDescription() {
    const saved = loadFromStorage();
    if (saved) {
      const container = getDescriptionContainer();
      const contentEl = getContentElement();
      if (container && contentEl) {
        setMarkdownSource(container, saved);
        contentEl.innerHTML = renderMarkdownToHtml(saved);
      }
    }
  }

  document.addEventListener('DOMContentLoaded', initializeDescription);
  if (document.readyState !== 'loading') {
    initializeDescription();
  }

  // ===== Public API =====

  window.startDescriptionEdit = function () {
    const container = getDescriptionContainer();
    const contentEl = getContentElement();

    if (!container || !contentEl || container.classList.contains('editing')) {
      return;
    }

    const currentMarkdown = getMarkdownSource(container);

    // Create textarea
    const textarea = document.createElement('textarea');
    textarea.value = currentMarkdown;
    textarea.placeholder = 'Enter description (Markdown supported)...';
    textarea.className = 'description-editor';

    // Enter edit mode
    container.classList.add('editing');
    contentEl.style.display = 'none';
    container.appendChild(textarea);

    textarea.focus();
    textarea.select();

    // Auto-resize
    function autoResize() {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 300) + 'px';
    }

    textarea.addEventListener('input', autoResize);
    textarea.addEventListener('paste', () => setTimeout(autoResize, 10));
    autoResize();

    // Event handlers
    textarea.addEventListener('blur', () => {
      setTimeout(() => {
        if (container.classList.contains('editing')) {
          saveEdit();
        }
      }, 100);
    });

    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        saveEdit();
      }
    });

    function saveEdit() {
      const newMarkdown = textarea.value;

      // Update content
      setMarkdownSource(container, newMarkdown);
      contentEl.innerHTML = renderMarkdownToHtml(newMarkdown);
      contentEl.style.display = '';

      // Save to storage
      saveToStorage(newMarkdown);

      // Clean up
      textarea.remove();
      container.classList.remove('editing');
    }
  };

  window.getCurrentDescription = function () {
    const container = getDescriptionContainer();
    const markdown = getMarkdownSource(container);
    // Return markdown for editing, HTML for download
    return {
      markdown: markdown,
      html: renderMarkdownToHtml(markdown)
    };
  };

})();
