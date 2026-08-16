/**
 * Description Editor with Markdown Support
 * Handles interactive editing and markdown rendering
 */
(function () {
  'use strict';

  const ROOT_ID = 'pysuricata-report';
  const STORAGE_KEY_PREFIX = 'pysuricata-description-';

  // ===== Private Helper Functions =====

  // Located by id, not by class. The redesign renamed this block's class from
  // `.description-value` to `.description-block`, and because every entry point
  // here fails soft on a null container, the rename turned "+ add a note" into
  // a button that did nothing at all -- silently, with no console error. The id
  // is what the template guarantees; the class is presentation and may move
  // again.
  function getDescriptionContainer() {
    return (
      document.getElementById('summary-description') ||
      document.querySelector(`#${ROOT_ID} .description-block`)
    );
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
    // Empty renders as empty: the row itself carries the invitation, and
    // `.is-empty` hides this element, so a placeholder here would be a string
    // nobody can ever see.
    if (!markdown || !markdown.trim()) {
      return '';
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

  // ===== State =====

  // The server picks the empty/filled presentation once, at render time. Once
  // the reader edits, the client owns it -- and must move all three parts
  // together. Missing this is not cosmetic: `.is-empty` sets `display: none` on
  // `.description-content`, so a note saved without clearing the class is
  // stored, escaped, inserted, and invisible.
  function applyState(container, markdown) {
    const filled = Boolean(markdown && markdown.trim());
    container.classList.toggle('is-empty', !filled);

    const label = container.querySelector('.description-block__label');
    if (label) label.textContent = filled ? 'Note' : 'Description';

    const action = container.querySelector('.description-block__action');
    if (action) action.textContent = filled ? 'edit' : '+ add a note';
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
        applyState(container, saved);
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
      applyState(container, newMarkdown);
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
