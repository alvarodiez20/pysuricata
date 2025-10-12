"""Markdown rendering utilities with lazy loading."""

from __future__ import annotations


def render_markdown_to_html(markdown_text: str) -> str:
    """
    Convert markdown to HTML with lazy loading.
    Falls back to simple line-break conversion if markdown not available.
    """
    if not markdown_text or not markdown_text.strip():
        return ""
    try:
        import markdown

        # Use safe extensions: tables, fenced_code, nl2br
        return markdown.markdown(
            markdown_text,
            extensions=["tables", "fenced_code", "nl2br"],
            output_format="html5",
        )
    except ImportError:
        # Fallback: escape HTML and convert newlines to <br>
        import html

        escaped = html.escape(markdown_text)
        return escaped.replace("\n", "<br>")
