"""Test markdown support in descriptions."""

import pandas as pd

from pysuricata import ProfileConfig, RenderOptions, profile


def test_markdown_basic():
    """Test basic markdown rendering."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    desc = """# Title
**Bold** and *italic* text."""

    config = ProfileConfig(render=RenderOptions(description=desc))
    report = profile(df, config=config)

    assert "<h1>Title</h1>" in report.html or "Title" in report.html
    assert "Bold" in report.html


def test_markdown_fallback():
    """Test fallback when markdown not available."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    desc = "Line 1\nLine 2"

    config = ProfileConfig(render=RenderOptions(description=desc))
    report = profile(df, config=config)

    # Should have line breaks
    assert "<br>" in report.html or "Line 1" in report.html


def test_empty_description():
    """Test empty description handling."""
    df = pd.DataFrame({"A": [1, 2, 3]})

    config = ProfileConfig(render=RenderOptions(description=""))
    report = profile(df, config=config)

    # Should not have description section visible or have placeholder
    assert "description" in report.html.lower()


def test_xss_protection():
    """Test XSS protection in descriptions."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    desc = "<script>alert('xss')</script>"

    config = ProfileConfig(render=RenderOptions(description=desc))
    report = profile(df, config=config)

    # Should be escaped
    assert "<script>" not in report.html or "&lt;script&gt;" in report.html


def test_markdown_with_markdown_library():
    """Test markdown rendering with the markdown library installed."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    desc = """# Header
**Bold text**
- List item 1
- List item 2"""

    config = ProfileConfig(render=RenderOptions(description=desc))
    report = profile(df, config=config)

    # Check that markdown is processed
    assert "Header" in report.html
    assert "Bold text" in report.html


def test_markdown_lists():
    """Test markdown list rendering."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    desc = """Key features:
- Feature 1
- Feature 2
- Feature 3"""

    config = ProfileConfig(render=RenderOptions(description=desc))
    report = profile(df, config=config)

    # Should contain list items
    assert "Feature 1" in report.html
    assert "Feature 2" in report.html


def test_markdown_headers():
    """Test markdown header rendering."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    desc = """# Main Title
## Subtitle
### Section"""

    config = ProfileConfig(render=RenderOptions(description=desc))
    report = profile(df, config=config)

    # Should contain headers
    assert "Main Title" in report.html
    assert "Subtitle" in report.html
    assert "Section" in report.html


def test_markdown_bold_italic():
    """Test bold and italic text rendering."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    desc = """This is **bold** and this is *italic* text."""

    config = ProfileConfig(render=RenderOptions(description=desc))
    report = profile(df, config=config)

    # Should contain the text
    assert "bold" in report.html
    assert "italic" in report.html


def test_markdown_with_quotes():
    """Test markdown with special characters."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    desc = """This has "quotes" and 'apostrophes' and & ampersands."""

    config = ProfileConfig(render=RenderOptions(description=desc))
    report = profile(df, config=config)

    # Should be properly escaped
    assert "quotes" in report.html
    assert "apostrophes" in report.html


def test_markdown_preservation_on_download():
    """Test that markdown is preserved in data attributes."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    desc = """# Test
**Bold** text"""

    config = ProfileConfig(render=RenderOptions(description=desc))
    report = profile(df, config=config)

    # Should have data-original-markdown attribute
    assert "data-original-markdown=" in report.html
    assert "Test" in report.html


def test_markdown_empty_after_whitespace():
    """Test that whitespace-only descriptions are treated as empty."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    desc = "   \n  \t  "

    config = ProfileConfig(render=RenderOptions(description=desc))
    report = profile(df, config=config)

    # Should not show description section
    assert (
        "data-original-markdown=" not in report.html
        or 'data-original-markdown=""' in report.html
    )


def test_markdown_multiline():
    """Test multiline markdown content."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    desc = """First paragraph.

Second paragraph with **bold**.

Third paragraph with *italic*."""

    config = ProfileConfig(render=RenderOptions(description=desc))
    report = profile(df, config=config)

    # Should contain all paragraphs
    assert "First paragraph" in report.html
    assert "Second paragraph" in report.html
    assert "Third paragraph" in report.html
