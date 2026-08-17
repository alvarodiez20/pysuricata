import base64
import glob
import os
import re
from functools import lru_cache

#: A CSS comment, non-greedy so it ends at the first `*/`. CSS comments do not
#: nest, so a `/*` inside a comment body is part of that comment and not the
#: start of another -- which is why `_13-utilities.css` counts 43 openers to 42
#: closers and is nonetheless well-formed.
#:
#: `/*!` is the convention for "keep this even when minifying", so it is left
#: alone. Nothing in the stylesheets uses it today; honouring it costs one
#: character and means a licence header added later survives.
_CSS_COMMENT = re.compile(r"/\*(?!!).*?\*/", re.S)

#: Runs of blank lines left behind once a comment between two rules is gone.
_BLANK_RUN = re.compile(r"\n\s*\n(\s*\n)+")


def strip_css_comments(css: str) -> str:
    """Drop comments from stylesheet text on its way into a report.

    The report inlines its own CSS, so every comment in `static/css/` was being
    shipped to every reader: **545 comments, 74,036 bytes -- 33% of the inlined
    stylesheet and 12.9% of the whole Titanic report.** The comments are worth
    having, and this is not an argument for deleting them; they are worth
    having *in the source*, which is the only place anybody reads them.

    Deliberately only comments and the blank runs they leave. Collapsing
    whitespace or rewriting values is a minifier, and a minifier is a much
    larger promise to keep correct -- `content` strings and `url()` payloads
    both have rules a naive pass gets wrong. There are none in these
    stylesheets today, and this stays safe if one appears tomorrow.
    """
    return _BLANK_RUN.sub("\n\n", _CSS_COMMENT.sub("", css))


def load_template(template_path: str) -> str:
    """
    Load an HTML template from a file.

    Args:
        template_path (str): The file path to the HTML template.

    Returns:
        str: The content of the HTML template.
    """
    with open(template_path, encoding="utf-8") as f:
        return f.read()


def load_css(css_path: str) -> str:
    """
    Load a CSS file and return its content wrapped in a <style> tag.
    Optimized for performance - no @import resolution needed.

    Args:
        css_path (str): The file path to the CSS file.

    Returns:
        str: A string with the CSS content wrapped in a <style> tag, or an empty string if the file is not found.
    """
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            css_content = f.read()
        return f"<style>{strip_css_comments(css_content)}</style>"
    return ""


@lru_cache(maxsize=4)
def load_css_dir(css_dir: str) -> str:
    """Read _*.css partials from a directory, concatenate in sorted order, wrap in <style>.

    Result is cached so repeated calls (e.g. generating multiple reports) pay no I/O cost.

    Args:
        css_dir: Path to the directory containing _*.css partial files.

    Returns:
        A string with the concatenated CSS wrapped in a <style> tag,
        or an empty string if no partials are found.
    """
    parts = []
    for path in sorted(glob.glob(os.path.join(css_dir, "_*.css"))):
        with open(path, encoding="utf-8") as f:
            parts.append(f.read())
    if not parts:
        return ""
    return f"<style>{strip_css_comments(''.join(parts))}</style>"


def embed_image(
    image_path: str, element_id: str, alt_text: str = "", mime_type: str = "image/png"
) -> str:
    """
    Embed an image into an HTML <img> tag using Base64 encoding.

    Args:
        image_path (str): The file path to the image.
        element_id (str): The HTML id attribute for the image.
        alt_text (str): Alternate text for the image.
        mime_type (str): MIME type of the image (default "image/png").

    Returns:
        str: An HTML <img> tag containing the embedded Base64 image.
             Returns an empty string if the image file does not exist.
    """
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return f'<img id="{element_id}" src="data:{mime_type};base64,{encoded}" alt="{alt_text}">'
    return ""


def embed_favicon(favicon_path: str) -> str:
    """
    Embed a favicon into an HTML <link> tag using Base64 encoding.

    Args:
        favicon_path (str): The file path to the favicon image.

    Returns:
        str: An HTML <link> tag containing the embedded favicon.
             Returns an empty string if the favicon file does not exist.
    """
    if os.path.exists(favicon_path):
        with open(favicon_path, "rb") as icon_file:
            encoded = base64.b64encode(icon_file.read()).decode("utf-8")
        return f'<link rel="icon" href="data:image/x-icon;base64,{encoded}" type="image/x-icon">'
    return ""


def load_script(script_path: str) -> str:
    """
    Load a JavaScript file and return its content.

    Args:
        script_path (str): The file path to the JavaScript file.

    Returns:
        str: The JavaScript content as a string, or an empty string if the file is not found.
    """
    if os.path.exists(script_path):
        with open(script_path, encoding="utf-8") as f:
            return f.read()
    return ""
