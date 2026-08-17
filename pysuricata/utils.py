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


def strip_js_comments(source: str) -> str:
    """Drop comments from script text on its way into a report.

    The same argument as :func:`strip_css_comments`, applied to the other half
    of what the report inlines. Measured across `static/js/`: **15,619 bytes of
    comments in 79,251 bytes of script, 20% of it**, shipped to every reader of
    every report. As with the stylesheets, the comments are worth having in the
    source, which is the only place anyone reads them.

    A regex will not do this one. CSS has no construct in which `/*` means
    something else; JavaScript has three, and each of them appears in these
    files:

    * a string containing `//`, which every URL in a comment-free line has;
    * a template literal, which may span lines and contain either marker;
    * a regex literal, where `/` opens a pattern rather than a comment --
      ``/\\/\\*/`` is a valid regex matching the characters ``/*``.

    So this is a scanner, not a substitution. It tracks which of those it is
    inside and only treats `/` as a comment when it is inside none of them.
    Telling a regex literal from a division is the classic ambiguity; it is
    resolved the way every JS lexer resolves it, by looking at the last
    significant token -- after a value, `/` divides; after an operator, a
    keyword or an opening bracket, it opens a pattern.

    Nothing else is touched. Whitespace stays, names stay, semicolons stay:
    this is not a minifier, and a minifier is a far larger promise to keep
    correct than the 20% is worth.
    """
    out: list[str] = []
    i, n = 0, len(source)
    # The last character that decides whether `/` starts a regex or divides.
    last = ""
    while i < n:
        char = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if char == "/" and nxt == "/":
            i = source.find("\n", i)
            if i == -1:
                break
            continue  # leave the newline itself

        if char == "/" and nxt == "*":
            keep = source.startswith("/*!", i)  # the "preserve" convention
            end = source.find("*/", i + 2)
            end = n if end == -1 else end + 2
            if keep:
                out.append(source[i:end])
            elif "\n" in source[i:end]:
                out.append("\n")  # a block comment on its own lines held a break
            i = end
            continue

        if char in "\"'`":
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == char:
                    break
                j += 1
            out.append(source[i : j + 1])
            last = char
            i = j + 1
            continue

        if char == "/" and _js_slash_starts_a_regex(last):
            j, in_class = i + 1, False
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == "[":
                    in_class = True
                elif source[j] == "]":
                    in_class = False
                elif source[j] == "/" and not in_class:
                    break
                elif source[j] == "\n":
                    break  # unterminated; treat as division after all
                j += 1
            out.append(source[i : j + 1])
            last = "/"
            i = j + 1
            continue

        out.append(char)
        if not char.isspace():
            last = char
        i += 1

    return _BLANK_RUN.sub("\n\n", "".join(out))


def _js_slash_starts_a_regex(last: str) -> bool:
    """Whether a `/` following ``last`` opens a pattern rather than divides.

    The lexer's rule, reduced to the single character before: division can only
    follow something that produced a value -- a name, a number, a closing
    bracket, a string. Everything else (an operator, a comma, an opening
    bracket, the start of the file) is followed by a regex if it is followed by
    a slash at all.
    """
    if last == "":
        return True
    return not (last.isalnum() or last in "_$)]}\"'`")


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
            return strip_js_comments(f.read())
    return ""
