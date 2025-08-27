import os
import base64
import pandas as pd
import numpy as np
from typing import Union, Optional, List




def load_template(template_path: str) -> str:
    """
    Load an HTML template from a file.

    Args:
        template_path (str): The file path to the HTML template.

    Returns:
        str: The content of the HTML template.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def load_css(css_path: str) -> str:
    """
    Load a CSS file and return its content wrapped in a <style> tag.

    Args:
        css_path (str): The file path to the CSS file.

    Returns:
        str: A string with the CSS content wrapped in a <style> tag, or an empty string if the file is not found.
    """
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        return f"<style>{css_content}</style>"
    return ""


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
        with open(script_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""
