import base64


def test_load_template_ok(tmp_path):
    p = tmp_path / "tmpl.html"
    content = "<html>üñïçødë</html>"
    p.write_text(content, encoding="utf-8")
    from pysuricata.utils import load_template

    assert load_template(str(p)) == content


def test_load_template_missing(tmp_path):
    from pysuricata.utils import load_template
    import pytest

    with pytest.raises(FileNotFoundError):
        load_template(str(tmp_path / "missing.html"))


def test_load_css_ok(tmp_path):
    css = "body{color:#123;} /* Ω */"
    p = tmp_path / "style.css"
    p.write_text(css, encoding="utf-8")
    from pysuricata.utils import load_css

    out = load_css(str(p))
    assert out.startswith("<style>") and out.endswith("</style>")
    assert css in out


def test_load_css_missing(tmp_path):
    from pysuricata.utils import load_css

    assert load_css(str(tmp_path / "no.css")) == ""


def test_load_script_ok(tmp_path):
    js = "console.log('hello Ω');"
    p = tmp_path / "code.js"
    p.write_text(js, encoding="utf-8")
    from pysuricata.utils import load_script

    assert load_script(str(p)) == js


def test_load_script_missing(tmp_path):
    from pysuricata.utils import load_script

    assert load_script(str(tmp_path / "missing.js")) == ""


def test_embed_image_ok(tmp_path):
    # Write a tiny PNG header with minimal content
    img_path = tmp_path / "img.png"
    # 1x1 transparent PNG
    data = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAAFtD5kAAAAASUVORK5CYII="
    )
    img_path.write_bytes(data)
    from pysuricata.utils import embed_image

    tag = embed_image(str(img_path), element_id="logo", alt_text="Logo", mime_type="image/png")
    assert tag.startswith('<img id="logo"')
    assert 'alt="Logo"' in tag
    assert 'src="data:image/png;base64,' in tag


def test_embed_image_missing(tmp_path):
    from pysuricata.utils import embed_image

    assert embed_image(str(tmp_path / "missing.png"), element_id="x") == ""


def test_embed_favicon_ok(tmp_path):
    ico = tmp_path / "fav.ico"
    ico.write_bytes(b"\x00\x00\x01\x00")  # minimal bytes; not a real icon but fine for base64
    from pysuricata.utils import embed_favicon

    link = embed_favicon(str(ico))
    assert link.startswith('<link rel="icon"')
    assert 'href="data:image/x-icon;base64,' in link


def test_embed_favicon_missing(tmp_path):
    from pysuricata.utils import embed_favicon

    assert embed_favicon(str(tmp_path / "missing.ico")) == ""

