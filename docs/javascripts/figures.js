/* Sizes the interactive-figure iframes to their content and keeps them on the
 * reader's colour scheme.
 *
 * The figures live in assets/diagrams/figures.html and post their height back
 * as they draw. Without this the page would need a fixed iframe height, which
 * means either a scrollbar inside the figure or dead space under it.
 */
(function () {
  "use strict";

  var FIGURE_SRC = /assets\/diagrams\/figures\.html/;

  function figureFrames() {
    return Array.prototype.filter.call(
      document.querySelectorAll("iframe"),
      function (f) {
        return FIGURE_SRC.test(f.getAttribute("src") || "");
      },
    );
  }

  window.addEventListener("message", function (event) {
    var data = event.data;
    if (!data || typeof data.pysuricataFigureHeight !== "number") return;
    figureFrames().forEach(function (frame) {
      if (frame.contentWindow === event.source) {
        frame.style.height = data.pysuricataFigureHeight + "px";
      }
    });
  });

  // mkdocs-material swaps the palette without a reload, so tell the figures to
  // follow. They read the same localStorage key on load; this covers the case
  // where the reader toggles while a figure is already on screen.
  function currentScheme() {
    var body = document.body;
    return body && body.getAttribute("data-md-color-scheme") === "slate"
      ? "dark"
      : "light";
  }

  var lastScheme = null;
  function syncScheme() {
    var scheme = currentScheme();
    if (scheme === lastScheme) return;
    lastScheme = scheme;
    figureFrames().forEach(function (frame) {
      var src = frame.getAttribute("src").split("&theme=")[0];
      frame.setAttribute("src", src + "&theme=" + scheme);
    });
  }

  function start() {
    lastScheme = currentScheme();
    new MutationObserver(syncScheme).observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
