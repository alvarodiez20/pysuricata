/* A 90-line stand-in for the authoring runtime these figures were built with.
 *
 * The original pulls React, ReactDOM and Babel-standalone from unpkg at page
 * load -- about 2 MB of CDN JavaScript, in-browser JSX transpilation, and three
 * cross-origin requests -- to render six figures whose drawing code is entirely
 * vanilla SVG DOM manipulation. The only React surface the figure logic touches
 * is createRef, setState, props and the four lifecycle hooks, so that is all
 * this provides. The figure logic itself is unmodified.
 *
 * Template contract, unchanged from the original markup:
 *   ref="{{ name }}"        binds the element to bindings.name.current
 *   onClick="{{ name }}"    binds a click handler
 *   onInput="{{ name }}"    binds an input handler
 *   >{{ name }}<            substitutes text, re-evaluated on every setState
 */
(function (global) {
  "use strict";

  const BINDING = /^\{\{\s*([A-Za-z0-9_$]+)\s*\}\}$/;

  class DCLogic {
    constructor(props) {
      this.props = props || {};
      this.state = {};
    }
    setState(patch) {
      const next = typeof patch === "function" ? patch(this.state) : patch;
      Object.assign(this.state, next);
      if (this._host) this._host.sync();
    }
  }

  function createRef() {
    return { current: null };
  }

  class Host {
    constructor(root, ComponentClass, props) {
      this.root = root;
      this.textNodes = [];
      this.component = new ComponentClass(props || {});
      this.component._host = this;
    }

    mount() {
      const c = this.component;
      // renderVals() is the figure's binding map. Read it once to wire refs and
      // handlers, then re-read it on each setState for the text bindings.
      const vals = c.renderVals();

      this.root.querySelectorAll("*").forEach((el) => {
        for (const attr of Array.from(el.attributes)) {
          const m = BINDING.exec(attr.value.trim());
          if (!m) continue;
          const value = vals[m[1]];
          if (attr.name === "ref") {
            if (value) value.current = el;
            el.removeAttribute("ref");
          } else if (attr.name.startsWith("on")) {
            const event = attr.name.slice(2).toLowerCase();
            if (typeof value === "function") el.addEventListener(event, value);
            el.removeAttribute(attr.name);
          }
        }
      });

      // Text bindings live in their own text nodes; remember them so setState
      // can refresh the labels without touching anything else.
      const walker = document.createTreeWalker(this.root, NodeFilter.SHOW_TEXT);
      for (let n = walker.nextNode(); n; n = walker.nextNode()) {
        const m = BINDING.exec(n.nodeValue.trim());
        if (m) this.textNodes.push([n, m[1]]);
      }

      this.sync();
      if (c.componentDidMount) c.componentDidMount();
      this.mounted = true;
      global.addEventListener("pagehide", () => {
        if (c.componentWillUnmount) c.componentWillUnmount();
      });
    }

    sync() {
      const c = this.component;
      const vals = c.renderVals();
      for (const [node, key] of this.textNodes) {
        const v = vals[key];
        if (v != null) node.nodeValue = String(v);
      }
      // componentDidUpdate re-applies the theme; it must not run before mount,
      // which is what sets the refs it reads.
      if (this.mounted && c.componentDidUpdate) c.componentDidUpdate();
    }
  }

  global.React = global.React || { createRef };
  global.DCLogic = DCLogic;
  global.mountDC = function (root, ComponentClass, props) {
    const host = new Host(root, ComponentClass, props);
    host.mount();
    return host;
  };
})(window);
