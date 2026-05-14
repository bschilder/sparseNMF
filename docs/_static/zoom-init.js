// medium-zoom initializer.
// Attaches click-to-zoom to every <img> in the main content area.
// Skips the site logo, navigation images, and anything opted out via
// the ``no-zoom`` CSS class.
//
// Why the unusual selector: sphinx-rtd-theme wraps the page content
// in ``<div role="main" class="document">`` (no ``<article>``).
// Other Sphinx themes wrap in ``<main>`` or ``<article>``. We target
// all three so the zoom works across themes without per-theme tweaks.
window.addEventListener("DOMContentLoaded", () => {
  if (typeof mediumZoom !== "function") {
    return; // CDN failed to load; fail open, no zoom but no breakage
  }
  const selector = [
    '[role="main"] img:not(.no-zoom):not(.logo)',
    "main img:not(.no-zoom):not(.logo)",
    "article img:not(.no-zoom):not(.logo)",
  ].join(", ");
  mediumZoom(selector, {
    background: "rgba(13, 17, 23, 0.92)",
    margin: 24,
    scrollOffset: 40,
  });
});
