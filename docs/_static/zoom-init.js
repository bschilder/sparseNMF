// medium-zoom initializer.
// Attaches click-to-zoom to every <img> in the article body. Skips the
// site logo (top-left) and any image explicitly opted out via the
// ``no-zoom`` CSS class.
window.addEventListener("DOMContentLoaded", () => {
  if (typeof mediumZoom !== "function") {
    return; // CDN failed to load; fail open, no zoom but no breakage
  }
  mediumZoom("article img:not(.no-zoom):not(.logo)", {
    background: "rgba(13, 17, 23, 0.92)",
    margin: 24,
    scrollOffset: 40,
  });
});
