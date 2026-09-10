/* Infra-Bench CLS — single-page site.
 *
 * Two jobs only: copy buttons on the citation blocks, and the hand-drawn
 * sketches in the page margins. */
(function () {
  'use strict';

  function $(id) { return document.getElementById(id); }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  // ------------------------------------------------------ citation copying

  function wireCopy() {
    var btns = document.querySelectorAll('.copy[data-copy]');
    Array.prototype.forEach.call(btns, function (btn) {
      btn.addEventListener('click', function () {
        var src = $(btn.dataset.copy);
        if (!src) return;
        var text = src.textContent;
        var done = function () {
          btn.textContent = 'Copied';
          btn.classList.add('done');
          setTimeout(function () {
            btn.textContent = 'Copy';
            btn.classList.remove('done');
          }, 1800);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () {
            fallbackCopy(text, done);
          });
        } else {
          fallbackCopy(text, done);
        }
      });
    });
  }

  function fallbackCopy(text, done) {
    var ta = el('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); done(); } catch (e) { /* no-op */ }
    document.body.removeChild(ta);
  }

  // ------------------------------------------------------ margin sketches
  //
  // Hand-drawn infrastructure down the outer margins, one drawing each, pinned
  // to fixed slots rather than shuffled so the layout is stable. Edit the `img`
  // field to rearrange.
  //
  // Anchors must not set `overflow`, since an absolutely positioned child of a
  // scroll container gets clipped. `f` is a size factor, not a width: CSS sizes
  // each drawing against the margin the layout actually leaves and multiplies
  // by this.

  var SKETCH_SLOTS = [
    { sel: 'header',   img: 'pylon',         side: 'left',  top: -30,   f: 1.00, rot: -4 },
    { sel: '#brief',   img: 'substation',    side: 'right', top: 120,   f: 0.94, rot: 4 },
    { sel: '#access',  img: 'train_station', side: 'left',  top: -10,   f: 0.98, rot: 4 },
    { sel: '#cite',    img: 'storage_tanks', side: 'right', top: 40,    f: 1.00, rot: -3 },
    // Bottom-anchored so it grows upward. Top-anchored, a 600px drawing in a
    // short footer hung below the page, which added dead scroll space.
    { sel: 'footer',   img: 'train_labeled', side: 'left',  bottom: 20, f: 0.96, rot: 3 }
  ];

  /* Only build them once the layout can show them.
   *
   * Hiding with `display: none` and trusting loading="lazy" to skip the fetch
   * does not work — measured, a phone still downloaded all five, about a
   * megabyte of pure decoration. Never creating the elements is the only way to
   * actually avoid the requests. */
  var SKETCH_MQ = '(min-width: 1280px)';

  function placeSketches() {
    if (!window.matchMedia(SKETCH_MQ).matches) return;
    if (document.querySelector('.sketch')) return;   // already built
    SKETCH_SLOTS.forEach(function (slot) {
      var host = document.querySelector(slot.sel);
      if (!host) return;
      host.classList.add('sketch-host');
      var img = el('img', 'sketch sketch-' + slot.side);
      img.src = 'figures/sketches/' + slot.img + '.png';
      img.alt = '';                       // decorative
      img.setAttribute('aria-hidden', 'true');
      img.loading = 'lazy';
      if (slot.bottom != null) img.style.bottom = slot.bottom + 'px';
      else img.style.top = slot.top + 'px';
      img.style.setProperty('--sk-f', slot.f);
      img.style.transform = 'rotate(' + slot.rot + 'deg)';
      host.appendChild(img);
    });
  }

  // -------------------------------------------------------------------- boot

  wireCopy();
  placeSketches();

  // Widening past the breakpoint after load should still get them.
  if (window.matchMedia) {
    var skMq = window.matchMedia(SKETCH_MQ);
    var onSkChange = function () { if (skMq.matches) placeSketches(); };
    if (skMq.addEventListener) skMq.addEventListener('change', onSkChange);
    else if (skMq.addListener) skMq.addListener(onSkChange);
  }
})();
