/* Infra-Bench CLS — shared script.
 *
 * On the results page it builds the filterable table from data/results.json.
 * On the notes page it wires the citation copy button. Both guard on element
 * presence, so this file is safe to include everywhere. */
(function () {
  'use strict';

  function $(id) { return document.getElementById(id); }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function optionEl(value, label) {
    var o = el('option', null, label);
    o.value = value;
    return o;
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

  // --------------------------------------------------------- help toggles

  /* Each "?" button reveals the panel named by its aria-controls. Opening one
   * closes the others, so the questions never get buried under two panels. */
  function wireHelp() {
    var btns = document.querySelectorAll('button.help[aria-controls]');
    Array.prototype.forEach.call(btns, function (btn) {
      btn.addEventListener('click', function () {
        var body = $(btn.getAttribute('aria-controls'));
        if (!body) return;
        var opening = body.hidden;
        Array.prototype.forEach.call(btns, function (other) {
          var b = $(other.getAttribute('aria-controls'));
          if (b) b.hidden = true;
          other.setAttribute('aria-expanded', 'false');
        });
        body.hidden = !opening;
        btn.setAttribute('aria-expanded', opening ? 'true' : 'false');
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

  // ------------------------------------------------------------- results

  var META, ROWS, CLASS_KEYS, REGION_KEYS, SECTOR_CLASSES;

  var state = {
    breakdown: 'overall',
    metric: 'macro_f1',
    protocol: 'all',
    scale: 'all',
    fm: 'all',
    baselines: true,
    std: true,
    sortKey: 'metric',
    sortDir: -1,
    picked: null,     // condition the recommender sent us to, as "fm|prot|scale"
    arriving: false   // true for the one render that should play the pulse
  };

  function condKey(r) { return r.fm + '|' + r.protocol + '|' + r.scale; }

  var METRICS = [
    ['macro_f1',           'Macro F1'],
    ['weighted_f1',        'Weighted F1'],
    ['accuracy',           'Accuracy'],
    ['macro_precision',    'Macro precision'],
    ['macro_recall',       'Macro recall'],
    ['weighted_precision', 'Weighted precision']
  ];

  var BREAKDOWNS = [
    ['overall', 'Overall'],
    ['class',   'By class'],
    ['sector',  'By sector'],
    ['region',  'By region']
  ];

  var SECTORS = [
    ['energy', 'Energy'], ['water', 'Water'],
    ['transport', 'Transport'], ['telecom', 'Telecom']
  ];

  var REPO_URL = 'https://github.com/justing0909/infra-bench-cls';

  var PROTOCOL_FULL = {
    LP: 'Linear probe — frozen backbone',
    FT: 'Fine-tune — full backbone',
    Sup: 'Supervised — trained end to end'
  };

  /* Scores are reported as percentages — a bare 0.509 reads as far more
   * precise and far less meaningful than 50.9%. */
  function pct(v, dp) {
    return (v == null || isNaN(v)) ? '—' : (v * 100).toFixed(dp == null ? 1 : dp) + '%';
  }

  /* Just the number, for a ± that follows a percentage. */
  function pctSpread(v) {
    return (v == null || isNaN(v)) ? '' : (v * 100).toFixed(1);
  }

  /* A difference between two percentages is measured in points, not percent. */
  function points(v) {
    return (v == null || isNaN(v)) ? '—' : (v * 100).toFixed(1) + '-point';
  }

  function article(word) {
    return /^[aeiou]/i.test(word) ? 'an' : 'a';
  }

  /* Per-class recall and precision, recovered from the summed confusion
   * matrix. Recall divides by the class's true support rather than the sliced
   * row total, so it matches the paper; precision uses the sliced column. */
  function classRates(row, key) {
    var i = CLASS_KEYS.indexOf(key);
    var cm = row.confusion_10;
    if (i < 0 || !cm) return null;
    var support = META.class_n_10[i] * META.seeds.length;
    var tp = cm[i][i];
    var col = 0;
    for (var r = 0; r < cm.length; r++) col += cm[r][i];
    return {
      recall: support ? tp / support : null,
      precision: col ? tp / col : null
    };
  }

  function mean(r, key) {
    var e = r[key];
    return e && e.mean != null ? e.mean : null;
  }

  /* Value for one row in one column of the current breakdown. */
  function cellEntry(r, key) {
    var e;
    if (state.breakdown === 'overall') {
      e = r[key];
    } else if (state.breakdown === 'class') {
      var list = r.per_class_f1 || [];
      for (var i = 0; i < list.length; i++) {
        if (list[i].key === key) { e = list[i]; break; }
      }
    } else if (state.breakdown === 'sector') {
      e = (r.per_sector_f1 || {})[key];
    } else if (state.breakdown === 'region') {
      e = (r.per_region_f1 || {})[key];
    }
    return (e && e.mean != null) ? e : null;
  }

  function cellVal(r, key) {
    var e = cellEntry(r, key);
    return e ? e.mean : null;
  }

  /* Columns of the current breakdown, past the model/adapt/scale block.
   *
   * Class and region breakdowns follow the recommender's selection, so the
   * table's Mean column is the same quantity the recommendation was ranked on.
   * With everything selected — the default — every column shows. */
  function columns() {
    if (state.breakdown === 'class') {
      var keys = classesNarrowed() ? pick.classes : CLASS_KEYS;
      return keys.map(function (k) {
        var i = CLASS_KEYS.indexOf(k);
        return { key: k, label: META.class_names_10[i], n: META.class_n_10[i] };
      });
    }
    if (state.breakdown === 'sector') {
      var sects = SECTORS.filter(function (s) {
        if (!classesNarrowed()) return true;
        // keep a sector only if at least one of its classes is selected
        return (SECTOR_CLASSES[s[0]] || []).some(function (k) {
          return pick.classes.indexOf(k) >= 0;
        });
      });
      return (sects.length ? sects : SECTORS).map(function (s) {
        return { key: s[0], label: s[1], n: lookupN('per_sector_f1', s[0]) };
      });
    }
    if (state.breakdown === 'region') {
      var rk = regionsNarrowed() ? pick.regions : REGION_KEYS;
      return (rk.length ? rk : REGION_KEYS).map(function (k) {
        return {
          key: k, label: META.region_display[k],
          n: lookupN('per_region_f1', k)
        };
      });
    }
    return [{ key: state.metric, label: metricLabel(state.metric) }];
  }

  function lookupN(field, key) {
    var n = null;
    ROWS.some(function (r) {
      var e = (r[field] || {})[key];
      if (e && e.n != null) { n = e.n; return true; }
      return false;
    });
    return n;
  }

  function metricLabel(key) {
    for (var i = 0; i < METRICS.length; i++) {
      if (METRICS[i][0] === key) return METRICS[i][1];
    }
    return key;
  }

  function visibleRows() {
    return ROWS.filter(function (r) {
      if (state.fm !== 'all' && r.fm !== state.fm) return false;
      if (!state.baselines && r.is_baseline) return false;
      if (state.protocol !== 'all' && r.protocol !== state.protocol) return false;
      if (state.scale !== 'all' && r.scale !== state.scale) return false;
      return true;
    });
  }

  function rowMean(r, cols) {
    var s = 0, n = 0;
    cols.forEach(function (c) {
      var v = cellVal(r, c.key);
      if (v != null) { s += v; n++; }
    });
    return n ? s / n : null;
  }

  // ----------------------------------------------------------------- render

  function render() {
    var cols = columns();
    var rows = visibleRows();
    var isOverall = state.breakdown === 'overall';

    $('ctl-metric').hidden = !isOverall;

    // notes that only apply to particular breakdowns
    var warn = $('region-warn');
    if (state.breakdown === 'region') {
      warn.hidden = false;
      warn.innerHTML = '<strong>These numbers are on the ' +
        META.per_region_class_basis + '-class basis.</strong> The stored ' +
        'results record per-region F1 already averaged over classes, so it ' +
        'cannot be re-fit to the 10-class subset used elsewhere. Compare ' +
        'models against each other here, not against the other views.';
    } else {
      warn.hidden = true;
    }

    var hint = $('view-hint');
    var msg = '';
    if (state.breakdown === 'class') {
      msg = 'Per-class F1 is unaffected by the three excluded classes, since ' +
            'a class’s F1 does not depend on which others are averaged with ' +
            'it.';
    } else if (state.breakdown === 'sector') {
      msg = 'The macro average of per-class F1s for each sector’s retained ' +
            'classes, each scored on the full test set. Telecom has only one ' +
            'class, so its column is just that class’s F1.';
    }
    hint.hidden = !msg;
    hint.textContent = msg;

    // sort
    var sorted = rows.slice().sort(function (a, b) {
      var av, bv;
      if (state.sortKey === 'model') {
        av = META.fm_order.indexOf(a.fm);
        bv = META.fm_order.indexOf(b.fm);
        if (av === bv) {
          av = a.protocol + a.scale;
          bv = b.protocol + b.scale;
          return av < bv ? -state.sortDir : av > bv ? state.sortDir : 0;
        }
        return (av - bv) * state.sortDir;
      }
      if (state.sortKey === 'gmacs') {
        av = (META.cost[a.fm] || {}).gmacs;
        bv = (META.cost[b.fm] || {}).gmacs;
      } else if (state.sortKey === 'metric') {
        av = isOverall ? mean(a, state.metric) : rowMean(a, cols);
        bv = isOverall ? mean(b, state.metric) : rowMean(b, cols);
      } else {
        av = cellVal(a, state.sortKey);
        bv = cellVal(b, state.sortKey);
      }
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return (av - bv) * state.sortDir;
    });

    // best per column, for the bar scale (foundation models only)
    var peak = {};
    cols.forEach(function (c) {
      var m = null;
      sorted.forEach(function (r) {
        var v = cellVal(r, c.key);
        if (v != null && (m == null || v > m)) m = v;
      });
      peak[c.key] = m;
    });

    var table = $('t-results');
    table.innerHTML = '';

    // ---- header
    var thead = el('thead');
    var htr = el('tr');
    htr.appendChild(th('Model', { cls: 'lead', sort: 'model' }));
    htr.appendChild(th('Adapt', { cls: 'lead' }));
    htr.appendChild(th('Scale', { cls: 'lead' }));

    if (isOverall) {
      htr.appendChild(th(metricLabel(state.metric), { sort: 'metric' }));
      htr.appendChild(th('', { cls: 'bar-head' }));
      htr.appendChild(th('GMACs', { sort: 'gmacs' }));
    } else {
      cols.forEach(function (c) {
        htr.appendChild(th(c.label, { sort: c.key, n: c.n }));
      });
      htr.appendChild(th('Mean', { sort: 'metric', cls: 'rowmean' }));
    }
    thead.appendChild(htr);
    table.appendChild(thead);

    // ---- body
    var tbody = el('tbody');
    sorted.forEach(function (r) {
      var cls = r.is_baseline ? 'baseline' : '';
      if (state.picked && condKey(r) === state.picked) {
        cls += ' picked' + (state.arriving ? ' arriving' : '');
      }
      var tr = el('tr', cls.trim() || null);

      tr.appendChild(el('td', 'lead', r.fm));
      var tdA = el('td');
      var tag = el('span', 'cond ' + r.protocol.toLowerCase(), r.protocol);
      tag.title = PROTOCOL_FULL[r.protocol] || r.protocol;
      tdA.appendChild(tag);
      tr.appendChild(tdA);
      tr.appendChild(el('td', null, r.scale));

      if (isOverall) {
        tr.appendChild(valCell(r[state.metric], true));

        var m = mean(r, state.metric) || 0;
        var top = peak[state.metric] || 1;
        var tdB = el('td', 'bar');
        var track = el('span', 'bar-track');
        var fill = el('span', 'bar-fill' + (m === peak[state.metric] ? ' top' : ''));
        fill.style.width = Math.round((m / top) * 100) + '%';
        track.appendChild(fill);
        tdB.appendChild(track);
        tr.appendChild(tdB);

        var cost = META.cost[r.fm] || {};
        var tdG = el('td');
        tdG.appendChild(document.createTextNode(
          cost.gmacs == null ? '—' : cost.gmacs.toFixed(0)));
        if (cost.proxy) {
          var dag = el('span', 'dag', '*');
          dag.title = 'Architecture-matched proxy';
          tdG.appendChild(dag);
        }
        tr.appendChild(tdG);
      } else {
        cols.forEach(function (c) {
          tr.appendChild(valCell(cellEntry(r, c.key), false));
        });
        var mv = rowMean(r, cols);
        tr.appendChild(el('td', 'rowmean', mv == null ? '—' : pct(mv)));
      }

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    markSort(thead);

    // The pulse should play once on arrival, not on every later re-render.
    state.arriving = false;

    var pickedShown = state.picked && sorted.some(function (r) {
      return condKey(r) === state.picked;
    });
    $('tally').textContent =
      sorted.length + ' of ' + ROWS.length + ' conditions' +
      (state.baselines ? '' : ' · baselines hidden') +
      ' · mean over ' + META.seeds.length + ' seeds' +
      ' · ' + (state.breakdown === 'region'
        ? META.per_region_class_basis + '-class basis'
        : '10-class basis') +
      (pickedShown ? ' · your recommended run is marked' : '') +
      ' · click a heading to re-sort';
  }

  function valCell(entry, lead) {
    if (!entry || entry.mean == null) {
      var na = el('td', 'na', '—');
      return na;
    }
    var td = el('td');
    td.appendChild(el('span', lead ? 'lead-val' : null, pct(entry.mean)));
    if (state.std && entry.std != null) {
      td.appendChild(el('span', 'sd', ' ±' + pctSpread(entry.std)));
    }
    return td;
  }

  function th(label, opts) {
    opts = opts || {};
    var n = el('th', opts.cls || null);
    n.appendChild(document.createTextNode(label));
    if (opts.n != null) n.appendChild(el('span', 'n', 'n=' + opts.n));
    if (opts.sort) {
      n.className = ((opts.cls || '') + ' sortable').trim();
      n.dataset.key = opts.sort;
      n.tabIndex = 0;
      n.appendChild(el('span', 'caret', ''));
      var go = function () {
        if (state.sortKey === opts.sort) state.sortDir = -state.sortDir;
        else {
          state.sortKey = opts.sort;
          state.sortDir = opts.sort === 'model' ? 1 : -1;
        }
        render();
      };
      n.addEventListener('click', go);
      n.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); go(); }
      });
    }
    return n;
  }

  function markSort(thead) {
    var ths = thead.querySelectorAll('th.sortable');
    Array.prototype.forEach.call(ths, function (n) {
      if (n.dataset.key === state.sortKey) {
        n.classList.add('sorted');
        n.querySelector('.caret').textContent = state.sortDir === -1 ? '↓' : '↑';
      }
    });
  }

  // --------------------------------------------------------- recommender
  //
  // Maps a few questions about the reader's setup onto the conditions the
  // benchmark actually measured, and reports the winner with the numbers
  // behind it. Deliberately shows its work: the score, the compute cost, the
  // runner-up, and whether the gap between them is bigger than seed noise.

  /* `classes` and `regions` start empty — nothing selected. Empty means "no
   * narrowing", which is not the same as "all": it ranks on the whole-set
   * metric rather than on an average of chosen classes. Both happen to give
   * macro F1 when the metric is macro F1, but keeping them distinct is what
   * lets the pickers start blank and lets a user clear back to blank. */
  var pick = {
    compute: 'modest',
    labels: 'many',
    metric: 'macro_f1',
    classes: [],
    regions: []
  };

  /* Any classes picked at all — used for column filtering and summaries. */
  function classesNarrowed() { return pick.classes.length > 0; }

  /* A *proper* subset: at least one class but not all of them. This is the only
   * case that forces the score onto per-class F1, because the stored results
   * break out F1 alone per class. Nothing picked and everything picked both
   * mean "the whole set", so both leave the metric selector free — picking all
   * ten and being told you cannot choose a metric makes no sense. */
  function classSubset() {
    return pick.classes.length > 0 && pick.classes.length < CLASS_KEYS.length;
  }

  function regionsNarrowed() { return pick.regions.length > 0; }

  var COMPUTE = [
    ['nogpu',  'No GPU'],
    ['modest', 'GPU ≤ 12 GB'],
    ['large',  'GPU 24 GB +']
  ];

  var LABELS = [
    ['many',   'Thousands'],
    ['few',    'A few hundred or less'],
    ['unsure', 'Not sure']
  ];

  /* One line each, for the metric picker's explanation. */
  var METRIC_BLURB = {
    macro_f1:
      'Balances how many real sites a model finds against how often it is ' +
      'right when it chooses one, averaging the ten classes equally so that a ' +
      'rare class counts as much as a common one. This is the headline metric.',
    weighted_f1:
      'The same balance, but weighted by how common each class is, so it ' +
      'tracks the frequent classes and largely ignores the rare ones.',
    accuracy:
      'The share of all test tiles put in the right class. It is easy to ' +
      'read but dominated by the common classes, so a model can score well ' +
      'here while missing every rare one.',
    macro_precision:
      'How often the model is right when it chooses a class, averaged ' +
      'equally across the ten, so high precision means few false alarms.',
    macro_recall:
      'The share of real sites the model finds in each class, averaged ' +
      'equally across the ten, so high recall means few misses.',
    weighted_precision:
      'How often the model is right when it chooses a class, weighted by how ' +
      'common each class is.'
  };

  /* VRAM ceiling for the "modest GPU" tier, in GB. */
  var MODEST_VRAM = 12;

  /* Above this many GMACs a backbone counts as expensive, and we offer a
     cheaper alternative alongside the winner. The set splits cleanly either
     side of ~300: ResNet-18 at 33, the Swin-B and ViT-B models at 247–275,
     then CROMA at 703 and the two ViT-L models at ~955. */
  var LIGHT_GMACS = 300;

  /* Mean and spread of a set of per-key entries, pooling std in quadrature. */
  function poolEntries(entries) {
    if (!entries.length) return null;
    if (entries.length === 1) return { mean: entries[0].mean, std: entries[0].std };
    var sum = 0, varSum = 0;
    entries.forEach(function (e) {
      sum += e.mean;
      varSum += (e.std || 0) * (e.std || 0);
    });
    var n = entries.length;
    return { mean: sum / n, std: Math.sqrt(varSum) / n };
  }

  /* Score a condition against whatever the reader has selected.
   *
   * Regions never enter the ranking. Per-region figures are reported on the
   * 13-class taxonomy and per-class ones on the 10-class subset, so there is
   * no per-region-per-class number anywhere in the data to rank on. Letting
   * regions drive the ranking only when no classes were picked — which is what
   * this used to do — meant that deselecting one sector silently swapped the
   * whole basis and could change the winner for no reason the reader could
   * see. Regions are now always reported separately, alongside. */
  function targetScore(r) {
    if (!classSubset()) return r[pick.metric];
    var byKey = {};
    (r.per_class_f1 || []).forEach(function (c) { byKey[c.key] = c; });
    return poolEntries(pick.classes.map(function (k) { return byKey[k]; })
      .filter(function (e) { return e && e.mean != null; }));
  }

  function className(key) {
    var i = CLASS_KEYS.indexOf(key);
    return i < 0 ? key : META.class_names_10[i];
  }

  /* Name the selection the way the ranking actually treats it. */
  function targetName() {
    if (!classSubset()) return metricLabel(pick.metric) + ', all ten classes';
    if (pick.classes.length === 1) return 'F1 on ' + className(pick.classes[0]);
    var sect = matchedSector();
    if (sect) return sect.charAt(0).toUpperCase() + sect.slice(1) + ' sector F1';
    return 'mean F1 across ' + pick.classes.length + ' classes';
  }

  /* If the picked classes are exactly one sector's members, name that sector —
   * this metric is defined identically, so the label is accurate not just
   * convenient. */
  function matchedSector() {
    var found = null;
    Object.keys(SECTOR_CLASSES).forEach(function (s) {
      var members = SECTOR_CLASSES[s];
      if (members.length !== pick.classes.length) return;
      if (members.every(function (k) { return pick.classes.indexOf(k) >= 0; })) found = s;
    });
    return found;
  }

  /* Can this condition be trained under the stated compute budget? */
  function affordable(r) {
    if (pick.compute === 'nogpu') {
      // Only a frozen backbone plus a linear head is realistic without a GPU.
      return r.protocol === 'LP';
    }
    if (pick.compute === 'modest') {
      if (r.protocol === 'LP') return true;
      // Recorded peak VRAM exists for the fine-tune runs. The supervised
      // ResNet-18 has none recorded but is 11 M params — it fits anywhere.
      if (r.peak_gpu_gb == null) return true;
      return r.peak_gpu_gb <= MODEST_VRAM;
    }
    return true;
  }

  function allowedScales() {
    if (pick.labels === 'many') return ['1.0x'];
    if (pick.labels === 'few') return ['0.3x'];
    return ['1.0x', '0.3x'];
  }

  function recommend() {
    var scales = allowedScales();
    var cands = [];
    ROWS.forEach(function (r) {
      if (scales.indexOf(r.scale) < 0) return;
      if (!affordable(r)) return;
      var s = targetScore(r);
      if (!s || s.mean == null) return;
      cands.push({ row: r, score: s });
    });
    cands.sort(function (a, b) { return b.score.mean - a.score.mean; });
    return cands;
  }

  function renderRec() {
    var box = $('rec');
    box.innerHTML = '';
    var cands = recommend();

    if (!cands.length) {
      box.appendChild(el('p', 'rec-empty',
        'No measured condition fits that combination.'));
      return;
    }

    var best = cands[0];
    var next = cands[1];
    var r = best.row;
    var cost = META.cost[r.fm] || {};

    // ---- headline
    var head = el('div', 'rec-head');
    head.appendChild(el('span', 'rec-kicker', 'Start with'));
    var name = el('p', 'rec-name');
    name.appendChild(document.createTextNode(r.fm));
    name.appendChild(el('span', 'rec-cond',
      r.protocol === 'Sup' ? 'trained from scratch'
        : (r.protocol === 'FT' ? 'fine-tuned' : 'linear probe') +
          ' · ' + r.scale + ' data'));
    head.appendChild(name);
    box.appendChild(head);

    if (r.is_baseline) {
      box.appendChild(el('p', 'rec-flag',
        r.fm === 'Supervised ResNet-18'
          ? 'This is the supervised baseline rather than a foundation model, ' +
            'so under these constraints you may not need one at all.'
          : 'This is the random-features floor. If it is winning, treat the ' +
            'whole comparison with suspicion.'));
    }

    // ---- the numbers
    var stats = el('dl', 'rec-stats');
    /* Each label/value pair is wrapped so the grid places them as one cell —
     * bare dt and dd are separate grid items and would split across columns. */
    function stat(k, v, note) {
      var cell = el('div');
      cell.appendChild(el('dt', null, k));
      var dd = el('dd', null, v);
      if (note) dd.appendChild(el('span', 'rec-note', note));
      cell.appendChild(dd);
      stats.appendChild(cell);
    }
    stat(targetName(),
      pct(best.score.mean) +
        (best.score.std != null ? ' ± ' + pctSpread(best.score.std) : ''),
      ' over ' + META.seeds.length + ' seeds');
    if (classSubset() || pick.metric !== 'macro_f1') {
      stat('Macro F1', pct(mean(r, 'macro_f1')), ' all ten classes');
    }
    stat('Compute', (cost.gmacs == null ? '—' : cost.gmacs.toFixed(0) + ' GMACs') +
      (cost.proxy ? ' *' : ''), ' one forward pass, batch 16');
    if (r.peak_gpu_gb != null) {
      stat('Peak VRAM', r.peak_gpu_gb.toFixed(1) + ' GB',
        r.wall_time_s != null
          ? ' · ' + Math.round(r.wall_time_s / 60) + ' min per seed on an A100'
          : null);
    }
    box.appendChild(stats);

    // ---- what the score actually means, in words
    //
    // F1 is not "gets it right 51% of the time" — it is a balance of two
    // different rates. With one class selected we can recover both from the
    // confusion matrix and say plainly what each one is.
    var glossed = false;
    if (pick.classes.length === 1) {
      var rates = classRates(r, pick.classes[0]);
      if (rates && rates.recall != null && rates.precision != null) {
        var label = className(pick.classes[0]).toLowerCase();
        var gloss = el('p', 'rec-gloss');
        gloss.appendChild(el('strong', null, 'In plain terms: '));
        gloss.appendChild(document.createTextNode(
          'it finds ' + pct(rates.recall, 0) + ' of the real ' + label +
          ' sites, and when it does choose ' + article(label) + ' ' + label +
          ' it is right ' + pct(rates.precision, 0) + ' of the time. The ' +
          pct(best.score.mean) + ' above is F1, which balances those two, so ' +
          'it is not a plain accuracy figure.'));
        box.appendChild(gloss);
        glossed = true;
      }
    }
    if (!glossed) {
      box.appendChild(el('p', 'rec-gloss',
        classSubset()
          ? 'This is the mean F1 across the classes you picked. F1 balances ' +
            'how many real sites a model finds against how often it is right ' +
            'when it chooses one, so read it as a score out of 100 rather ' +
            'than as plain accuracy. Pick a single class to see the two rates ' +
            'separately.'
          : METRIC_BLURB[pick.metric] || ''));
    }

    // ---- the cheapest backbone worth mentioning
    //
    // Ranking on score alone can hand someone who just told us they have no
    // GPU a 955-GMAC ViT-L. Rather than hide the best result, show what the
    // cheap option costs them in accuracy and let them choose.
    var light = null;
    if ((cost.gmacs || 0) > LIGHT_GMACS) {
      cands.forEach(function (c) {
        var g = (META.cost[c.row.fm] || {}).gmacs;
        if (g == null || g > LIGHT_GMACS) return;
        if (c.row.fm === 'Random Features') return;
        if (!light || c.score.mean > light.score.mean) light = c;
      });
    }

    function costClause(c) {
      var g = (META.cost[c.row.fm] || {}).gmacs;
      return ' It also needs ' + g.toFixed(0) + ' GMACs against ' +
        cost.gmacs.toFixed(0) + ', or ' + (cost.gmacs / g).toFixed(1) +
        '× less compute, so it is the better buy if ' +
        (pick.compute === 'nogpu' ? 'you are extracting features on CPU'
                                  : 'throughput matters to you') + '.';
    }

    // ---- runner-up, and whether the gap is real
    if (next) {
      var gap = best.score.mean - next.score.mean;
      var noise = (best.score.std || 0) + (next.score.std || 0);
      var tied = gap < noise;
      var line = el('p', 'rec-second');
      line.appendChild(el('strong', null, tied ? 'Effectively tied with ' : 'Runner-up: '));
      line.appendChild(document.createTextNode(
        next.row.fm + ' (' + next.row.protocol + ', ' + next.row.scale + ') at ' +
        pct(next.score.mean) + '. '));
      line.appendChild(document.createTextNode(
        tied
          ? 'The ' + points(gap) + ' gap is smaller than their combined ' +
            'seed spread, so treat them as interchangeable and pick on cost ' +
            'or convenience instead.'
          : 'That is a ' + points(gap) + ' gap, wider than the combined ' +
            'seed spread.'));
      // Fold the compute argument in rather than repeating the same model on
      // its own line directly underneath.
      if (light && light.row === next.row) {
        line.appendChild(document.createTextNode(costClause(light)));
        light = null;
      }
      box.appendChild(line);
    }

    if (light) {
      var lg = (META.cost[light.row.fm] || {}).gmacs;
      var p2 = el('p', 'rec-second');
      p2.appendChild(el('strong', null, 'Lighter option: '));
      p2.appendChild(document.createTextNode(
        light.row.fm + ' (' + light.row.protocol + ', ' + light.row.scale +
        ') needs ' + lg.toFixed(0) + ' GMACs against ' +
        cost.gmacs.toFixed(0) + ', or ' + (cost.gmacs / lg).toFixed(1) +
        '× less compute, and scores ' + pct(light.score.mean) + ', ' +
        points(best.score.mean - light.score.mean) + ' lower. ' +
        (pick.compute === 'nogpu'
          ? 'Worth considering if you are extracting features on CPU.'
          : 'Worth considering if throughput matters more than accuracy.')));
      box.appendChild(p2);
    }

    // ---- region cross-check, reported alongside rather than mixed in
    if (regionsNarrowed()) {
      var regionsOf = function (o) {
        return poolEntries(pick.regions.map(function (k) {
          return (o.per_region_f1 || {})[k];
        }).filter(function (e) { return e && e.mean != null; }));
      };
      var here = regionsOf(r);
      var leader = null;
      ROWS.forEach(function (o) {
        if (allowedScales().indexOf(o.scale) < 0 || !affordable(o)) return;
        var e = regionsOf(o);
        if (e && (!leader || e.mean > leader.e.mean)) leader = { row: o, e: e };
      });
      var where = pick.regions.length === 1
        ? META.region_display[pick.regions[0]]
        : pick.regions.map(function (k) { return META.region_display[k]; }).join(', ');
      var p = el('p', 'rec-region');
      p.appendChild(el('strong', null, 'In ' + where + ': '));
      if (here) {
        p.appendChild(document.createTextNode(
          'this pick scores ' + pct(here.mean) +
          (pick.regions.length > 1 ? ' averaged over those regions. ' : '. ')));
      }
      if (leader && leader.row.fm !== r.fm) {
        p.appendChild(document.createTextNode(
          leader.row.fm + ' (' + leader.row.protocol + ', ' + leader.row.scale +
          ') leads there at ' + pct(leader.e.mean) + ', so it is worth a look too. '));
      } else if (leader) {
        p.appendChild(document.createTextNode('It leads there too. '));
      }
      p.appendChild(el('em', null,
        'Reported separately because region figures are on the ' +
        META.per_region_class_basis + '-class taxonomy, and the data holds no ' +
        'per-region, per-class number, so region cannot enter the ranking.'));
      box.appendChild(p);
    }

    // ---- jump into the table with these constraints applied
    var act = el('div', 'rec-actions');
    var btn = el('button', 'rec-btn', 'Show this in the table');
    btn.type = 'button';
    btn.addEventListener('click', function () {
      state.protocol = r.protocol;
      state.scale = r.scale;

      // Land on the breakdown that shows what was actually ranked, sorted by
      // that same thing — otherwise the recommended row can arrive sitting
      // below rows that beat it on a metric the reader never asked about.
      if (classSubset()) {
        var sect = matchedSector();
        if (sect) {
          state.breakdown = 'sector';
          state.sortKey = sect;
        } else {
          state.breakdown = 'class';
          state.sortKey = pick.classes.length === 1 ? pick.classes[0] : 'metric';
        }
      } else {
        state.breakdown = 'overall';
        state.metric = pick.metric;
        state.sortKey = 'metric';
      }
      state.sortDir = -1;
      state.picked = condKey(r);
      state.arriving = true;
      repaintSegs();
      render();
      document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
    });
    act.appendChild(btn);

    // Same treatment as the button beside it, but an anchor, since it
    // navigates rather than acting on the page.
    var repo = el('a', 'rec-btn', 'Have your own labels? Try it out');
    repo.href = REPO_URL;
    repo.target = '_blank';
    repo.rel = 'noopener';
    act.appendChild(repo);

    box.appendChild(act);
  }

  // ----------------------------------------------------------- chip pickers

  function toggleIn(arr, key) {
    var i = arr.indexOf(key);
    if (i < 0) arr.push(key);
    else arr.splice(i, 1);
  }

  function buildChips() {
    var classBox = $('q-classes');
    var regionBox = $('q-regions');

    function classChip(k) {
      var i = CLASS_KEYS.indexOf(k);
      var on = pick.classes.indexOf(k) >= 0;
      var b = el('button', 'chip' + (on ? ' on' : ''), META.class_names_10[i]);
      b.type = 'button';
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
      b.title = 'n=' + META.class_n_10[i] + ' test tiles per seed';
      b.appendChild(el('span', 'chip-n', String(META.class_n_10[i])));
      b.addEventListener('click', function () {
        toggleIn(pick.classes, k);
        paint(); renderRec(); render();
      });
      return b;
    }

    /* A quiet text control, for select-all / clear. */
    function textBtn(label, fn) {
      var b = el('button', 'reset-btn', label);
      b.type = 'button';
      b.addEventListener('click', function () { fn(); paint(); renderRec(); render(); });
      return b;
    }

    function paint() {
      // ---- one row per sector: the sector button, then its classes
      classBox.innerHTML = '';
      SECTORS.forEach(function (s) {
        var members = SECTOR_CLASSES[s[0]] || [];
        var on = members.length > 0 && members.every(function (k) {
          return pick.classes.indexOf(k) >= 0;
        });
        var some = members.some(function (k) { return pick.classes.indexOf(k) >= 0; });

        var b = el('button',
          'sector-btn' + (on ? ' on' : (some ? ' part' : '')), s[1]);
        b.type = 'button';
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
        b.title = (on ? 'Remove' : 'Add') + ' all ' + members.length + ' ' +
          s[1].toLowerCase() + ' classes';
        b.appendChild(el('span', 'sector-n', String(members.length)));
        b.addEventListener('click', function () {
          // Toggling a sector off leaves the rest alone, so clicking through
          // every sector ends at nothing selected rather than snapping back.
          members.forEach(function (k) {
            var i = pick.classes.indexOf(k);
            if (on && i >= 0) pick.classes.splice(i, 1);
            else if (!on && i < 0) pick.classes.push(k);
          });
          paint(); renderRec(); render();
        });
        classBox.appendChild(b);

        var row = el('div', 'sector-classes');
        members.forEach(function (k) { row.appendChild(classChip(k)); });
        classBox.appendChild(row);
      });

      var classActions = el('div', 'pick-actions');
      if (pick.classes.length < CLASS_KEYS.length) {
        classActions.appendChild(textBtn('Select all ten', function () {
          pick.classes = CLASS_KEYS.slice();
        }));
      }
      if (classesNarrowed()) {
        classActions.appendChild(textBtn('Clear', function () {
          pick.classes = [];
        }));
      }
      classBox.appendChild(classActions);

      // ---- region chips
      regionBox.innerHTML = '';
      REGION_KEYS.forEach(function (k) {
        var on = pick.regions.indexOf(k) >= 0;
        var b = el('button', 'chip' + (on ? ' on' : ''), META.region_display[k]);
        b.type = 'button';
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
        var n = lookupN('per_region_f1', k);
        if (n) {
          b.title = 'n=' + n + ' test tiles per seed';
          b.appendChild(el('span', 'chip-n', String(n)));
        }
        b.addEventListener('click', function () {
          toggleIn(pick.regions, k);
          paint(); renderRec(); render();
        });
        regionBox.appendChild(b);
      });

      var regionActions = el('div', 'pick-actions');
      if (pick.regions.length < REGION_KEYS.length) {
        regionActions.appendChild(textBtn('Select all seven', function () {
          pick.regions = REGION_KEYS.slice();
        }));
      }
      if (regionsNarrowed()) {
        regionActions.appendChild(textBtn('Clear', function () {
          pick.regions = [];
        }));
      }
      regionBox.appendChild(regionActions);

      // ---- plain-language summaries
      var sect = matchedSector();
      $('sum-classes').textContent = !classSubset()
        ? (classesNarrowed()
            ? 'All ten picked, so the whole set is scored on ' +
              metricLabel(pick.metric) + '.'
            : 'Nothing picked, so the whole set is scored on ' +
              metricLabel(pick.metric) + '. Pick some classes to narrow it.')
        : (sect
            ? 'The ' + sect + ' sector in full, which is exactly the ' +
              'per-sector F1 reported in the paper.'
            : pick.classes.length + (pick.classes.length === 1 ? ' class' : ' classes') +
              ': ' + pick.classes.map(className).join(', ') + '.');
      // Locked only on a proper subset, where per-class F1 is all there is.
      var mp = $('metric-pick');
      mp.classList.toggle('is-muted', classSubset());
      $('q-metric').disabled = classSubset();
      $('sum-metric').textContent = classSubset()
        ? 'Not adjustable while only some classes are picked, because the ' +
          'stored results break out F1 alone per class. Pick all ten, or ' +
          'none, to score on another metric.'
        : METRIC_BLURB[pick.metric] || '';

      $('sum-regions').textContent = !regionsNarrowed()
        ? 'Nothing picked, so no regional breakdown is shown.'
        : (pick.regions.length === REGION_KEYS.length
            ? 'All seven, reported alongside the ranking rather than folded in.'
            : pick.regions.length + ' of 7, reported alongside the ranking. ' +
              'Region and class figures sit on different class bases and ' +
              'cannot be combined into one score.');
    }

    paint();
    repaintChips = paint;
  }

  // ------------------------------------------------------------------ setup

  var repaintSegs = function () {};
  var repaintChips = function () {};

  /* A segmented control bound to `store[key]`. Returns its repaint function so
   * callers that change the bound value programmatically can resync the UI. */
  function seg(id, options, key, opts) {
    opts = opts || {};
    var store = opts.store || state;
    var after = opts.after;
    var onPick = opts.onPick;
    var box = $(id);
    function paint() {
      box.innerHTML = '';
      options.forEach(function (o) {
        var b = el('button', store[key] === o[0] ? 'on' : null, o[1]);
        b.type = 'button';
        if (o[2]) b.title = o[2];
        b.addEventListener('click', function () {
          store[key] = o[0];
          if (onPick) onPick();
          paint();
          (after || render)();
        });
        box.appendChild(b);
      });
    }
    paint();
    return paint;
  }

  function setupResults(data) {
    META = data.meta;
    ROWS = data.conditions;

    var src = null;
    for (var i = 0; i < ROWS.length && !src; i++) {
      if (ROWS[i].per_class_f1 && ROWS[i].per_class_f1.length) src = ROWS[i].per_class_f1;
    }
    CLASS_KEYS = (src || []).map(function (c) { return c.key; });
    REGION_KEYS = Object.keys(META.region_display);

    // Sector membership comes from the ontology key prefix, which is how the
    // generator groups them too — so a full sector selection reproduces the
    // paper's per-sector F1 exactly.
    SECTOR_CLASSES = {};
    SECTORS.forEach(function (s) { SECTOR_CLASSES[s[0]] = []; });
    CLASS_KEYS.forEach(function (k) {
      var sect = k.split('.')[0];
      if (SECTOR_CLASSES[sect]) SECTOR_CLASSES[sect].push(k);
    });


    // A sort keyed to a column that the new breakdown doesn't have would
    // silently do nothing, so fall back to the headline column on switch.
    var paintBreakdown = seg('f-breakdown', BREAKDOWNS, 'breakdown', {
      onPick: function () {
        state.sortKey = 'metric';
        state.sortDir = -1;
      }
    });

    var protos = [];
    ROWS.forEach(function (r) {
      if (protos.indexOf(r.protocol) < 0) protos.push(r.protocol);
    });
    var paintProtocol = seg('f-protocol',
      [['all', 'All']].concat(protos.map(function (p) {
        return [p, p, PROTOCOL_FULL[p] || p];
      })), 'protocol');

    var scales = [];
    ROWS.forEach(function (r) {
      if (scales.indexOf(r.scale) < 0) scales.push(r.scale);
    });
    scales.sort().reverse();
    var paintScale = seg('f-scale', [['all', 'All']].concat(scales.map(function (s) {
      return [s, s];
    })), 'scale');

    repaintSegs = function () {
      paintBreakdown();
      paintProtocol();
      paintScale();
      repaintChips();
      $('f-metric').value = state.metric;
    };

    // ---- recommender controls
    seg('q-compute', COMPUTE, 'compute', {
      store: pick, after: renderRec
    });
    seg('q-labels', LABELS, 'labels', {
      store: pick, after: renderRec
    });

    // ---- model filter for the table
    var fsel = $('f-fm');
    fsel.appendChild(optionEl('all', 'All models'));
    META.fm_order.forEach(function (fm) {
      fsel.appendChild(optionEl(fm, fm));
    });
    fsel.value = state.fm;
    fsel.addEventListener('change', function () {
      state.fm = fsel.value;
      render();
    });

    // ---- recommender metric, plus the explanation list
    var msel = $('q-metric');
    METRICS.forEach(function (m) {
      msel.appendChild(optionEl(m[0], m[1]));
    });
    msel.value = pick.metric;
    msel.addEventListener('change', function () {
      pick.metric = msel.value;
      repaintChips();
      renderRec();
      render();
    });

    var list = $('metric-list');
    METRICS.forEach(function (m) {
      list.appendChild(el('dt', null, m[1]));
      list.appendChild(el('dd', null, METRIC_BLURB[m[0]] || ''));
    });

    buildChips();

    var sel = $('f-metric');
    METRICS.forEach(function (m) {
      var o = el('option', null, m[1]);
      o.value = m[0];
      sel.appendChild(o);
    });
    sel.value = state.metric;
    sel.addEventListener('change', function () {
      state.metric = sel.value;
      state.sortKey = 'metric';
      state.sortDir = -1;
      render();
    });

    $('f-baselines').addEventListener('change', function () {
      state.baselines = this.checked;
      render();
    });
    $('f-std').addEventListener('change', function () {
      state.std = this.checked;
      render();
    });

    $('help-test-n').textContent = META.test_n_per_seed.toLocaleString();

    $('footer-meta').textContent =
      ROWS.length + ' conditions · ' + META.fm_order.length + ' models · ' +
      META.seeds.length + ' seeds (' + META.seeds.join(', ') + ') · ' +
      META.class_basis + ' of ' +
      (META.class_basis + META.excluded_classes.length) + ' classes · ' +
      META.test_n_per_seed.toLocaleString() + ' test tiles per seed';

    $('loading').hidden = true;
    $('app').hidden = false;
    renderRec();
    render();
  }

  // ------------------------------------------------------ margin sketches
  //
  // Hand-drawn infrastructure down the outer margins, one drawing each.
  //
  // Assignments are pinned rather than shuffled. Randomizing meant the layout
  // changed every visit, which made it impossible to say "move that one" about
  // anything on screen. Edit the `img` field below to rearrange.
  //
  // Two constraints on anchors. They must not set `overflow`, since an
  // absolutely positioned child of a scroll container gets clipped — that
  // rules out .scroll. And their innerHTML must never be replaced, which rules
  // out .rec: it is rebuilt on every interaction, taking any child with it.
  // #quickstart is the stable stand-in, anchored from its bottom so the
  // drawing still lands beside the recommendation card.
  //
  // `f` is a size factor, not a width. CSS sizes each drawing against the
  // margin the layout actually leaves and multiplies by this, so the set keeps
  // a little variety without any of them being pinned to a fixed pixel size.

  var SKETCH_SLOTS = [
    { sel: 'header',      img: 'pylon',         side: 'left',  top: -30,   f: 1.00, rot: -4 },
    { sel: '.qs',         img: 'substation',    side: 'right', top: -20,   f: 0.94, rot: 4 },
    { sel: '#quickstart', img: 'train_station', side: 'left',  bottom: 20, f: 0.98, rot: 4 },
    { sel: '#results',    img: 'storage_tanks', side: 'right', top: 250,   f: 1.00, rot: -3 },
    { sel: 'footer',      img: 'train_labeled', side: 'left',  top: -20,   f: 0.96, rot: 3 }
  ];

  /* Only build them once the layout can show them.
   *
   * Hiding with `display: none` and trusting loading="lazy" to skip the fetch
   * does not work — measured, a phone still downloaded all five, about a
   * megabyte of pure decoration. Never creating the elements is the only way to
   * actually avoid the requests. */
  var SKETCH_MQ = '(min-width: 1280px)';

  function placeSketches() {
    if (!$('t-results')) return;          // front page only
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
  wireHelp();
  placeSketches();

  // Widening past the breakpoint after load should still get them.
  if (window.matchMedia) {
    var skMq = window.matchMedia(SKETCH_MQ);
    var onSkChange = function () { if (skMq.matches) placeSketches(); };
    if (skMq.addEventListener) skMq.addEventListener('change', onSkChange);
    else if (skMq.addListener) skMq.addListener(onSkChange);
  }

  if ($('t-results')) {
    fetch('data/results.json')
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(setupResults)
      .catch(function (err) {
        var n = $('loading');
        n.className = 'warn';
        n.textContent = 'Could not load data/results.json (' + err.message +
          '). If you opened this file straight from disk, browsers block ' +
          'local fetches — serve the folder over HTTP instead, e.g. ' +
          '"python -m http.server".';
      });
  }
})();
