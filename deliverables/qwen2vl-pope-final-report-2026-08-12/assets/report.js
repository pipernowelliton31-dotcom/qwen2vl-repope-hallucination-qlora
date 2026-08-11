"use strict";

(() => {
  const DATA = window.QLORA_REPORT_DATA;
  if (!DATA) throw new Error("QLORA_REPORT_DATA is missing.");

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const pct = value => `${(value * 100).toFixed(2)}%`;
  const pp = value => `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)} pp`;
  const cap = value => value.charAt(0).toUpperCase() + value.slice(1);
  const overall = run => run.metrics.overall;
  const getRun = id => DATA.repopeRuns.find(run => run.id === id);
  const getDev = (experiment, checkpoint = 0) => DATA.devCheckpoints.find(run => run.experiment === experiment && run.checkpoint === checkpoint);

  const fresh = getRun("e0_fresh_base_repope_256vt");
  const e3_256 = getRun("e3_checkpoint-1000_repope_256vt");
  const e3_512 = getRun("e3_checkpoint-1000_repope_512vt");

  function svg(markup) {
    const host = document.createElement("div");
    host.innerHTML = markup.trim();
    return host.firstElementChild;
  }

  function renderHeroMiniChart() {
    const host = $("#hero-mini-chart");
    const rows = [
      { label: "Fresh E0", value: overall(fresh).accuracy, color: "#91a7b2" },
      { label: "E3 · 256", value: overall(e3_256).accuracy, color: "#b5a588" },
      { label: "E3 · 512", value: overall(e3_512).accuracy, color: "#d6e2dc" },
    ];
    const width = 360;
    const height = 120;
    const min = .9;
    const x = value => 86 + ((value - min) / (.95 - min)) * 250;
    let markup = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Fresh E0、E3 256 和 E3 512 的 Accuracy 对比">`;
    rows.forEach((row, index) => {
      const y = 18 + index * 40;
      markup += `<text x="0" y="${y + 5}" fill="#aebbc0" font-size="11">${row.label}</text>`;
      markup += `<line x1="86" y1="${y}" x2="336" y2="${y}" stroke="rgba(255,255,255,.1)" stroke-width="8" stroke-linecap="round"></line>`;
      markup += `<line x1="86" y1="${y}" x2="${x(row.value)}" y2="${y}" stroke="${row.color}" stroke-width="8" stroke-linecap="round"></line>`;
      markup += `<text x="345" y="${y + 5}" fill="#e8eef0" font-family="Georgia" font-size="12" text-anchor="end">${pct(row.value)}</text>`;
    });
    markup += "</svg>";
    host.replaceChildren(svg(markup));
  }

  function runLabel(run) {
    if (run.id.startsWith("e0_")) return "E0 fresh";
    const suffix = run.visualTokens === 512 ? " · 512vt" : run.exploratory ? `-${run.checkpoint}` : `-${run.checkpoint}`;
    return `${run.experiment}${suffix}`;
  }

  function renderFrontier() {
    const host = $("#frontier-chart");
    const width = 790;
    const height = 500;
    const pad = { left: 70, right: 68, top: 44, bottom: 64 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const xMin = 3.4;
    const xMax = 10;
    const yMin = 85.5;
    const yMax = 94.5;
    const x = value => pad.left + ((value - xMin) / (xMax - xMin)) * plotW;
    const y = value => pad.top + ((yMax - value) / (yMax - yMin)) * plotH;
    const points = DATA.repopeRuns.map(run => ({
      run,
      fpr: overall(run).false_positive_rate * 100,
      recall: overall(run).recall * 100,
    }));
    const frontier = points
      .filter(point => !points.some(other => (
        other !== point && other.fpr <= point.fpr && other.recall >= point.recall &&
        (other.fpr < point.fpr || other.recall > point.recall)
      )))
      .sort((a, b) => a.fpr - b.fpr);
    const offsets = {
      e0_fresh_base_repope_256vt: [-2, -18, "middle"],
      "e1_checkpoint-1000_repope_256vt": [8, -14, "start"],
      "e2_checkpoint-1000_repope_256vt": [9, 17, "start"],
      "e3_checkpoint-1000_repope_256vt": [8, 18, "start"],
      "e3_checkpoint-1000_repope_512vt": [8, -16, "start"],
      "e3_exploratory_high_recall_checkpoint-2000_repope_256vt": [-8, -14, "end"],
      "e4_checkpoint-1000_repope_256vt": [8, -15, "start"],
      "e4_exploratory_high_recall_checkpoint-2000_repope_256vt": [-8, 18, "end"],
    };
    let markup = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="frontier-title frontier-desc"><title id="frontier-title">RePOPE Recall 与 FPR 前沿图</title><desc id="frontier-desc">横轴 FPR 越低越好，纵轴 Recall 越高越好。E3 checkpoint-1000 从 256 到 512 visual tokens 向左上移动。</desc>`;
    [4, 5, 6, 7, 8, 9, 10].forEach(tick => {
      markup += `<line class="svg-grid" x1="${x(tick)}" y1="${pad.top}" x2="${x(tick)}" y2="${height - pad.bottom}"></line>`;
      markup += `<text class="svg-axis" x="${x(tick)}" y="${height - 35}" text-anchor="middle">${tick}%</text>`;
    });
    [86, 88, 90, 92, 94].forEach(tick => {
      markup += `<line class="svg-grid" x1="${pad.left}" y1="${y(tick)}" x2="${width - pad.right}" y2="${y(tick)}"></line>`;
      markup += `<text class="svg-axis" x="${pad.left - 14}" y="${y(tick) + 4}" text-anchor="end">${tick}%</text>`;
    });
    markup += `<text class="svg-axis" x="${pad.left + plotW / 2}" y="${height - 8}" text-anchor="middle">FALSE POSITIVE RATE · LOWER IS BETTER →</text>`;
    markup += `<text class="svg-axis" transform="translate(18 ${pad.top + plotH / 2}) rotate(-90)" text-anchor="middle">RECALL · HIGHER IS BETTER →</text>`;
    markup += `<polyline points="${frontier.map(point => `${x(point.fpr)},${y(point.recall)}`).join(" ")}" fill="none" stroke="#c4b28e" stroke-width="2" stroke-dasharray="6 7" opacity=".78"></polyline>`;
    markup += `<line x1="${x(overall(e3_256).false_positive_rate * 100)}" y1="${y(overall(e3_256).recall * 100)}" x2="${x(overall(e3_512).false_positive_rate * 100)}" y2="${y(overall(e3_512).recall * 100)}" stroke="#d4c4a4" stroke-width="3" marker-end="url(#arrow)"></line>`;
    markup += `<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#d4c4a4"></path></marker></defs>`;
    points.forEach(point => {
      const run = point.run;
      const isFresh = run.experiment === "E0";
      const isBest = run.id === e3_512.id;
      const color = run.exploratory ? "#9d6657" : isFresh ? "#c4b28e" : "#91a7b2";
      const radius = isBest ? 10 : 7;
      const [dx, dy, anchor] = offsets[run.id];
      markup += `<circle cx="${x(point.fpr)}" cy="${y(point.recall)}" r="${radius + (isBest ? 5 : 0)}" fill="${isBest ? "rgba(196,178,142,.16)" : "transparent"}"></circle>`;
      markup += `<circle cx="${x(point.fpr)}" cy="${y(point.recall)}" r="${radius}" fill="${color}" stroke="#182126" stroke-width="3"><title>${runLabel(run)} · Recall ${point.recall.toFixed(2)}% · FPR ${point.fpr.toFixed(2)}%</title></circle>`;
      markup += `<text class="svg-label" x="${x(point.fpr) + dx}" y="${y(point.recall) + dy}" text-anchor="${anchor}">${runLabel(run)}</text>`;
    });
    markup += "</svg>";
    host.replaceChildren(svg(markup));
  }

  function renderMetricRibbon() {
    const host = $("#metric-ribbon");
    const e3Best = overall(e3_512);
    const highRecall = [...DATA.repopeRuns].sort((a, b) => overall(b).recall - overall(a).recall)[0];
    host.innerHTML = [
      ["Best Accuracy", pct(e3Best.accuracy), "E3-1000 · 512vt"],
      ["Best F1", pct(e3Best.f1), "E3-1000 · 512vt"],
      ["Lowest FPR", pct(overall(fresh).false_positive_rate), "Fresh E0 · 256vt"],
      ["Highest Recall", pct(overall(highRecall).recall), runLabel(highRecall)],
    ].map(([label, value, note]) => `<article><span>${label}</span><strong>${value}</strong><p>${note}</p></article>`).join("");
  }

  function renderRepopeTable() {
    const table = $("#repope-table");
    const headers = ["配置", "VT", "Acc", "Precision", "Recall", "F1", "FPR", "Adv. FPR", "Yes ratio", "TP / FP / TN / FN"];
    const body = DATA.repopeRuns.map(run => {
      const m = overall(run);
      const classes = [run.id === e3_512.id ? "highlight" : "", run.exploratory ? "exploratory" : ""].filter(Boolean).join(" ");
      const tag = run.id === e3_512.id ? `<span class="row-tag">best observed</span>` : run.id === e3_256.id ? `<span class="row-tag">formal</span>` : "";
      return `<tr class="${classes}"><th scope="row">${runLabel(run)}${tag}</th><td>${run.visualTokens}</td><td>${pct(m.accuracy)}</td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td>${pct(m.f1)}</td><td>${pct(m.false_positive_rate)}</td><td>${pct(run.metrics.adversarial.false_positive_rate)}</td><td>${pct(m.yes_ratio)}</td><td>${m.tp} / ${m.fp} / ${m.tn} / ${m.fn}</td></tr>`;
    }).join("");
    table.innerHTML = `<thead><tr>${headers.map((header, index) => `<th scope="col"${index ? "" : " class=\"config-column\""}>${header}</th>`).join("")}</tr></thead><tbody>${body}</tbody>`;
  }

  function renderDevCharts() {
    const host = $("#dev-charts");
    const width = 500;
    const height = 270;
    const pad = { left: 45, right: 46, top: 25, bottom: 38 };
    const steps = [500, 1000, 1500, 2000];
    const x = index => pad.left + index * ((width - pad.left - pad.right) / 3);
    const yRecall = value => pad.top + ((98 - value) / 10) * (height - pad.top - pad.bottom);
    const yFpr = value => pad.top + ((6.5 - value) / 6.5) * (height - pad.top - pad.bottom);
    host.innerHTML = ["E1", "E2", "E3", "E4"].map(experiment => {
      const rows = steps.map(step => getDev(experiment, step));
      let markup = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${experiment} checkpoint 的 Recall 和 FPR 动态">`;
      [90, 92, 94, 96, 98].forEach(tick => {
        markup += `<line class="svg-grid-light" x1="${pad.left}" y1="${yRecall(tick)}" x2="${width - pad.right}" y2="${yRecall(tick)}"></line>`;
        markup += `<text class="svg-axis-light" x="${pad.left - 8}" y="${yRecall(tick) + 3}" text-anchor="end">${tick}</text>`;
      });
      [0, 2, 4, 6].forEach(tick => markup += `<text class="svg-axis-light" x="${width - pad.right + 8}" y="${yFpr(tick) + 3}">${tick}</text>`);
      markup += `<polyline points="${rows.map((run, index) => `${x(index)},${yRecall(overall(run).recall * 100)}`).join(" ")}" fill="none" stroke="#6d8491" stroke-width="3"></polyline>`;
      markup += `<polyline points="${rows.map((run, index) => `${x(index)},${yFpr(overall(run).false_positive_rate * 100)}`).join(" ")}" fill="none" stroke="#9d6657" stroke-width="3"></polyline>`;
      rows.forEach((run, index) => {
        const m = overall(run);
        markup += `<circle cx="${x(index)}" cy="${yRecall(m.recall * 100)}" r="5" fill="#6d8491"><title>Recall ${pct(m.recall)}</title></circle>`;
        markup += `<circle cx="${x(index)}" cy="${yFpr(m.false_positive_rate * 100)}" r="5" fill="#9d6657"><title>FPR ${pct(m.false_positive_rate)}</title></circle>`;
        markup += `<text class="svg-axis-light" x="${x(index)}" y="${height - 12}" text-anchor="middle">${steps[index]}</text>`;
      });
      markup += "</svg>";
      const label = DATA.models[experiment];
      return `<article class="dev-chart-card"><header><h3>${experiment} · ${label}</h3><span>Recall / FPR</span></header>${markup}<div class="dev-chart-legend"><span><i></i>Recall · 左轴</span><span><i class="fpr"></i>FPR · 右轴</span></div></article>`;
    }).join("");
  }

  function renderDevTable() {
    const table = $("#dev-table");
    const headers = ["实验", "Checkpoint", "Acc", "Precision", "Recall", "F1", "FPR", "Adv. FPR", "Yes ratio", "TP / FP / TN / FN"];
    const body = DATA.devCheckpoints.map(run => {
      const m = overall(run);
      const selected = run.checkpoint === 1000 && ["E3", "E4"].includes(run.experiment);
      return `<tr class="${selected ? "highlight" : ""}"><th scope="row">${run.experiment} · ${DATA.models[run.experiment]}</th><td>${run.checkpoint || "—"}</td><td>${pct(m.accuracy)}</td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td>${pct(m.f1)}</td><td>${pct(m.false_positive_rate)}</td><td>${pct(run.metrics.adversarial.false_positive_rate)}</td><td>${pct(m.yes_ratio)}</td><td>${m.tp} / ${m.fp} / ${m.tn} / ${m.fn}</td></tr>`;
    }).join("");
    table.innerHTML = `<thead><tr>${headers.map(header => `<th scope="col">${header}</th>`).join("")}</tr></thead><tbody>${body}</tbody>`;
  }

  function renderGapChart() {
    const host = $("#gap-chart");
    const pairs = [
      ["E0", getDev("E0"), fresh],
      ["E1-1000", getDev("E1", 1000), getRun("e1_checkpoint-1000_repope_256vt")],
      ["E2-1000", getDev("E2", 1000), getRun("e2_checkpoint-1000_repope_256vt")],
      ["E3-1000", getDev("E3", 1000), e3_256],
      ["E3-2000", getDev("E3", 2000), getRun("e3_exploratory_high_recall_checkpoint-2000_repope_256vt")],
      ["E4-1000", getDev("E4", 1000), getRun("e4_checkpoint-1000_repope_256vt")],
      ["E4-2000", getDev("E4", 2000), getRun("e4_exploratory_high_recall_checkpoint-2000_repope_256vt")],
    ];
    const width = 760;
    const height = 430;
    const pad = { left: 54, right: 22, top: 30, bottom: 65 };
    const x = index => pad.left + index * ((width - pad.left - pad.right) / (pairs.length - 1));
    const y = value => pad.top + ((10 - value) / 10) * (height - pad.top - pad.bottom);
    let markup = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="代表配置从 Dev 到 RePOPE 的 FPR 上升">`;
    [0, 2, 4, 6, 8, 10].forEach(tick => {
      markup += `<line class="svg-grid-light" x1="${pad.left}" y1="${y(tick)}" x2="${width - pad.right}" y2="${y(tick)}"></line><text class="svg-axis-light" x="${pad.left - 8}" y="${y(tick) + 3}" text-anchor="end">${tick}%</text>`;
    });
    pairs.forEach(([label, dev, test], index) => {
      const dv = overall(dev).false_positive_rate * 100;
      const tv = overall(test).false_positive_rate * 100;
      markup += `<line x1="${x(index)}" y1="${y(dv)}" x2="${x(index)}" y2="${y(tv)}" stroke="#bcc8ce" stroke-width="5" stroke-linecap="round"></line>`;
      markup += `<circle cx="${x(index)}" cy="${y(dv)}" r="6" fill="#6d8491"><title>${label} Dev FPR ${dv.toFixed(2)}%</title></circle>`;
      markup += `<circle cx="${x(index)}" cy="${y(tv)}" r="7" fill="#9d6657"><title>${label} RePOPE FPR ${tv.toFixed(2)}%</title></circle>`;
      markup += `<text class="svg-axis-light" x="${x(index)}" y="${height - 30}" text-anchor="middle" transform="rotate(-28 ${x(index)} ${height - 30})">${label}</text>`;
    });
    markup += `<g transform="translate(${pad.left},10)"><circle cx="0" cy="0" r="5" fill="#6d8491"></circle><text class="svg-axis-light" x="10" y="4">Dev</text><circle cx="55" cy="0" r="5" fill="#9d6657"></circle><text class="svg-axis-light" x="65" y="4">RePOPE</text></g></svg>`;
    host.replaceChildren(svg(markup));
  }

  let activeErrorModel = "E0";
  let lastDialogTrigger = null;

  function renderErrorTabs() {
    const host = $("#error-tabs");
    host.innerHTML = Object.keys(DATA.models).map((model, index) => `<button class="error-tab" type="button" role="tab" id="tab-${model.toLowerCase()}" aria-controls="error-grid" aria-selected="${model === activeErrorModel}" tabindex="${model === activeErrorModel ? 0 : -1}" data-model="${model}">${model} · ${DATA.models[model]}</button>`).join("");
    $$(".error-tab", host).forEach((button, index, buttons) => {
      button.addEventListener("click", () => setErrorModel(button.dataset.model));
      button.addEventListener("keydown", event => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        let target = index;
        if (event.key === "ArrowLeft") target = (index - 1 + buttons.length) % buttons.length;
        if (event.key === "ArrowRight") target = (index + 1) % buttons.length;
        if (event.key === "Home") target = 0;
        if (event.key === "End") target = buttons.length - 1;
        setErrorModel(buttons[target].dataset.model);
        buttons[target].focus();
      });
    });
  }

  function setErrorModel(model) {
    activeErrorModel = model;
    $$(".error-tab").forEach(button => {
      const active = button.dataset.model === model;
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    renderErrorCases();
  }

  function renderErrorCases() {
    const summary = DATA.errorSummary[activeErrorModel];
    const focusLabel = DATA.models[activeErrorModel];
    $("#error-summary").innerHTML = `<h3>${activeErrorModel} · ${focusLabel}</h3><div class="error-summary-stats"><div><span>Total errors</span><strong>${summary.total}</strong></div><div><span>False positives</span><strong>${summary.fp}</strong></div><div><span>False negatives</span><strong>${summary.fn}</strong></div><div><span>Selected</span><strong>5</strong></div></div>`;
    const cases = DATA.errorCases.filter(item => item.focusModel === activeErrorModel);
    const grid = $("#error-grid");
    grid.setAttribute("aria-labelledby", `tab-${activeErrorModel.toLowerCase()}`);
    grid.innerHTML = cases.map(item => {
      const answers = Object.entries(item.answers).map(([model, answer]) => `<div class="answer-chip ${answer === item.truth ? "correct" : "wrong"}"><span>${model}</span><strong>${answer}</strong></div>`).join("");
      return `<article class="error-card reveal is-visible"><button class="case-image-button" type="button" data-case="${item.id}" aria-label="放大查看 ${item.focusModel} 错误案例：${item.question}"><img src="${item.image}" alt="${item.imageSource}，题目：${item.question}" loading="lazy"><span class="image-meta"><span>${item.imageSource.replace("COCO_val2014_", "COCO ")}</span><span>点击查看原图 ↗</span></span></button><div class="case-body"><div class="case-tags"><span class="case-tag ${item.errorType.toLowerCase()}">${item.errorType}</span><span class="case-tag">${item.split}</span>${item.sharedFailure ? `<span class="case-tag shared">共同失败</span>` : ""}</div><h3>${item.question}</h3><div class="truth-row"><span>RePOPE truth · <strong>${item.truth.toUpperCase()}</strong></span><span>${item.focusModel} · <strong>${item.answers[item.focusModel].toUpperCase()}</strong></span></div><div class="answer-strip" aria-label="五个模型的答案">${answers}</div><p class="case-diagnostic">${item.diagnostic}</p></div></article>`;
    }).join("");
    $$('[data-case]', grid).forEach(button => button.addEventListener("click", () => openCase(button.dataset.case, button)));
  }

  function openCase(id, trigger) {
    const item = DATA.errorCases.find(entry => entry.id === id);
    const dialog = $("#case-dialog");
    lastDialogTrigger = trigger;
    const image = $(".dialog-media img", dialog);
    image.src = item.image;
    image.alt = `${item.imageSource}：${item.question}`;
    $(".dialog-kicker", dialog).textContent = `${item.focusModel} · ${item.split} · ${item.errorType}${item.sharedFailure ? " · shared failure" : ""}`;
    $("#dialog-title", dialog).textContent = item.question;
    $(".dialog-truth", dialog).innerHTML = `RePOPE 真值：<strong>${item.truth.toUpperCase()}</strong> · 焦点模型答案：<strong>${item.answers[item.focusModel].toUpperCase()}</strong>`;
    $(".dialog-answers", dialog).innerHTML = Object.entries(item.answers).map(([model, answer]) => `<div class="dialog-answer ${answer === item.truth ? "correct" : "wrong"}"><span>${model} · ${DATA.models[model]}</span><strong>${answer}</strong></div>`).join("");
    $(".dialog-diagnostic", dialog).textContent = item.diagnostic;
    $(".dialog-source", dialog).textContent = `${item.imageSource} · RePOPE ${item.split} · question_id ${item.questionId}`;
    dialog.showModal();
    $(".dialog-close", dialog).focus();
  }

  function setupDialog() {
    const dialog = $("#case-dialog");
    const close = () => dialog.close();
    $(".dialog-close", dialog).addEventListener("click", close);
    dialog.addEventListener("click", event => {
      const bounds = dialog.getBoundingClientRect();
      const outside = event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom;
      if (outside) close();
    });
    dialog.addEventListener("close", () => {
      if (lastDialogTrigger) lastDialogTrigger.focus();
    });
  }

  function renderTransitionFlow() {
    const host = $("#transition-flow");
    const t = DATA.ablations.visualTokens.overall;
    const rows = [
      ["FN → TP", t.FN_TO_TP, "corrected", "#53776e"],
      ["FP → TN", t.FP_TO_TN, "corrected", "#6d8491"],
      ["TP → FN", t.TP_TO_FN, "regressed", "#b5a588"],
      ["TN → FP", t.TN_TO_FP, "regressed", "#9d6657"],
    ];
    const width = 700;
    const height = 350;
    const max = Math.max(...rows.map(row => row[1]));
    let markup = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="256 到 512 visual tokens 的四类预测迁移">`;
    rows.forEach((row, index) => {
      const y = 48 + index * 78;
      const thickness = 10 + (row[1] / max) * 22;
      markup += `<text class="svg-label-dark" x="12" y="${y + 4}">${row[0]}</text>`;
      markup += `<path d="M120 ${y} C280 ${y}, 385 ${y}, 540 ${y}" fill="none" stroke="${row[3]}" stroke-width="${thickness}" stroke-linecap="round" opacity=".78"></path>`;
      markup += `<circle cx="120" cy="${y}" r="${thickness / 2 + 3}" fill="${row[3]}"></circle><circle cx="540" cy="${y}" r="${thickness / 2 + 3}" fill="${row[3]}"></circle>`;
      markup += `<text x="580" y="${y + 7}" fill="${row[3]}" font-family="Georgia" font-size="22">${row[1]}</text><text class="svg-axis-light" x="624" y="${y + 4}">${row[2]}</text>`;
    });
    markup += "</svg>";
    host.replaceChildren(svg(markup));
  }

  function renderSplitComparison() {
    const host = $("#split-comparison");
    host.innerHTML = ["random", "popular", "adversarial"].map(split => {
      const before = e3_256.metrics[split];
      const after = e3_512.metrics[split];
      return `<article class="split-card"><h3>${cap(split)}</h3><div class="split-pair"><span>Recall</span><strong>${pp(after.recall - before.recall)}</strong></div><div class="split-pair"><span>FPR</span><strong>${pp(after.false_positive_rate - before.false_positive_rate)}</strong></div><div class="split-pair"><span>F1</span><strong>${pp(after.f1 - before.f1)}</strong></div><div class="split-pair"><span>Net correct</span><strong>+${after.correct - before.correct}</strong></div></article>`;
    }).join("");
  }

  function setupReveal() {
    const nodes = $$(".reveal");
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window)) {
      nodes.forEach(node => node.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { threshold: .15 });
    nodes.forEach((node, index) => {
      node.style.transitionDelay = `${Math.min(index % 4, 3) * .08}s`;
      observer.observe(node);
    });
  }

  function setupNavigation() {
    const topbar = $(".topbar");
    const menu = $(".menu-button");
    const links = $(".nav-links");
    const closeMenu = () => {
      links.classList.remove("open");
      menu.setAttribute("aria-expanded", "false");
      document.body.classList.remove("menu-open");
    };
    window.addEventListener("scroll", () => topbar.classList.toggle("scrolled", window.scrollY > 40), { passive: true });
    menu.addEventListener("click", () => {
      const open = menu.getAttribute("aria-expanded") !== "true";
      menu.setAttribute("aria-expanded", String(open));
      links.classList.toggle("open", open);
      document.body.classList.toggle("menu-open", open);
    });
    $$("a", links).forEach(link => link.addEventListener("click", closeMenu));
    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && links.classList.contains("open")) {
        closeMenu();
        menu.focus();
      }
    });
  }

  function verifyCanonicalData() {
    const m = overall(fresh);
    const tuple = [m.tp, m.fp, m.tn, m.fn].join("/");
    if (tuple !== "3062/187/4459/477") throw new Error(`Fresh baseline mismatch: ${tuple}`);
    if (DATA.errorCases.length !== 25) throw new Error("Error atlas must contain 25 cases.");
    const images = new Set(DATA.errorCases.map(item => item.imageSource));
    if (images.size !== 25) throw new Error("Error atlas image sources must be unique.");
  }

  function init() {
    verifyCanonicalData();
    renderHeroMiniChart();
    renderFrontier();
    renderMetricRibbon();
    renderRepopeTable();
    renderDevCharts();
    renderDevTable();
    renderGapChart();
    renderErrorTabs();
    renderErrorCases();
    setupDialog();
    renderTransitionFlow();
    renderSplitComparison();
    setupReveal();
    setupNavigation();
    window.__reportReady = true;
  }

  document.addEventListener("DOMContentLoaded", init);
})();
