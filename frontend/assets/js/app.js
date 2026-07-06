/* ==========================================================================
   EaseMind 共享客户端 JS：token 管理、统一请求、UI 工具
   ========================================================================== */

const App = (() => {
  const TOKEN_KEY = "easemind_token";
  const USER_KEY = "easemind_user";

  // ---- 存储 ----
  function getToken() { return localStorage.getItem(TOKEN_KEY); }
  function setToken(t) { localStorage.setItem(TOKEN_KEY, t); }
  function getUser() {
    try { return JSON.parse(localStorage.getItem(USER_KEY) || "null"); }
    catch { return null; }
  }
  function setUser(u) { localStorage.setItem(USER_KEY, JSON.stringify(u)); }
  function clearAuth() { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(USER_KEY); }

  // ---- 统一请求 ----
  async function request(url, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (!(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;

    let body = options.body;
    if (body && typeof body !== "string" && !(body instanceof FormData)) {
      body = JSON.stringify(body);
    }

    const resp = await fetch(url, { ...options, headers, body });

    if (resp.status === 401) {
      clearAuth();
      redirectLogin();
      throw new Error("未登录或登录已失效");
    }

    const contentType = resp.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const data = await resp.json();
      if (!resp.ok) {
        // FastAPI 422 验证错误：detail 是数组 [{msg, ...}]，需要展开成可读字符串
        let msg = data.detail || data.message || `请求失败 (${resp.status})`;
        if (Array.isArray(msg)) {
          msg = msg.map(e => e.msg || JSON.stringify(e)).join("; ");
        } else if (typeof msg === "object") {
          msg = JSON.stringify(msg);
        }
        throw new Error(msg);
      }
      return data;
    }
    if (!resp.ok) throw new Error(`请求失败 (${resp.status})`);
    return resp;
  }

  const api = {
    get: (url) => request(url, { method: "GET" }),
    post: (url, body) => request(url, { method: "POST", body }),
    patch: (url, body) => request(url, { method: "PATCH", body }),
    delete: (url) => request(url, { method: "DELETE" }),
    upload: (url, formData) => request(url, { method: "POST", body: formData }),
  };

  // ---- 路由 ----
  function redirectLogin() {
    window.location.href = "/pages/auth.html";
  }
  function redirectHome() {
    const u = getUser();
    window.location.href = u && u.role === "admin" ? "/pages/admin-dashboard.html" : "/pages/chat.html";
  }

  function requireAuth() {
    if (!getToken()) { redirectLogin(); return false; }
    return true;
  }
  function requireAdmin() {
    const u = getUser();
    if (!getToken() || !u) { redirectLogin(); return false; }
    if (u.role !== "admin") {
      window.location.href = "/pages/chat.html";
      return false;
    }
    return true;
  }

  // ---- Toast 通知 ----
  function ensureToastContainer() {
    let c = document.querySelector(".toast-container");
    if (!c) {
      c = document.createElement("div");
      c.className = "toast-container";
      document.body.appendChild(c);
    }
    return c;
  }
  function toast(message, type = "info", duration = 3000) {
    const c = ensureToastContainer();
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = message;
    c.appendChild(el);
    setTimeout(() => {
      el.style.transition = "opacity 0.3s, transform 0.3s";
      el.style.opacity = "0";
      el.style.transform = "translateX(20px)";
      setTimeout(() => el.remove(), 300);
    }, duration);
  }

  // ---- 模态框 ----
  function modal({ title, body, footer }) {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    const m = document.createElement("div");
    m.className = "modal";
    m.innerHTML = `
      <div class="modal-head">
        <div class="modal-title">${title || ""}</div>
        <button class="btn btn-ghost btn-sm" data-close>关闭</button>
      </div>
      <div class="modal-body"></div>
      ${footer ? `<div class="modal-foot"></div>` : ""}
    `;
    m.querySelector(".modal-body").appendChild(body);
    if (footer) m.querySelector(".modal-foot").appendChild(footer);
    overlay.appendChild(m);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay || e.target.dataset.close !== undefined) overlay.remove();
    });
    document.body.appendChild(overlay);
    return { overlay, close: () => overlay.remove() };
  }

  // ---- 侧边栏 ----
  function buildSidebar(activeKey) {
    const user = getUser();
    if (!user) return "";
    const isAdmin = user.role === "admin";
    const initials = user.username.slice(0, 1).toUpperCase();

    const navItems = isAdmin ? [
      { key: "dashboard", label: "概览", icon: "◈", href: "/pages/admin-dashboard.html" },
      { key: "models", label: "模型管理", icon: "❖", href: "/pages/models.html" },
      { key: "datasets", label: "数据集", icon: "▤", href: "/pages/datasets.html" },
      { key: "training", label: "模型训练", icon: "✦", href: "/pages/training.html" },
      { key: "finetune", label: "傻瓜微调", icon: "✿", href: "/pages/finetune.html" },
      { key: "chat", label: "在线对话", icon: "💬", href: "/pages/chat.html" },
    ] : [
      { key: "chat", label: "在线对话", icon: "💬", href: "/pages/chat.html" },
    ];

    const itemsHtml = navItems.map(it => `
      <a class="nav-item ${it.key === activeKey ? "active" : ""}" href="${it.href}">
        <span class="nav-icon">${it.icon}</span>
        <span>${it.label}</span>
      </a>
    `).join("");

    const adminSection = isAdmin ? `
      <div class="nav-section-label">管理</div>
      ${itemsHtml}
      <div class="nav-section-label">对话</div>
      <a class="nav-item ${activeKey === 'chat' ? 'active' : ''}" href="/pages/chat.html">
        <span class="nav-icon">💬</span><span>在线对话</span>
      </a>
    ` : `
      <div class="nav-section-label">功能</div>
      ${itemsHtml}
    `;

    // 简化：直接用单一列表
    const simpleItems = isAdmin ? [
      { key: "dashboard", label: "概览", icon: "◈", href: "/pages/admin-dashboard.html" },
      { key: "models", label: "模型管理", icon: "❖", href: "/pages/models.html" },
      { key: "datasets", label: "数据集", icon: "▤", href: "/pages/datasets.html" },
      { key: "training", label: "模型训练", icon: "✦", href: "/pages/training.html" },
      { key: "finetune", label: "傻瓜微调", icon: "✿", href: "/pages/finetune.html" },
      { key: "distillation", label: "模型蒸馏", icon: "🔬", href: "/pages/distillation.html" },
      { key: "chat", label: "在线对话", icon: "💬", href: "/pages/chat.html" },
      { key: "apikeys", label: "API 密钥", icon: "🔑", href: "/pages/apikeys.html" },
    ] : [
      { key: "chat", label: "在线对话", icon: "💬", href: "/pages/chat.html" },
      { key: "apikeys", label: "API 密钥", icon: "🔑", href: "/pages/apikeys.html" },
    ];

    const finalItems = simpleItems.map(it => `
      <a class="nav-item ${it.key === activeKey ? "active" : ""}" href="${it.href}">
        <span class="nav-icon">${it.icon}</span>
        <span>${it.label}</span>
      </a>
    `).join("");

    return `
      <aside class="sidebar">
        <div class="sidebar-brand">
          <div class="logo">E</div>
          <div>
            <div class="brand-name">EaseMind</div>
            <div class="brand-sub">AI 训练平台</div>
          </div>
        </div>
        <nav class="sidebar-nav">
          <div class="nav-section-label">${isAdmin ? "管理工作台" : "我的工作台"}</div>
          ${finalItems}
        </nav>
        <div class="sidebar-footer">
          <div class="user-chip">
            <div class="user-avatar">${initials}</div>
            <div class="user-info">
              <div class="user-name">${escapeHtml(user.username)}</div>
              <div class="user-role">${isAdmin ? "管理员" : "普通用户"}</div>
            </div>
            <button class="btn btn-ghost btn-sm" title="退出登录" onclick="App.logout()">退出</button>
          </div>
        </div>
      </aside>
    `;
  }

  function buildTopbar(title) {
    return `
      <header class="topbar">
        <div class="topbar-title">${escapeHtml(title)}</div>
        <div class="topbar-actions">
          <a class="btn btn-ghost btn-sm" href="/docs" target="_blank">API 文档</a>
        </div>
      </header>
    `;
  }

  function renderShell(activeKey, title, contentHtml) {
    document.body.innerHTML = `
      <div class="app-shell">
        ${buildSidebar(activeKey)}
        <div class="main-area">
          ${buildTopbar(title)}
          <div class="content">${contentHtml}</div>
        </div>
      </div>
    `;
  }

  // ---- 工具 ----
  function escapeHtml(s) {
    if (s == null) return "";
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  function formatTime(s) {
    if (!s) return "-";
    const d = new Date(s);
    if (isNaN(d)) return "-";
    return d.toLocaleString("zh-CN", { hour12: false });
  }

  function statusBadge(status) {
    const map = {
      ready: ["badge-success", "就绪"],
      pending: ["badge-warning", "等待中"],
      downloading: ["badge-accent", "下载中"],
      running: ["badge-accent", "训练中"],
      completed: ["badge-success", "已完成"],
      failed: ["badge-danger", "失败"],
      cancelled: ["badge-neutral", "已取消"],
    };
    const [cls, label] = map[status] || ["badge-neutral", status];
    return `<span class="badge ${cls}">${label}</span>`;
  }

  function logout() {
    clearAuth();
    redirectLogin();
  }

  // SSE 流式读取
  async function streamSSE(url, onMessage) {
    const token = getToken();
    const resp = await fetch(url, {
      headers: { "Authorization": `Bearer ${token}`, "Accept": "text/event-stream" },
    });
    if (!resp.ok || !resp.body) {
      throw new Error(`流式请求失败 (${resp.status})`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (line.startsWith("data:")) {
          const data = line.slice(5).trim();
          try { onMessage(JSON.parse(data)); }
          catch { onMessage({ type: "raw", content: data }); }
        }
      }
    }
  }

  // 带 body 的 SSE（用 POST），通过 fetch + ReadableStream 实现
  async function streamSSEPost(url, body, onMessage) {
    const token = getToken();
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
      },
      body: JSON.stringify(body),
    });
    if (!resp.ok || !resp.body) {
      const txt = await resp.text().catch(() => "");
      throw new Error(`流式请求失败 (${resp.status}) ${txt}`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (line.startsWith("data:")) {
          const data = line.slice(5).trim();
          try { onMessage(JSON.parse(data)); }
          catch { onMessage({ type: "raw", content: data }); }
        }
      }
    }
  }

  function drawLossChart(container, points, options) {
    options = options || {};
    const showDistill = options.showDistill || false;
    const W = 600, H = 240, PAD = { l: 50, r: 16, t: 20, b: 36 };
    container.innerHTML = "";
    if (!points || points.length === 0) {
      container.innerHTML = '<div class="empty-state" style="padding:40px"><div class="empty-icon">📉</div><div class="empty-text">暂无 loss 数据</div></div>';
      return;
    }
    const xs = points.map(p => p.step);
    const lossVals = points.map(p => p.loss);
    const distillVals = showDistill ? points.map(p => p.distill_loss).filter(v => v != null) : [];
    const ceVals = showDistill ? points.map(p => p.ce_loss).filter(v => v != null) : [];
    const allVals = [...lossVals, ...distillVals, ...ceVals];
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    let yMin = Math.min(...allVals), yMax = Math.max(...allVals);
    if (yMin === yMax) { yMin -= 0.1; yMax += 0.1; }
    const yPad = (yMax - yMin) * 0.1;
    yMin -= yPad; yMax += yPad;
    const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b;
    const sx = x => PAD.l + ((x - xMin) / (xMax - xMin || 1)) * plotW;
    const sy = y => PAD.t + plotH - ((y - yMin) / (yMax - yMin || 1)) * plotH;
    const path = (vals, key) => points.filter(p => p[key] != null)
      .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.step).toFixed(1)},${sy(p[key]).toFixed(1)}`).join(" ");
    let gridLines = "";
    for (let i = 0; i <= 4; i++) {
      const y = PAD.t + (plotH / 4) * i;
      const val = (yMax - (yMax - yMin) * (i / 4)).toFixed(3);
      gridLines += `<line x1="${PAD.l}" y1="${y}" x2="${W - PAD.r}" y2="${y}" stroke="var(--border)" stroke-width="0.5" stroke-dasharray="2,3"/>`;
      gridLines += `<text x="${PAD.l - 6}" y="${y + 3}" text-anchor="end" font-size="9" fill="var(--text-muted)">${val}</text>`;
    }
    let xLabels = "";
    for (let i = 0; i <= 4; i++) {
      const x = PAD.l + (plotW / 4) * i;
      const val = Math.round(xMin + (xMax - xMin) * (i / 4));
      xLabels += `<text x="${x}" y="${H - PAD.b + 16}" text-anchor="middle" font-size="9" fill="var(--text-muted)">${val}</text>`;
    }
    const colors = { loss: "#d97757", distill: "#5b8def", ce: "#22a06b" };
    let paths = `<path d="${path(lossVals, "loss")}" fill="none" stroke="${colors.loss}" stroke-width="2"/>`;
    if (showDistill) {
      paths += `<path d="${path(distillVals, "distill_loss")}" fill="none" stroke="${colors.distill}" stroke-width="1.5" stroke-dasharray="4,2"/>`;
      paths += `<path d="${path(ceVals, "ce_loss")}" fill="none" stroke="${colors.ce}" stroke-width="1.5" stroke-dasharray="4,2"/>`;
    }
    let legend = showDistill
      ? `<g transform="translate(${PAD.l},4)"><circle cx="0" cy="4" r="3" fill="${colors.loss}"/><text x="8" y="7" font-size="9" fill="var(--text-muted)">total</text><circle cx="50" cy="4" r="3" fill="${colors.distill}"/><text x="58" y="7" font-size="9" fill="var(--text-muted)">distill</text><circle cx="100" cy="4" r="3" fill="${colors.ce}"/><text x="108" y="7" font-size="9" fill="var(--text-muted)">ce</text></g>`
      : "";
    const svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block">
      ${gridLines}${xLabels}${paths}${legend}
      <text x="${PAD.l}" y="${H - 4}" font-size="9" fill="var(--text-muted)">step →</text>
      <text x="10" y="${PAD.t + 4}" font-size="9" fill="var(--text-muted)" transform="rotate(-90 10 ${PAD.t + 4})">loss →</text>
    </svg>`;
    container.innerHTML = svg;
  }

  return {
    getToken, getUser, setToken, setUser, clearAuth,
    api, requireAuth, requireAdmin, redirectLogin, redirectHome,
    toast, modal, renderShell, buildSidebar, buildTopbar,
    escapeHtml, formatTime, statusBadge, logout,
    streamSSE, streamSSEPost, drawLossChart,
  };
})();
