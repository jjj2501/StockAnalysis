const API_BASE = "/api"; // 使用相对路径，因为前端由同一后端托管
let priceChart = null;

// 页面加载时的路由逻辑
document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;
    if (path.includes('factors.html')) {
        initFactorsPage();
    } else if (path.includes('data.html')) {
        // 延迟一小会儿确保 DOM 渲染完成（虽然 DOMContentLoaded 已经触发）
        setTimeout(loadMonitoredStocks, 100);
    }
    initFactorTabs();
});

function showLoading(text = "正在努力分析中...") {
    const loader = document.getElementById('global-loading');
    const msg = document.getElementById('loading-msg');
    if (loader && msg) {
        msg.textContent = text;
        loader.classList.remove('hidden');
    }
}

function hideLoading() {
    const loader = document.getElementById('global-loading');
    if (loader) {
        loader.classList.add('hidden');
    }
}

async function initFactorsPage() {
    const params = new URLSearchParams(window.location.search);
    const symbol = params.get('symbol');
    if (!symbol) {
        alert("未发现股票代码");
        window.location.href = 'index.html';
        return;
    }

    document.getElementById('stock-title').textContent = `${symbol} 量化因子深度分析`;
    showLoading(`正在获取 ${symbol} 的详细因子数据...`);
    await fetchFactorsData(symbol);
    hideLoading();
}

function useSymbol(symbol) {
    document.getElementById('symbol-input').value = symbol;
    analyzeStock();
}

async function analyzeStock() {
    const symbol = document.getElementById('symbol-input').value.trim();
    if (!symbol) return;

    // 更新详情页链接
    const detailLink = document.getElementById('view-factors-btn');
    if (detailLink) {
        detailLink.href = `factors.html?symbol=${symbol}`;
        detailLink.classList.remove('hidden');
    }

    // UI States
    showLoading(`正在分析 ${symbol} ... 请稍候`);
    const resultsEl = document.getElementById('results');
    const btn = document.getElementById('analyze-btn');

    resultsEl.classList.add('hidden');
    if (btn) btn.disabled = true;

    try {
        // 并行调用接口
        const responsePromise = fetch(`${API_BASE}/analyze/${symbol}`);
        const historyPromise = fetchHistoryData(symbol);
        const factorsPromise = fetchFactorsData(symbol);

        const response = await responsePromise;
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || `分析请求失败 (${response.status})`);
        }
        const data = await response.json();
        renderResults(data);

        await Promise.all([historyPromise, factorsPromise]);
    } catch (e) {
        alert("过程出错: " + e.message);
        console.error(e);
    } finally {
        hideLoading();
        if (btn) btn.disabled = false;
    }
}

async function fetchFactorsData(symbol, category = 'all') {
    try {
        const catParam = category === 'all' ? '' : `?cat=${category}`;
        const response = await fetch(`${API_BASE}/factors/${symbol}${catParam}`);
        if (!response.ok) return;
        const data = await response.json();

        // 渲染因子
        renderFactors(data, category !== 'all');

        // 如果在独立页面，更新日期
        const dateEl = document.getElementById('update-date');
        if (dateEl) dateEl.textContent = data.date;

        const infoEl = document.getElementById('stock-info');
        if (infoEl && data.date) {
            infoEl.textContent = `最新分析日期: ${data.date}`;
        }
    } catch (e) {
        console.error("Failed to fetch factors:", e);
    }
}

function initFactorTabs() {
    // 监听所有 Tab 按钮的点击
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('tab-btn')) {
            const btn = e.target;
            const container = btn.closest('.card, .container');
            if (!container) return;

            // 切换 active 状态
            container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const category = btn.getAttribute('data-cat');
            const symbolInput = document.getElementById('symbol-input');
            const symbol = (symbolInput && symbolInput.value.trim()) ||
                new URLSearchParams(window.location.search).get('symbol');

            if (symbol) {
                const loadingText = category === 'all' ? '正在获取全部因子...' : `正在获取 ${btn.textContent} 数据...`;
                showLoading(loadingText);
                fetchFactorsData(symbol, category).finally(() => hideLoading());
            }
        }
    });
}

function renderFactors(data, isPartial = false) {
    const categories = {
        'tech-factors': data.technical,
        'fundamental-factors': data.fundamental,
        'sentiment-factors': data.sentiment,
        'northbound-factors': data.northbound
    };

    for (const [id, factors] of Object.entries(categories)) {
        const groupEl = document.getElementById(id);
        const container = groupEl?.querySelector('.factor-list');
        if (!container) continue;

        if (!isPartial || (factors && Object.keys(factors).length > 0)) {
            container.innerHTML = '';
            if (isPartial && (!factors || Object.keys(factors).length === 0)) continue;

            if (!factors || factors.status === "暂无数据" || Object.keys(factors).length === 0) {
                container.innerHTML = '<div class="factor-item"><span class="factor-name">暂无相关因子数据</span></div>';
            } else {
                for (const [name, info] of Object.entries(factors)) {
                    if (typeof info !== 'object') continue;
                    const div = document.createElement('div');
                    div.className = 'factor-item';
                    const sigClass = getSignalClass(info.signal);
                    div.innerHTML = `
                        <span class="factor-name">${name}</span>
                        <div class="factor-info">
                            <span class="factor-val">${info.value !== undefined ? info.value.toFixed(2) + (info.unit || '') : '--'}</span>
                            <span class="factor-sig ${sigClass}">${info.signal}</span>
                        </div>
                    `;
                    container.appendChild(div);
                }
            }
        }

        if (window.location.pathname.includes('factors.html')) {
            const activeTabBtn = document.querySelector('.tab-btn.active');
            if (activeTabBtn) {
                const activeTab = activeTabBtn.getAttribute('data-cat');
                if (activeTab !== 'all') {
                    const map = { 'technical': 'tech-factors', 'fundamental': 'fundamental-factors', 'sentiment': 'sentiment-factors', 'northbound': 'northbound-factors' };
                    groupEl.style.display = (map[activeTab] === id) ? 'block' : 'none';
                } else {
                    groupEl.style.display = 'block';
                }
            }
        }
    }
}

function getSignalClass(signal) {
    if (signal.includes('涨') || signal.includes('买') || signal.includes('强') || signal.includes('活跃') || signal.includes('放量') || signal.includes('低估') || signal.includes('优异')) return 'sig-up';
    if (signal.includes('跌') || signal.includes('卖') || signal.includes('弱') || signal.includes('低迷') || signal.includes('缩量') || signal.includes('高估') || signal.includes('资金流出')) return 'sig-down';
    return '';
}

async function fetchHistoryData(symbol) {
    try {
        const response = await fetch(`${API_BASE}/history/${symbol}?days=30`);
        if (!response.ok) return;
        const data = await response.json();
        renderHistoryTable(data.history);
    } catch (e) {
        console.error("Failed to fetch history:", e);
    }
}

function renderHistoryTable(history) {
    const body = document.getElementById('history-body');
    if (!body) return;
    body.innerHTML = '';

    [...history].reverse().forEach(item => {
        const row = document.createElement('tr');
        const isUp = item.close >= item.open;
        const priceClass = isUp ? 'price-up' : 'price-down';

        row.innerHTML = `
            <td>${item.date}</td>
            <td class="${priceClass}">${item.open.toFixed(2)}</td>
            <td class="${priceClass}">${item.high.toFixed(2)}</td>
            <td class="${priceClass}">${item.low.toFixed(2)}</td>
            <td class="${priceClass}">${item.close.toFixed(2)}</td>
            <td>${(item.volume / 10000).toFixed(2)} 万</td>
        `;
        body.appendChild(row);
    });
}

function renderResults(data) {
    document.getElementById('results').classList.remove('hidden');
    const trend = data.data.predicted_trend;
    const trendEl = document.getElementById('pred-trend');
    trendEl.textContent = trend === 'UP' ? '看涨' : '看跌';
    trendEl.className = `prediction-value ${trend}`;
    document.getElementById('curr-price').textContent = data.data.current_price.toFixed(2);
    document.getElementById('confidence').textContent = data.data.confidence.toFixed(1);
    renderChart(data.data.history_dates, data.data.history_prices);
    const reportContent = data.report || "No report generated.";
    document.getElementById('report-content').innerHTML = marked.parse(reportContent);
}

function renderChart(dates, prices) {
    const ctxEl = document.getElementById('priceChart');
    if (!ctxEl) return;
    const ctx = ctxEl.getContext('2d');
    if (priceChart) { priceChart.destroy(); }
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.5)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: '收盘价',
                data: prices,
                borderColor: '#3b82f6',
                borderWidth: 3,
                backgroundColor: gradient,
                fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
            scales: { x: { display: false }, y: { grid: { color: '#334155', borderDash: [5, 5] }, ticks: { color: '#94a3b8' } } }
        }
    });
}

// --- 数据管理中心逻辑 (NEW) ---

async function loadMonitoredStocks() {
    const body = document.getElementById('data-body');
    if (!body) return;

    try {
        const response = await fetch(`${API_BASE}/data/stocks`);
        const stocks = await response.json();

        body.innerHTML = '';
        if (stocks.length === 0) {
            body.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 2rem; color: var(--text-muted);">监控池为空，请添加股票</td></tr>';
            return;
        }

        stocks.forEach(stock => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${stock.symbol}</strong></td>
                <td><span class="factor-sig ${stock.status === '已缓存' ? 'sig-up' : ''}">${stock.status}</span></td>
                <td>${stock.last_sync}</td>
                <td class="table-actions">
                    <button class="btn-small" onclick="syncStock('${stock.symbol}')">同步</button>
                    <button class="btn-small danger" onclick="removeStockFromPool('${stock.symbol}')">删除</button>
                </td>
            `;
            body.appendChild(row);
        });
    } catch (e) {
        console.error("加载监控列表失败:", e);
    }
}

async function doAddStock() {
    const input = document.getElementById('modal-symbol');
    const symbol = input.value.trim();
    if (!symbol) return;

    showLoading("正在添加股票...");
    try {
        const response = await fetch(`${API_BASE}/data/stocks/${symbol}`, { method: 'POST' });
        if (response.ok) {
            hideAddModal();
            input.value = '';
            await loadMonitoredStocks();
        }
    } catch (e) {
        alert("添加失败");
    } finally {
        hideLoading();
    }
}

async function removeStockFromPool(symbol) {
    if (!confirm(`确认从监控池移除 ${symbol}？`)) return;

    try {
        const response = await fetch(`${API_BASE}/data/stocks/${symbol}`, { method: 'DELETE' });
        if (response.ok) {
            await loadMonitoredStocks();
        }
    } catch (e) {
        alert("删除失败");
    }
}

async function syncStock(symbol) {
    showLoading(`正在同步 ${symbol} 历史数据...`);
    try {
        const response = await fetch(`${API_BASE}/data/sync?symbol=${symbol}`, { method: 'POST' });
        if (response.ok) {
            await loadMonitoredStocks();
        }
    } catch (e) {
        alert("同步出错");
    } finally {
        hideLoading();
    }
}

async function syncAllStocks() {
    showLoading("正在全量同步监控池数据...");
    try {
        const response = await fetch(`${API_BASE}/data/sync`, { method: 'POST' });
        if (response.ok) {
            await loadMonitoredStocks();
        }
    } catch (e) {
        alert("同步出错");
    } finally {
        hideLoading();
    }
}

function showAddModal() { document.getElementById('add-modal').classList.remove('hidden'); }
function hideAddModal() { document.getElementById('add-modal').classList.add('hidden'); }
