const API_BASE = "/api";
let priceChart = null;

async function loginUser(username, password, remember) {
    const response = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            username: username,
            password: password
        })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
        throw new Error(data.detail || '登录失败');
    }
    
    if (data.token) {
        localStorage.setItem('access_token', data.token);
        localStorage.setItem('user_info', JSON.stringify(data.user || {}));
        if (remember) {
            localStorage.setItem('remember_me', 'true');
        }
        return { success: true, data: data };
    }
    
    return { success: false, message: '登录失败' };
}

async function registerUser(email, username, password) {
    const response = await fetch(`${API_BASE}/register`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            email: email,
            username: username,
            password: password,
            full_name: username
        })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
        throw new Error(data.detail || '注册失败');
    }
    
    return { success: true, data: data };
}

async function requestPasswordReset(email) {
    const response = await fetch(`${API_BASE}/auth/password-reset/request`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: email })
    });
    
    if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || '请求失败');
    }
    
    return { success: true };
}

function logout() {
    const token = localStorage.getItem('access_token');
    if (token) {
        fetch(`${API_BASE}/auth/logout`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        }).catch(() => {});
    }
    
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user_info');
    localStorage.removeItem('remember_me');
    
    window.location.href = 'login.html';
}

function getAuthHeaders() {
    const token = localStorage.getItem('access_token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
}

async function fetchWithAuth(url, options = {}) {
    const headers = {
        ...options.headers,
        ...getAuthHeaders()
    };
    
    const response = await fetch(url, { ...options, headers });
    
    if (response.status === 401) {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        localStorage.removeItem('user_info');
        window.location.href = 'login.html';
        throw new Error('认证已过期，请重新登录');
    }
    
    return response;
}

function checkAuthStatus() {
    const accessToken = localStorage.getItem('access_token');
    const userInfo = localStorage.getItem('user_info');
    
    const loginLink = document.getElementById('login-link');
    const registerLink = document.getElementById('register-link');
    const userInfoEl = document.getElementById('user-info');
    const usernameDisplay = document.getElementById('username-display');
    const userAvatar = document.getElementById('user-avatar');
    const userName = document.getElementById('user-name');
    const userRole = document.getElementById('user-role');
    
    if (accessToken) {
        if (loginLink) loginLink.classList.add('hidden');
        if (registerLink) registerLink.classList.add('hidden');
        if (userInfoEl) userInfoEl.classList.remove('hidden');
        
        if (userInfo) {
            try {
                const user = JSON.parse(userInfo);
                if (usernameDisplay) usernameDisplay.textContent = user.username || user.email || '用户';
                if (userAvatar) userAvatar.textContent = (user.username || user.email || 'U').charAt(0).toUpperCase();
                if (userName) userName.textContent = user.username || '用户';
                if (userRole) userRole.textContent = user.role || '投资者';
            } catch (e) {}
        }
    } else {
        if (loginLink) loginLink.classList.remove('hidden');
        if (registerLink) registerLink.classList.remove('hidden');
        if (userInfoEl) userInfoEl.classList.add('hidden');
        
        if (userAvatar) userAvatar.textContent = '?';
        if (userName) userName.textContent = '未登录';
        if (userRole) userRole.textContent = '请先登录';
    }
}

function showLoading(text = "正在加载...") {
    const loader = document.getElementById('loading-overlay');
    const msg = document.getElementById('loading-text');
    if (loader) {
        if (msg) msg.textContent = text;
        loader.classList.remove('hidden');
        loader.style.display = 'flex';
    }
}

function hideLoading() {
    const loader = document.getElementById('loading-overlay');
    if (loader) {
        loader.classList.add('hidden');
        loader.style.display = 'none';
    }
}

function updateProgress(progress, status) {
    const container = document.getElementById('progress-container');
    const indicator = document.getElementById('progress-bar');
    const statusText = document.getElementById('progress-status');
    const percentText = document.getElementById('progress-percent');

    if (container && container.classList.contains('hidden')) {
        container.classList.remove('hidden');
    }

    if (indicator) indicator.style.width = `${progress}%`;
    if (statusText) statusText.textContent = status;
    if (percentText) percentText.textContent = `${progress}%`;

    if (progress >= 100) {
        setTimeout(() => {
            if (container) container.classList.add('hidden');
            if (indicator) indicator.style.width = '0%';
            if (percentText) percentText.textContent = '0%';
        }, 1000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    checkAuthStatus();
    
    const path = window.location.pathname;
    if (path.includes('factors.html')) {
        initFactorsPage();
    } else if (path.includes('data.html')) {
        setTimeout(loadMonitoredStocks, 100);
    }
    
    initFactorTabs();
    
    const urlParams = new URLSearchParams(window.location.search);
    const symbolParam = urlParams.get('symbol');
    if (symbolParam) {
        const symbolInput = document.getElementById('symbol-input');
        if (symbolInput) {
            symbolInput.value = symbolParam;
        }
    }
});

function useSymbol(symbol) {
    const symbolInput = document.getElementById('symbol-input');
    if (symbolInput) {
        symbolInput.value = symbol;
        analyzeStock();
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

    const titleEl = document.getElementById('stock-title');
    if (titleEl) titleEl.textContent = `${symbol} 量化因子深度分析`;
    showLoading(`正在获取 ${symbol} 的详细因子数据...`);
    await fetchFactorsData(symbol);
    hideLoading();
}

async function fetchFactorsData(symbol, category = 'all') {
    try {
        const catParam = category === 'all' ? '' : `?cat=${category}`;
        const response = await fetch(`${API_BASE}/factors/${symbol}${catParam}`);
        if (!response.ok) return;
        const data = await response.json();

        renderFactors(data, category !== 'all');

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
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('tab-btn')) {
            const btn = e.target;
            const container = btn.closest('.card, .container');
            if (!container) return;

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
    }
}

function getSignalClass(signal) {
    if (!signal) return '';
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
    const body = document.getElementById('history-body') || document.getElementById('history-table');
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
            <td>${(item.volume / 10000).toFixed(0)} 万</td>
        `;
        body.appendChild(row);
    });
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
            scales: { x: { display: false }, y: { grid: { color: '#e2e8f0' }, ticks: { color: '#94a3b8' } } }
        }
    });
}

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
                <td><span class="badge ${stock.status === '已缓存' ? 'badge-success' : 'badge-warning'}">${stock.status}</span></td>
                <td>${stock.last_sync}</td>
                <td>
                    <button class="btn btn-sm btn-outline" onclick="syncStock('${stock.symbol}')">同步</button>
                    <button class="btn btn-sm btn-danger" onclick="removeStockFromPool('${stock.symbol}')">删除</button>
                </td>
            `;
            body.appendChild(row);
        });
    } catch (e) {
        console.error("加载监控列表失败:", e);
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
