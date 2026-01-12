const API_BASE = "/api"; // 使用相对路径，因为前端由同一后端托管
let priceChart = null;

function useSymbol(symbol) {
    document.getElementById('symbol-input').value = symbol;
    analyzeStock();
}

async function analyzeStock() {
    const symbol = document.getElementById('symbol-input').value.trim();
    if (!symbol) return;

    // UI States
    const loadingEl = document.getElementById('loading');
    const resultsEl = document.getElementById('results');
    const btn = document.getElementById('analyze-btn');

    loadingEl.classList.remove('hidden');
    resultsEl.classList.add('hidden');
    btn.disabled = true;

    try {
        // 调用分析接口
        const response = await fetch(`${API_BASE}/analyze/${symbol}`);
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || `请求失败 (${response.status})`);
        }
        const data = await response.json();
        renderResults(data);
    } catch (e) {
        alert("分析过程出错: " + e.message);
        console.error(e);
    } finally {
        loadingEl.classList.add('hidden');
        btn.disabled = false;
    }
}

function renderResults(data) {
    document.getElementById('results').classList.remove('hidden');

    // 1. Prediction Card
    const trend = data.data.predicted_trend; // "UP" or "DOWN"
    const trendEl = document.getElementById('pred-trend');
    trendEl.textContent = trend === 'UP' ? '看涨' : '看跌';
    trendEl.className = `prediction-value ${trend}`;

    document.getElementById('curr-price').textContent = data.data.current_price.toFixed(2);
    document.getElementById('confidence').textContent = data.data.confidence.toFixed(1);

    // 2. Chart
    renderChart(data.data.history_dates, data.data.history_prices);

    // 3. Report
    const reportContent = data.report || "No report generated.";
    document.getElementById('report-content').innerHTML = marked.parse(reportContent);
}

function renderChart(dates, prices) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    if (priceChart) {
        priceChart.destroy();
    }

    // 创建线性渐变背景
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
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    grid: {
                        color: '#334155',
                        borderDash: [5, 5]
                    },
                    ticks: {
                        color: '#94a3b8',
                        font: { family: 'Inter' }
                    }
                }
            }
        }
    });
}
