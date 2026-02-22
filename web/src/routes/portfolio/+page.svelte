<script>
    import { onMount, tick } from "svelte";
    import Card from "$lib/components/Card.svelte";
    import { marked } from "marked"; // 用于渲染 AI Markdown 报告
    import {
        Chart,
        PieController,
        ArcElement,
        Tooltip,
        Legend,
    } from "chart.js";

    Chart.register(PieController, ArcElement, Tooltip, Legend);

    // 投资组合响应式状态 (包含全球多空资产)
    let portfolio = $state([
        {
            name: "苹果公司",
            symbol: "AAPL",
            shares: 50,
            price: 247.7,
            cost: 200.0,
            market: "US",
            currency: "USD",
            asset_type: "STOCK",
        },
        {
            name: "腾讯控股",
            symbol: "0700",
            shares: 500,
            price: 405.2,
            cost: 350.0,
            market: "HK",
            currency: "HKD",
            asset_type: "STOCK",
        },
        {
            name: "贵州茅台",
            symbol: "600519",
            shares: 100,
            price: 1680.5,
            cost: 1580.0,
            market: "CN",
            currency: "CNY",
            asset_type: "STOCK",
        },
        {
            name: "比特币核心",
            symbol: "BTC-USD",
            shares: 1,
            price: 95500.0,
            cost: 85000.0,
            market: "CRYPTO",
            currency: "USD",
            asset_type: "CRYPTO",
        },
        {
            name: "美元现金活期",
            symbol: "CASH_USD",
            shares: 100000,
            price: 1.0,
            cost: 1.0,
            market: "CASH",
            currency: "USD",
            asset_type: "CASH",
        },
    ]);

    /** @type {HTMLCanvasElement|null} */
    let chartCanvas = $state(null);
    /** @type {any} */
    let pieChartInstance = $state(null);

    // 计算总资产与盈亏
    let totalPositionValue = $derived(
        portfolio.reduce((sum, s) => sum + s.price * s.shares, 0),
    );
    let totalCost = $derived(
        portfolio.reduce((sum, s) => sum + s.cost * s.shares, 0),
    );
    let totalPnl = $derived(totalPositionValue - totalCost);
    let totalPnlPct = $derived(
        totalCost > 0 ? (totalPnl / totalCost) * 100 : 0,
    );

    // 风控状态
    let isAnalyzing = $state(false);
    let riskReport = $state(null);
    let riskError = $state("");

    async function analyzeRisk(forceRefresh = false) {
        if (isAnalyzing) return;
        isAnalyzing = true;
        if (!forceRefresh) {
            riskReport = null;
        }
        riskError = "";

        try {
            // 构造请求数据，附带 market 和 currency 给后端穿透汇率中心
            const payload = portfolio.map((s) => ({
                symbol: s.symbol,
                shares: s.shares,
                price: s.price,
                market: s.market,
                currency: s.currency,
                asset_type: s.asset_type,
            }));

            const res = await fetch("/api/portfolio/risk", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    portfolio: payload,
                    force_refresh: forceRefresh,
                }),
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "风控分析请求失败");
            }

            riskReport = await res.json();

            // 绘制饼图
            await tick();
            if (riskReport && riskReport.assets_breakdown && chartCanvas) {
                renderPieChart(riskReport.assets_breakdown);
            }
        } catch (e) {
            riskError = e.message;
        } finally {
            isAnalyzing = false;
        }
    }

    function renderPieChart(/** @type {any[]} */ breakdown) {
        if (pieChartInstance) {
            pieChartInstance.destroy();
        }

        const ctx = chartCanvas.getContext("2d");
        const labels = breakdown.map((item) => item.symbol);
        const data = breakdown.map((item) => item.weight * 100);

        // 自动按市值给颜色
        const bgColors = [
            "rgba(244, 63, 94, 0.7)", // rose
            "rgba(59, 130, 246, 0.7)", // blue
            "rgba(16, 185, 129, 0.7)", // emerald
            "rgba(245, 158, 11, 0.7)", // amber
            "rgba(139, 92, 246, 0.7)", // violet
        ];

        pieChartInstance = new Chart(ctx, {
            type: "pie",
            data: {
                labels: labels,
                datasets: [
                    {
                        data: data,
                        backgroundColor: bgColors.slice(0, data.length),
                        borderWidth: 1,
                        borderColor: "rgba(255, 255, 255, 0.1)",
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "right",
                        labels: { color: "rgba(255, 255, 255, 0.7)" },
                    },
                    tooltip: {
                        callbacks: {
                            label: function (tooltipItem) {
                                return ` ${tooltipItem.label}: ${tooltipItem.raw.toFixed(1)}%`;
                            },
                        },
                    },
                },
            },
        });
    }
</script>

<svelte:head>
    <title>投资组合与风控 - AlphaPulse</title>
</svelte:head>

<div class="space-y-8">
    <div class="flex justify-between items-center">
        <div>
            <h2 class="text-2xl font-bold">投资组合</h2>
            <p class="text-white/40 mt-1">管理持仓并进行智能风控诊断</p>
        </div>
        <button
            onclick={() => analyzeRisk(false)}
            disabled={isAnalyzing}
            class="px-5 py-2.5 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-colors flex items-center gap-2"
        >
            {#if isAnalyzing}
                <svg
                    class="w-5 h-5 animate-spin"
                    fill="none"
                    viewBox="0 0 24 24"
                >
                    <circle
                        class="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        stroke-width="4"
                    ></circle>
                    <path
                        class="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                </svg>
                正在风控诊断 (需等待几十秒获取历史数据与AI分析)...
            {:else}
                🛡️ 一键风控与 AI 诊断
            {/if}
        </button>
    </div>

    <!-- 总资产概览 -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
            <div class="text-sm text-white/40 mb-1">总资产 (含可用资金)</div>
            <div class="text-3xl font-bold">
                ¥ {(totalPositionValue + 246913).toLocaleString()}
            </div>
            <div
                class="text-sm {totalPnl >= 0
                    ? 'text-emerald-400'
                    : 'text-rose-400'} mt-1"
            >
                {totalPnl >= 0 ? "↑" : "↓"}
                {totalPnl >= 0 ? "+" : ""}{totalPnlPct.toFixed(2)}% 总盈亏
            </div>
        </Card>
        <Card>
            <div class="text-sm text-white/40 mb-1">
                持仓总市值 (原币种加总假象)
            </div>
            <div class="text-3xl font-bold text-white/50">
                {totalPositionValue.toLocaleString()}
            </div>
            <div class="text-xs text-rose-400 mt-1">⚠️ 未折算汇率</div>
        </Card>
        <Card>
            <div class="text-sm text-white/40 mb-1">可用资金</div>
            <div class="text-3xl font-bold">¥ 246,913</div>
            <div class="text-xs text-white/30 mt-1">
                占比 {((246913 / (totalPositionValue + 246913)) * 100).toFixed(
                    1,
                )}%
            </div>
        </Card>
    </div>

    {#if riskError}
        <div
            class="p-4 rounded-xl bg-orange-500/10 border border-orange-500/20 text-orange-400 text-sm flex items-center gap-3"
        >
            <span>⚠️</span>
            {riskError}
        </div>
    {/if}

    <!-- 风险诊断报告仪盘 -->
    {#if riskReport}
        <Card>
            {#snippet header()}
                <div class="flex justify-between items-center w-full">
                    <div class="flex items-center gap-3">
                        <h3 class="font-bold text-lg">风控诊断报告</h3>
                        {#if riskReport.cached}
                            <span
                                class="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-medium border border-emerald-500/20"
                            >
                                ⚡ 极速缓存
                            </span>
                        {/if}
                    </div>
                    <button
                        onclick={() => analyzeRisk(true)}
                        disabled={isAnalyzing}
                        class="px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-white/70 transition-colors flex items-center gap-2 disabled:opacity-50"
                    >
                        {#if isAnalyzing}
                            <span class="animate-spin">⏳</span> 重新计算中...
                        {:else}
                            🔄 强制重新计算
                        {/if}
                    </button>
                </div>
            {/snippet}

            <!-- 第一排：核心绩效与机构指标 -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-4 mt-4">
                <div
                    class="p-5 bg-indigo-500/10 rounded-xl border border-indigo-500/20 text-center"
                >
                    <div class="text-sm text-indigo-300/70 mb-2">
                        夏普比率 (Sharpe)
                    </div>
                    <div class="text-2xl font-bold text-indigo-400">
                        {riskReport.performance.sharpe_ratio.toFixed(2)}
                    </div>
                </div>
                <div
                    class="p-5 bg-rose-500/10 rounded-xl border border-rose-500/20 text-center"
                >
                    <div class="text-sm text-rose-300/70 mb-2">
                        历史最大回撤
                    </div>
                    <div class="text-2xl font-bold text-rose-500">
                        {(
                            riskReport.performance.max_drawdown_pct * 100
                        ).toFixed(2)}%
                    </div>
                    <div class="text-xs text-rose-400 mt-1 font-medium">
                        极限亏损: ¥{Math.abs(
                            riskReport.performance.max_drawdown_amount,
                        ).toLocaleString([], { maximumFractionDigits: 0 })}
                    </div>
                </div>
                <div
                    class="p-5 bg-emerald-500/5 rounded-xl border border-emerald-500/10 text-center"
                >
                    <div class="text-sm text-emerald-300/70 mb-2">
                        索提诺比率 (Sortino)
                    </div>
                    <div class="text-2xl font-bold text-amber-400">
                        {riskReport.performance.sortino_ratio.toFixed(2)}
                    </div>
                </div>
                <div
                    class="p-5 bg-white/[0.02] rounded-xl border border-white/5 text-center"
                >
                    <div class="text-sm text-white/40 mb-2">组合当前总值</div>
                    <div class="text-2xl font-bold text-emerald-400">
                        ¥ {riskReport.total_value.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                        })}
                    </div>
                </div>
            </div>

            <!-- 分布占比区域 (Pie Chart) -->
            <div class="mb-8 p-5 bg-black/20 rounded-xl border border-white/5">
                <div class="text-sm font-medium text-white/60 mb-4">
                    穿透汇率后的人民币真实法币净值占比分析
                </div>
                <div class="h-48 w-full flex justify-center">
                    <canvas bind:this={chartCanvas}></canvas>
                </div>
            </div>

            <!-- 第二排：尾部风险预警 -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div
                    class="p-5 bg-white/[0.02] rounded-xl border border-white/5 text-center"
                >
                    <div class="text-sm text-white/40 mb-2">年化波动率</div>
                    <div class="text-2xl font-bold text-white/80">
                        {(riskReport.annual_volatility * 100).toFixed(2)}%
                    </div>
                </div>
                <div
                    class="p-5 bg-white/[0.02] rounded-xl border border-white/5 text-center"
                >
                    <div class="text-sm text-white/40 mb-2">
                        99% VaR (日内极限回撤)
                    </div>
                    <div class="text-2xl font-bold text-orange-400">
                        {(riskReport.metrics.historical.var_99 * 100).toFixed(
                            2,
                        )}%
                    </div>
                </div>
                <div
                    class="p-5 bg-white/[0.02] rounded-xl border border-white/5 text-center"
                >
                    <div class="text-sm text-white/40 mb-2">
                        99% CVaR (黑天鹅损失)
                    </div>
                    <div class="text-2xl font-bold text-red-500">
                        {(riskReport.metrics.historical.cvar_99 * 100).toFixed(
                            2,
                        )}%
                    </div>
                </div>
            </div>

            <div class="border-t border-white/10 pt-6">
                <div class="flex items-center gap-2 mb-4">
                    <span class="text-xl">🤖</span>
                    <h3 class="text-lg font-bold">AI 智能分析与建仓建议</h3>
                </div>
                <!-- 渲染 Markdown -->
                <div
                    class="prose prose-invert prose-sm max-w-none text-white/80 bg-black/20 p-6 rounded-xl border border-white/5"
                >
                    <!-- eslint-disable-next-line svelte/no-at-html-tags -->
                    {@html marked(riskReport.ai_report)}
                </div>
            </div>
        </Card>
    {/if}

    <!-- 持仓列表 -->
    <Card title="持仓明细">
        <div class="overflow-x-auto">
            <table class="w-full text-sm">
                <thead>
                    <tr class="border-b border-white/5">
                        <th
                            class="text-left py-3 px-3 text-white/40 font-medium"
                            >股票</th
                        >
                        <th
                            class="text-left py-3 px-3 text-white/40 font-medium"
                            >市场 / 计价币种</th
                        >
                        <th
                            class="text-right py-3 px-3 text-white/40 font-medium"
                            >持仓数量</th
                        >
                        <th
                            class="text-right py-3 px-3 text-white/40 font-medium"
                            >现价</th
                        >
                        <th
                            class="text-right py-3 px-3 text-white/40 font-medium"
                            >成本价</th
                        >
                        <th
                            class="text-right py-3 px-3 text-white/40 font-medium"
                            >盈亏</th
                        >
                        <th
                            class="text-right py-3 px-3 text-white/40 font-medium"
                            >涨跌幅</th
                        >
                    </tr>
                </thead>
                <tbody>
                    {#each portfolio as stock}
                        {@const pnl = (stock.price - stock.cost) * stock.shares}
                        {@const pnlPct =
                            ((stock.price - stock.cost) / stock.cost) * 100}
                        <tr
                            class="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors"
                        >
                            <td class="py-3 px-3">
                                <div class="flex items-center gap-3">
                                    <div
                                        class="w-9 h-9 rounded-lg bg-primary-600/20 flex items-center justify-center text-primary-500 font-bold text-sm"
                                    >
                                        {stock.name[0]}
                                    </div>
                                    <div>
                                        <div class="font-medium">
                                            {stock.name}
                                        </div>
                                        <div class="text-xs text-white/30">
                                            {stock.symbol}
                                        </div>
                                    </div>
                                </div>
                            </td>
                            <td class="py-3 px-3 text-left">
                                <div class="flex flex-col gap-1">
                                    <span
                                        class="w-min px-2 py-0.5 rounded-md bg-white/10 text-xs text-white/70 font-mono"
                                        >{stock.market}</span
                                    >
                                    <span
                                        class="w-min px-2 py-0.5 rounded-md bg-yellow-500/10 text-xs text-yellow-400 font-mono"
                                        >{stock.currency}</span
                                    >
                                </div>
                            </td>
                            <td class="py-3 px-3 text-right">{stock.shares}</td>
                            <td class="py-3 px-3 text-right"
                                >{stock.price.toFixed(2)}</td
                            >
                            <td class="py-3 px-3 text-right text-white/50"
                                >{stock.cost.toFixed(2)}</td
                            >
                            <td
                                class="py-3 px-3 text-right font-semibold {pnl >=
                                0
                                    ? 'text-emerald-400'
                                    : 'text-rose-400'}"
                            >
                                {pnl >= 0 ? "+" : ""}{pnl.toFixed(0)}
                            </td>
                            <td class="py-3 px-3 text-right">
                                <span
                                    class="px-2 py-0.5 rounded-lg text-xs font-medium {pnlPct >=
                                    0
                                        ? 'bg-emerald-500/10 text-emerald-400'
                                        : 'bg-rose-500/10 text-rose-400'}"
                                >
                                    {pnlPct >= 0 ? "+" : ""}{pnlPct.toFixed(2)}%
                                </span>
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
    </Card>
</div>
