<script>
    import { onMount } from "svelte";
    import Card from "$lib/components/Card.svelte";
    import { marked } from "marked"; // 用于渲染 AI Markdown 报告

    // 投资组合响应式状态 (目前前端写死初始组合，后续可做成增删改查)
    let portfolio = $state([
        {
            name: "贵州茅台",
            symbol: "600519",
            shares: 100,
            price: 1680.5,
            cost: 1580.0,
        },
        {
            name: "宁德时代",
            symbol: "300750",
            shares: 200,
            price: 198.3,
            cost: 210.0,
        },
        {
            name: "中国平安",
            symbol: "601318",
            shares: 500,
            price: 48.6,
            cost: 45.2,
        },
        {
            name: "五粮液",
            symbol: "000858",
            shares: 300,
            price: 156.3,
            cost: 162.5,
        },
    ]);

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
            // 构造请求数据 (只传 symbol, shares, price)
            const payload = portfolio.map((s) => ({
                symbol: s.symbol,
                shares: s.shares,
                price: s.price,
            }));

            const res = await fetch(
                "http://localhost:8000/api/portfolio/risk",
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        portfolio: payload,
                        force_refresh: forceRefresh,
                    }),
                },
            );

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "风控分析请求失败");
            }

            riskReport = await res.json();
        } catch (e) {
            riskError = e.message;
        } finally {
            isAnalyzing = false;
        }
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
            <div class="text-sm text-white/40 mb-1">持仓总市值</div>
            <div class="text-3xl font-bold">
                ¥ {totalPositionValue.toLocaleString()}
            </div>
            <div class="text-xs text-white/30 mt-1">
                占比 {(
                    (totalPositionValue / (totalPositionValue + 246913)) *
                    100
                ).toFixed(1)}%
            </div>
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
            <div slot="header" class="flex justify-between items-center w-full">
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
                        ¥ {riskReport.total_value.toLocaleString()}
                    </div>
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
