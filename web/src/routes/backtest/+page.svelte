<script>
    import Card from "$lib/components/Card.svelte";
    import { onMount } from "svelte";
    import {
        Chart,
        LineController,
        LineElement,
        PointElement,
        LinearScale,
        CategoryScale,
        Filler,
        Tooltip,
        Legend,
    } from "chart.js";

    // 注册 Chart.js 所需组件
    Chart.register(
        LineController,
        LineElement,
        PointElement,
        LinearScale,
        CategoryScale,
        Filler,
        Tooltip,
        Legend,
    );

    let symbol = $state("600519");
    let startDate = $state("20230101");
    let endDate = $state("20240101");
    let strategy = $state("macd");
    let loading = $state(false);
    /** @type {any} */
    let result = $state(null);
    let error = $state("");
    let aiReport = $state("");
    let aiLoading = $state(false);

    /** @type {HTMLCanvasElement|null} */
    let chartCanvas = $state(null);
    /** @type {any} */
    let chartInstance = $state(null);

    const strategies = [
        { value: "macd", label: "MACD 金叉/死叉" },
        { value: "rsi", label: "RSI 超买超卖" },
        { value: "ai", label: "AI 模型预测" },
    ];

    async function runBacktest() {
        loading = true;
        error = "";
        result = null;
        aiReport = "";
        try {
            const res = await fetch(
                `/api/backtest/${symbol}?start_date=${startDate}&end_date=${endDate}&strategy_type=${strategy}`,
            );
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "回测接口响应异常");
            if (data.result?.error) throw new Error(data.result.error);
            result = data;
            // 渲染图表
            setTimeout(() => renderChart(), 100);
        } catch (/** @type {any} */ e) {
            error = e.message;
        } finally {
            loading = false;
        }
    }

    // 渲染净值曲线图
    function renderChart() {
        if (!chartCanvas || !result?.result?.equity_curve) return;
        if (chartInstance) chartInstance.destroy();

        const curve = result.result.equity_curve;
        const labels = curve.map(
            (/** @type {any} */ p) => p.date || p[0] || "",
        );
        const values = curve.map(
            (/** @type {any} */ p) => p.value ?? p.equity ?? p[1] ?? p,
        );

        chartInstance = new Chart(chartCanvas, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "净值",
                        data: values,
                        borderColor: "#3b82f6",
                        backgroundColor: "rgba(59, 130, 246, 0.08)",
                        borderWidth: 2,
                        fill: true,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        tension: 0.3,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: "index" },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "rgba(0,0,0,0.85)",
                        titleColor: "#94a3b8",
                        bodyColor: "#e2e8f0",
                        borderColor: "rgba(255,255,255,0.1)",
                        borderWidth: 1,
                        padding: 10,
                        cornerRadius: 8,
                    },
                },
                scales: {
                    x: {
                        ticks: {
                            color: "rgba(255,255,255,0.3)",
                            maxTicksLimit: 8,
                            font: { size: 10 },
                        },
                        grid: { color: "rgba(255,255,255,0.03)" },
                        border: { color: "rgba(255,255,255,0.05)" },
                    },
                    y: {
                        ticks: {
                            color: "rgba(255,255,255,0.3)",
                            font: { size: 10 },
                        },
                        grid: { color: "rgba(255,255,255,0.03)" },
                        border: { color: "rgba(255,255,255,0.05)" },
                    },
                },
            },
        });
    }

    async function generateAiAnalysis() {
        if (!result) return;
        aiLoading = true;
        try {
            const res = await fetch("/api/backtest/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    symbol: result.symbol,
                    strategy: result.strategy,
                    summary: result.result.summary,
                }),
            });
            if (!res.ok) throw new Error("分析请求失败");
            const data = await res.json();
            let raw = data.report || "未能生成报告";
            // 过滤 <think> 标签及其内容
            raw = raw.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
            aiReport = raw;
        } catch (/** @type {any} */ e) {
            error = "生成分析失败: " + e.message;
        } finally {
            aiLoading = false;
        }
    }

    // 格式化百分比
    function fmt(/** @type {number} */ val) {
        return typeof val === "number" ? val.toFixed(2) : "0.00";
    }

    // AI 报告按段落渲染，识别标题 (###)
    function parseReport(/** @type {string} */ text) {
        const lines = text.split("\n").filter((l) => l.trim());
        /** @type {Array<{type:string, content:string}>} */
        const blocks = [];
        for (const line of lines) {
            if (line.startsWith("### ")) {
                blocks.push({
                    type: "heading",
                    content: line.replace("### ", ""),
                });
            } else if (line.startsWith("## ")) {
                blocks.push({
                    type: "heading",
                    content: line.replace("## ", ""),
                });
            } else if (line.startsWith("**") && line.endsWith("**")) {
                blocks.push({
                    type: "subheading",
                    content: line.replace(/\*\*/g, ""),
                });
            } else if (line.startsWith("- ") || line.startsWith("• ")) {
                blocks.push({
                    type: "bullet",
                    content: line.replace(/^[-•]\s*/, ""),
                });
            } else {
                blocks.push({ type: "text", content: line });
            }
        }
        return blocks;
    }
</script>

<svelte:head>
    <title>策略回测 - AlphaPulse</title>
</svelte:head>

<div class="space-y-8">
    <div>
        <h2 class="text-2xl font-bold">智能策略回测</h2>
        <p class="text-white/40 mt-1">验证您的投资逻辑，让数据说话</p>
    </div>

    <!-- 回测配置面板 -->
    <Card title="回测参数配置">
        <div
            class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 items-end"
        >
            <div>
                <label
                    for="bt-symbol"
                    class="block text-xs text-white/40 mb-1.5 font-medium"
                    >股票代码</label
                >
                <input
                    id="bt-symbol"
                    type="text"
                    bind:value={symbol}
                    class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all"
                />
            </div>
            <div>
                <label
                    for="bt-start"
                    class="block text-xs text-white/40 mb-1.5 font-medium"
                    >开始日期</label
                >
                <input
                    id="bt-start"
                    type="text"
                    bind:value={startDate}
                    placeholder="YYYYMMDD"
                    class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all"
                />
            </div>
            <div>
                <label
                    for="bt-end"
                    class="block text-xs text-white/40 mb-1.5 font-medium"
                    >结束日期</label
                >
                <input
                    id="bt-end"
                    type="text"
                    bind:value={endDate}
                    placeholder="YYYYMMDD"
                    class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all"
                />
            </div>
            <div>
                <label
                    for="bt-strategy"
                    class="block text-xs text-white/40 mb-1.5 font-medium"
                    >策略类型</label
                >
                <select
                    id="bt-strategy"
                    bind:value={strategy}
                    class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all appearance-none"
                >
                    {#each strategies as s}
                        <option value={s.value} class="bg-surface-900"
                            >{s.label}</option
                        >
                    {/each}
                </select>
            </div>
            <button
                onclick={runBacktest}
                disabled={loading}
                class="px-5 py-2.5 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 text-white text-sm font-medium rounded-xl transition-all shadow-lg shadow-primary-600/20 whitespace-nowrap"
            >
                {loading ? "⏳ 回测中..." : "▶ 运行回测"}
            </button>
        </div>
    </Card>

    {#if error}
        <div
            class="bg-rose-500/10 border border-rose-500/20 text-rose-400 px-4 py-3 rounded-xl text-sm"
        >
            ⚠️ {error}
        </div>
    {/if}

    {#if loading}
        <div class="flex items-center justify-center py-20">
            <div class="flex flex-col items-center gap-4">
                <div
                    class="w-10 h-10 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin"
                ></div>
                <span class="text-white/40 text-sm">正在执行回测计算...</span>
            </div>
        </div>
    {/if}

    {#if result}
        {@const summary = result.result.summary}

        <!-- 关键指标 -->
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
                <div class="text-sm text-white/40 mb-1">总收益率</div>
                <div
                    class="text-2xl font-bold {summary.total_return_pct >= 0
                        ? 'text-emerald-400'
                        : 'text-rose-400'}"
                >
                    {fmt(summary.total_return_pct)}%
                </div>
            </Card>
            <Card>
                <div class="text-sm text-white/40 mb-1">年化收益</div>
                <div
                    class="text-2xl font-bold {summary.annual_return_pct >= 0
                        ? 'text-emerald-400'
                        : 'text-rose-400'}"
                >
                    {fmt(summary.annual_return_pct)}%
                </div>
            </Card>
            <Card>
                <div class="text-sm text-white/40 mb-1">最大回撤</div>
                <div class="text-2xl font-bold text-amber-400">
                    {fmt(summary.max_drawdown_pct)}%
                </div>
            </Card>
            <Card>
                <div class="text-sm text-white/40 mb-1">夏普比率</div>
                <div class="text-2xl font-bold text-primary-500">
                    {fmt(summary.sharpe_ratio)}
                </div>
            </Card>
        </div>

        <!-- 净值曲线（真实 Chart.js 图表） -->
        <Card title="📈 净值曲线">
            <div class="h-72 relative">
                <canvas bind:this={chartCanvas}></canvas>
            </div>
        </Card>

        <!-- AI 策略诊断 -->
        <Card title="🤖 AI 策略诊断">
            {#if aiReport}
                <div class="space-y-3">
                    {#each parseReport(aiReport) as block}
                        {#if block.type === "heading"}
                            <h4
                                class="text-base font-semibold text-primary-400 pt-2 border-b border-white/5 pb-1"
                            >
                                {block.content}
                            </h4>
                        {:else if block.type === "subheading"}
                            <h5
                                class="text-sm font-semibold text-white/80 pt-1"
                            >
                                {block.content}
                            </h5>
                        {:else if block.type === "bullet"}
                            <div
                                class="flex gap-2 text-sm text-white/70 leading-relaxed"
                            >
                                <span class="text-primary-500 shrink-0 mt-0.5"
                                    >•</span
                                >
                                <span>{block.content}</span>
                            </div>
                        {:else}
                            <p class="text-sm text-white/70 leading-relaxed">
                                {block.content}
                            </p>
                        {/if}
                    {/each}
                </div>
            {:else if aiLoading}
                <div
                    class="flex items-center justify-center py-8 text-white/40 text-sm"
                >
                    <div
                        class="w-5 h-5 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin mr-3"
                    ></div>
                    正在调用大模型诊断...
                </div>
            {:else}
                <div class="text-center py-6">
                    <p class="text-sm text-white/30 mb-3">
                        基于回测数据生成专业策略分析报告
                    </p>
                    <button
                        onclick={generateAiAnalysis}
                        class="px-6 py-3 bg-gradient-to-r from-primary-600 to-purple-600 hover:from-primary-700 hover:to-purple-700 text-white text-sm font-medium rounded-xl transition-all shadow-lg"
                    >
                        ✨ 生成诊断报告
                    </button>
                </div>
            {/if}
        </Card>

        <!-- 交易流水 -->
        <Card title="📋 交易历史">
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b border-white/5">
                            <th
                                class="text-left py-3 px-3 text-white/40 font-medium"
                                >日期</th
                            >
                            <th
                                class="text-left py-3 px-3 text-white/40 font-medium"
                                >类型</th
                            >
                            <th
                                class="text-right py-3 px-3 text-white/40 font-medium"
                                >价格</th
                            >
                            <th
                                class="text-right py-3 px-3 text-white/40 font-medium"
                                >数量</th
                            >
                            <th
                                class="text-right py-3 px-3 text-white/40 font-medium"
                                >金额</th
                            >
                        </tr>
                    </thead>
                    <tbody>
                        {#each result.result.trades || [] as trade}
                            <tr
                                class="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors"
                            >
                                <td class="py-2.5 px-3 text-white/70"
                                    >{trade.date}</td
                                >
                                <td
                                    class="py-2.5 px-3 font-semibold {trade.type ===
                                    'BUY'
                                        ? 'text-rose-400'
                                        : 'text-emerald-400'}"
                                >
                                    {trade.type === "BUY" ? "买入" : "卖出"}
                                </td>
                                <td class="py-2.5 px-3 text-right"
                                    >{trade.price.toFixed(2)}</td
                                >
                                <td class="py-2.5 px-3 text-right"
                                    >{trade.shares}</td
                                >
                                <td
                                    class="py-2.5 px-3 text-right text-white/70"
                                >
                                    {Number(
                                        trade.cost ?? trade.revenue ?? 0,
                                    ).toFixed(2)}
                                </td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        </Card>
    {/if}
</div>
