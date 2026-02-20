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
        BarController,
        BarElement,
    } from "chart.js";

    Chart.register(
        LineController,
        LineElement,
        PointElement,
        LinearScale,
        CategoryScale,
        Filler,
        Tooltip,
        Legend,
        BarController,
        BarElement,
    );

    const stats = [
        { label: "上证指数", value: "3,245.12", change: "+1.2%", up: true },
        { label: "深证成指", value: "11,023.45", change: "-0.5%", up: false },
        { label: "创业板指", value: "2,312.88", change: "+0.8%", up: true },
        { label: "科创50", value: "985.67", change: "+2.1%", up: true },
    ];

    /** @type {HTMLCanvasElement|null} */
    let heatCanvas = $state(null);

    onMount(() => {
        if (!heatCanvas) return;
        // 生成近30天模拟行情数据
        const labels = Array.from({ length: 30 }, (_, i) => `${i + 1}日`);
        const shData = labels.map(() => 3100 + Math.random() * 200);
        const szData = labels.map(() => 10800 + Math.random() * 400);

        new Chart(heatCanvas, {
            type: "line",
            data: {
                labels,
                datasets: [
                    {
                        label: "上证指数",
                        data: shData,
                        borderColor: "#3b82f6",
                        backgroundColor: "rgba(59, 130, 246, 0.06)",
                        borderWidth: 2,
                        fill: true,
                        pointRadius: 0,
                        tension: 0.4,
                        yAxisID: "y",
                    },
                    {
                        label: "深证成指",
                        data: szData,
                        borderColor: "#a855f7",
                        backgroundColor: "rgba(168, 85, 247, 0.04)",
                        borderWidth: 2,
                        fill: true,
                        pointRadius: 0,
                        tension: 0.4,
                        yAxisID: "y1",
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: "index" },
                plugins: {
                    legend: {
                        display: true,
                        position: "top",
                        labels: {
                            color: "rgba(255,255,255,0.5)",
                            font: { size: 11 },
                            boxWidth: 12,
                            padding: 16,
                        },
                    },
                    tooltip: {
                        backgroundColor: "rgba(0,0,0,0.85)",
                        cornerRadius: 8,
                        padding: 10,
                        titleColor: "#94a3b8",
                        bodyColor: "#e2e8f0",
                        borderColor: "rgba(255,255,255,0.1)",
                        borderWidth: 1,
                    },
                },
                scales: {
                    x: {
                        ticks: {
                            color: "rgba(255,255,255,0.25)",
                            font: { size: 10 },
                            maxTicksLimit: 10,
                        },
                        grid: { color: "rgba(255,255,255,0.03)" },
                        border: { color: "rgba(255,255,255,0.05)" },
                    },
                    y: {
                        type: "linear",
                        position: "left",
                        ticks: {
                            color: "rgba(59,130,246,0.5)",
                            font: { size: 10 },
                        },
                        grid: { color: "rgba(255,255,255,0.03)" },
                        border: { color: "rgba(255,255,255,0.05)" },
                    },
                    y1: {
                        type: "linear",
                        position: "right",
                        ticks: {
                            color: "rgba(168,85,247,0.5)",
                            font: { size: 10 },
                        },
                        grid: { display: false },
                        border: { color: "rgba(255,255,255,0.05)" },
                    },
                },
            },
        });
    });
</script>

<svelte:head>
    <title>AlphaPulse - 智能投资分析平台</title>
</svelte:head>

<div class="space-y-8">
    <!-- 市场指数 -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {#each stats as stat}
            <Card>
                <div class="text-sm text-white/50 mb-1">{stat.label}</div>
                <div class="text-2xl font-bold mb-2">{stat.value}</div>
                <div
                    class="flex items-center gap-1 text-sm {stat.up
                        ? 'text-emerald-400'
                        : 'text-rose-400'}"
                >
                    <span>{stat.up ? "↑" : "↓"}</span>
                    <span>{stat.change}</span>
                </div>
            </Card>
        {/each}
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- 行情走势图 -->
        <Card title="📈 近期行情走势" class="lg:col-span-2 min-h-[400px]">
            <div class="h-80 relative">
                <canvas bind:this={heatCanvas}></canvas>
            </div>
        </Card>

        <!-- 智能选股推荐 -->
        <Card title="🎯 智能选股推荐">
            <div class="space-y-4">
                {#each [{ name: "贵州茅台", code: "600519", ret: "+3.2%", conf: "92%" }, { name: "宁德时代", code: "300750", ret: "+4.2%", conf: "91%" }, { name: "比亚迪", code: "002594", ret: "+5.2%", conf: "90%" }, { name: "五粮液", code: "000858", ret: "+2.8%", conf: "89%" }, { name: "招商银行", code: "600036", ret: "+1.5%", conf: "88%" }] as stock}
                    <div
                        class="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors cursor-pointer group"
                    >
                        <div class="flex items-center gap-3">
                            <div
                                class="w-10 h-10 rounded-lg bg-primary-600/20 flex items-center justify-center font-bold text-primary-500"
                            >
                                {stock.name[0]}
                            </div>
                            <div>
                                <div class="font-medium">{stock.name}</div>
                                <div class="text-xs text-white/40">
                                    {stock.code}
                                </div>
                            </div>
                        </div>
                        <div class="text-right">
                            <div class="text-sm font-semibold text-emerald-400">
                                {stock.ret}
                            </div>
                            <div class="text-[10px] text-white/30">
                                置信度 {stock.conf}
                            </div>
                        </div>
                    </div>
                {/each}
            </div>

            <button
                class="w-full mt-6 py-3 rounded-xl bg-primary-600/10 text-primary-500 text-sm font-medium hover:bg-primary-600 hover:text-white transition-all"
            >
                查看完整列表
            </button>
        </Card>
    </div>
</div>
