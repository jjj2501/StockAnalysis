<script>
    import Card from "$lib/components/Card.svelte";

    let stocks = $state([
        {
            name: "贵州茅台",
            code: "600519",
            price: 1680.5,
            change: 2.3,
            signal: "强烈买入",
        },
        {
            name: "宁德时代",
            code: "300750",
            price: 198.3,
            change: -1.2,
            signal: "持有",
        },
        {
            name: "比亚迪",
            code: "002594",
            price: 267.8,
            change: 3.5,
            signal: "买入",
        },
        {
            name: "中国平安",
            code: "601318",
            price: 48.6,
            change: 0.8,
            signal: "持有",
        },
        {
            name: "招商银行",
            code: "600036",
            price: 34.2,
            change: -0.3,
            signal: "观望",
        },
    ]);

    let newCode = $state("");

    function addStock() {
        if (!newCode.trim()) return;
        stocks = [
            ...stocks,
            {
                name: `股票 ${newCode}`,
                code: newCode,
                price: Number((Math.random() * 200 + 10).toFixed(2)),
                change: Number((Math.random() * 6 - 3).toFixed(2)),
                signal: "分析中",
            },
        ];
        newCode = "";
    }

    function removeStock(/** @type {string} */ code) {
        stocks = stocks.filter((s) => s.code !== code);
    }
</script>

<svelte:head>
    <title>智能预警 - AlphaPulse</title>
</svelte:head>

<div class="space-y-8">
    <div
        class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4"
    >
        <div>
            <h2 class="text-2xl font-bold">智能预警 · 自选股</h2>
            <p class="text-white/40 mt-1">实时追踪关注个股，获取 AI 智能提醒</p>
        </div>
        <div class="flex gap-2">
            <input
                type="text"
                bind:value={newCode}
                placeholder="输入股票代码"
                class="bg-white/5 border border-white/10 px-3 py-2 rounded-xl text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-primary-500/50 w-40 transition-all"
                onkeydown={(e) => e.key === "Enter" && addStock()}
            />
            <button
                onclick={addStock}
                class="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white text-sm font-medium rounded-xl transition-all shadow-lg shadow-primary-600/20"
            >
                + 添加
            </button>
        </div>
    </div>

    <div class="grid grid-cols-1 gap-3">
        {#each stocks as stock (stock.code)}
            <div
                class="glass rounded-2xl p-5 flex items-center justify-between group hover:border-white/10 transition-all"
            >
                <div class="flex items-center gap-4">
                    <div
                        class="w-11 h-11 rounded-xl bg-primary-600/20 flex items-center justify-center text-primary-500 font-bold"
                    >
                        {stock.name[0]}
                    </div>
                    <div>
                        <div class="font-semibold">{stock.name}</div>
                        <div class="text-xs text-white/30">{stock.code}</div>
                    </div>
                </div>

                <div class="flex items-center gap-8">
                    <div class="text-right">
                        <div class="font-bold text-lg">{stock.price}</div>
                        <div
                            class="text-sm {Number(stock.change) >= 0
                                ? 'text-emerald-400'
                                : 'text-rose-400'}"
                        >
                            {Number(stock.change) >= 0
                                ? "+"
                                : ""}{stock.change}%
                        </div>
                    </div>
                    <div class="text-right hidden md:block">
                        <span
                            class="px-3 py-1 rounded-lg text-xs font-medium
                            {stock.signal === '强烈买入'
                                ? 'bg-emerald-500/10 text-emerald-400'
                                : stock.signal === '买入'
                                  ? 'bg-primary-500/10 text-primary-400'
                                  : stock.signal === '观望'
                                    ? 'bg-amber-500/10 text-amber-400'
                                    : 'bg-white/5 text-white/50'}"
                        >
                            {stock.signal}
                        </span>
                    </div>
                    <button
                        onclick={() => removeStock(stock.code)}
                        class="opacity-0 group-hover:opacity-100 text-white/30 hover:text-rose-400 transition-all p-1"
                    >
                        ✕
                    </button>
                </div>
            </div>
        {/each}
    </div>

    {#if stocks.length === 0}
        <Card>
            <div class="text-center py-16">
                <div class="text-5xl mb-4">🔔</div>
                <h3 class="text-lg font-semibold mb-2">暂无自选股</h3>
                <p class="text-white/40 text-sm">
                    点击上方"添加"按钮添加关注的股票
                </p>
            </div>
        </Card>
    {/if}
</div>
