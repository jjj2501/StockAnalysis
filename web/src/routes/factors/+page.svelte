<script>
    import Card from "$lib/components/Card.svelte";

    // 因子分类元数据
    const categoryMeta = {
        technical: {
            label: "技术指标",
            icon: "📈",
            desc: "基于价格和成交量的技术分析",
        },
        fundamental: {
            label: "基本面",
            icon: "📋",
            desc: "估值、盈利与公司规模",
        },
        sentiment: {
            label: "市场情绪",
            icon: "🧠",
            desc: "市场参与度与活跃程度",
        },
        northbound: {
            label: "北上资金",
            icon: "💰",
            desc: "沪深港通外资持仓动态",
        },
    };

    // 因子 key → 中文名映射
    /** @type {Record<string, string>} */
    const factorNames = {
        RSI: "RSI 相对强弱",
        WR: "威廉指标 (WR)",
        ROC: "变动速率 (ROC)",
        BB: "布林带位置",
        Turnover: "换手率",
        VolumeRatio: "量比",
        PE: "市盈率 (PE)",
        PB: "市净率 (PB)",
        MarketCap: "总市值",
        ROE: "净资产收益率",
        HoldingRatio: "持股比例",
        NetBuy: "当日增持",
    };

    // 信号颜色映射
    /** @type {Record<string, {bg:string, text:string}>} */
    const signalColors = {
        超买: { bg: "bg-rose-500/15", text: "text-rose-400" },
        超卖: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        看涨: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        看跌: { bg: "bg-rose-500/15", text: "text-rose-400" },
        动能强: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        动能弱: { bg: "bg-amber-500/15", text: "text-amber-400" },
        上轨压力: { bg: "bg-rose-500/15", text: "text-rose-400" },
        下轨支撑: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        轨道内: { bg: "bg-blue-500/15", text: "text-blue-400" },
        中性: { bg: "bg-slate-500/15", text: "text-slate-400" },
        活跃: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        低迷: { bg: "bg-rose-500/15", text: "text-rose-400" },
        放量: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        缩量: { bg: "bg-amber-500/15", text: "text-amber-400" },
        持平: { bg: "bg-slate-500/15", text: "text-slate-400" },
        低估: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        高估: { bg: "bg-rose-500/15", text: "text-rose-400" },
        合理: { bg: "bg-blue-500/15", text: "text-blue-400" },
        破净: { bg: "bg-amber-500/15", text: "text-amber-400" },
        大盘股: { bg: "bg-blue-500/15", text: "text-blue-400" },
        小盘股: { bg: "bg-purple-500/15", text: "text-purple-400" },
        一般: { bg: "bg-slate-500/15", text: "text-slate-400" },
        高仓位: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        低仓位: { bg: "bg-amber-500/15", text: "text-amber-400" },
        连续买入: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        资金流出: { bg: "bg-rose-500/15", text: "text-rose-400" },
    };

    let symbol = $state("600519");
    let loading = $state(false);
    /** @type {any} */
    let data = $state(null);
    let error = $state("");

    async function loadFactors() {
        loading = true;
        error = "";
        data = null;
        try {
            const res = await fetch(
                `http://localhost:8000/api/factors/${symbol}`,
            );
            if (!res.ok) throw new Error(`服务端错误: ${res.status}`);
            data = await res.json();
            if (data.error) throw new Error(data.error);
        } catch (/** @type {any} */ e) {
            error = e.message;
        } finally {
            loading = false;
        }
    }

    // 获取信号的颜色样式
    function getSignalStyle(/** @type {string} */ signal) {
        return (
            signalColors[signal] || {
                bg: "bg-slate-500/15",
                text: "text-slate-400",
            }
        );
    }

    // 格式化数值显示
    function fmtValue(
        /** @type {number} */ val,
        /** @type {string|undefined} */ unit,
    ) {
        const formatted =
            typeof val === "number"
                ? Math.abs(val) >= 10000
                    ? (val / 10000).toFixed(2) + "万"
                    : val.toFixed(2)
                : String(val);
        return unit ? `${formatted}${unit}` : formatted;
    }

    // 获取因子中文名
    function getName(/** @type {string} */ key) {
        return factorNames[key] || key;
    }

    // 计算进度条百分比
    function getBarPercent(
        /** @type {string} */ key,
        /** @type {number} */ val,
    ) {
        if (key === "RSI") return Math.min(val, 100);
        if (key === "WR") return Math.min(100 + val, 100);
        if (key === "Turnover") return Math.min(val * 15, 100);
        if (key === "VolumeRatio") return Math.min(val * 40, 100);
        if (key === "PE") return Math.min((val / 80) * 100, 100);
        if (key === "PB") return Math.min(val * 10, 100);
        if (key === "ROE") return Math.min(val * 3, 100);
        if (key === "HoldingRatio") return Math.min(val * 5, 100);
        if (key === "ROC") return Math.min(50 + val * 3, 100);
        return 50;
    }

    // 进度条颜色
    function getBarColor(/** @type {string} */ signal) {
        const s = signalColors[signal];
        if (!s) return "bg-slate-500/50";
        if (s.text.includes("emerald")) return "bg-emerald-500/70";
        if (s.text.includes("rose")) return "bg-rose-500/70";
        if (s.text.includes("amber")) return "bg-amber-500/70";
        if (s.text.includes("blue")) return "bg-blue-500/70";
        if (s.text.includes("purple")) return "bg-purple-500/70";
        return "bg-slate-500/50";
    }
</script>

<svelte:head>
    <title>量化因子分析 - AlphaPulse</title>
</svelte:head>

<div class="space-y-8">
    <!-- 标题区 -->
    <div
        class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4"
    >
        <div>
            <h2 class="text-2xl font-bold">量化因子分析</h2>
            <p class="text-white/40 mt-1">多维度评估股票投资价值</p>
        </div>
        <div class="flex gap-3">
            <input
                type="text"
                bind:value={symbol}
                placeholder="输入股票代码"
                class="bg-white/5 border border-white/10 px-4 py-2.5 rounded-xl text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/20 w-48 transition-all"
                onkeydown={(/** @type {KeyboardEvent} */ e) =>
                    e.key === "Enter" && loadFactors()}
            />
            <button
                onclick={loadFactors}
                disabled={loading}
                class="px-5 py-2.5 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 text-white text-sm font-medium rounded-xl transition-all shadow-lg shadow-primary-600/20"
            >
                {loading ? "⏳ 分析中..." : "🔍 开始分析"}
            </button>
        </div>
    </div>

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
                    class="w-12 h-12 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin"
                ></div>
                <span class="text-white/40 text-sm"
                    >正在获取量化因子数据，请稍候...</span
                >
                <span class="text-white/20 text-xs"
                    >首次加载可能需要 10~30 秒</span
                >
            </div>
        </div>
    {/if}

    {#if data && !loading}
        <!-- 股票信息栏 -->
        <div
            class="flex flex-wrap items-center gap-4 px-4 py-3 rounded-xl bg-white/[0.03] border border-white/5"
        >
            <div class="flex items-center gap-2">
                <div
                    class="w-8 h-8 rounded-lg bg-primary-600/20 flex items-center justify-center text-primary-500 font-bold text-sm"
                >
                    A
                </div>
                <div>
                    <span class="text-sm font-semibold text-white"
                        >{data.symbol}</span
                    >
                    <span class="text-xs text-white/30 ml-2"
                        >📅 数据日期: {data.date}</span
                    >
                </div>
            </div>
        </div>

        <!-- 因子分类卡片 -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            {#each Object.entries(categoryMeta) as [catKey, meta]}
                <Card>
                    <!-- 卡片标题 -->
                    <div
                        class="flex items-center justify-between mb-5 pb-3 border-b border-white/5"
                    >
                        <div class="flex items-center gap-2.5">
                            <span class="text-xl">{meta.icon}</span>
                            <div>
                                <h3 class="font-semibold text-white text-base">
                                    {meta.label}
                                </h3>
                                <span class="text-[10px] text-white/25"
                                    >{meta.desc}</span
                                >
                            </div>
                        </div>
                    </div>

                    {#if data[catKey] && typeof data[catKey] === "object" && Object.keys(data[catKey]).length > 0}
                        <div class="space-y-5">
                            {#each Object.entries(data[catKey]) as [key, factor]}
                                <div
                                    class="group hover:bg-white/[0.02] -mx-2 px-2 py-1 rounded-lg transition-colors"
                                >
                                    <!-- 因子名 + 信号标签 -->
                                    <div
                                        class="flex items-center justify-between mb-2"
                                    >
                                        <span
                                            class="text-sm text-white/70 font-medium"
                                            >{getName(key)}</span
                                        >
                                        <span
                                            class="px-2.5 py-0.5 rounded-full text-[11px] font-medium {getSignalStyle(
                                                /** @type {{signal:string}} */ (
                                                    factor
                                                ).signal,
                                            ).bg} {getSignalStyle(
                                                /** @type {{signal:string}} */ (
                                                    factor
                                                ).signal,
                                            ).text}"
                                        >
                                            {/** @type {{signal:string}} */ (
                                                factor
                                            ).signal}
                                        </span>
                                    </div>
                                    <!-- 进度条 + 数值 -->
                                    <div class="flex items-center gap-3">
                                        <div
                                            class="flex-1 h-1.5 bg-white/[0.04] rounded-full overflow-hidden"
                                        >
                                            <div
                                                class="h-full rounded-full transition-all duration-700 ease-out {getBarColor(
                                                    /** @type {{signal:string}} */ (
                                                        factor
                                                    ).signal,
                                                )}"
                                                style="width: {Math.max(
                                                    getBarPercent(
                                                        key,
                                                        /** @type {{value:number}} */ (
                                                            factor
                                                        ).value,
                                                    ),
                                                    3,
                                                )}%"
                                            ></div>
                                        </div>
                                        <span
                                            class="text-sm font-bold text-white/90 min-w-[80px] text-right tabular-nums tracking-tight"
                                        >
                                            {fmtValue(
                                                /** @type {{value:number}} */ (
                                                    factor
                                                ).value,
                                                /** @type {{unit?:string}} */ (
                                                    factor
                                                ).unit,
                                            )}
                                        </span>
                                    </div>
                                </div>
                            {/each}
                        </div>
                    {:else}
                        <!-- 无数据的友好提示 -->
                        <div
                            class="flex flex-col items-center justify-center py-10 text-center"
                        >
                            <div class="text-4xl mb-3 opacity-20">
                                {meta.icon}
                            </div>
                            <p class="text-sm text-white/25 mb-1">
                                {meta.label}数据暂未获取
                            </p>
                            <p class="text-xs text-white/15">
                                可能因数据源延迟或非交易时段
                            </p>
                        </div>
                    {/if}
                </Card>
            {/each}
        </div>
    {:else if !loading && !error}
        <!-- 空状态引导 -->
        <Card>
            <div class="text-center py-16">
                <div class="text-5xl mb-4 opacity-50">🔍</div>
                <h3 class="text-lg font-semibold mb-2">输入股票代码开始分析</h3>
                <p class="text-white/40 text-sm mb-6">
                    支持沪深 A 股，输入代码后点击"开始分析"
                </p>
                <div class="flex flex-wrap justify-center gap-2">
                    {#each [{ code: "600519", name: "贵州茅台" }, { code: "000858", name: "五粮液" }, { code: "300750", name: "宁德时代" }, { code: "002594", name: "比亚迪" }, { code: "600036", name: "招商银行" }] as stock}
                        <button
                            onclick={() => {
                                symbol = stock.code;
                                loadFactors();
                            }}
                            class="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 text-xs text-white/50 hover:bg-primary-600/10 hover:text-primary-500 hover:border-primary-500/20 transition-all"
                        >
                            {stock.name} ({stock.code})
                        </button>
                    {/each}
                </div>
            </div>
        </Card>
    {/if}

    <div class="text-center text-xs text-white/20 py-2">
        数据来源: AKShare · 更新频率: 每日收盘后
    </div>
</div>
