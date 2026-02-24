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
        alternative: {
            label: "另类因子",
            icon: "🕵️",
            desc: "外资机构席位分解与另类分析",
        },
        news: {
            label: "新闻舆情",
            icon: "📰",
            desc: "个股最新新闻与情感分析",
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
        FlowCap: "流通市值",
        LatestPrice: "最新价",
        Industry: "所属行业",
        ROE: "净资产收益率",
        HoldingRatio: "外资持股占比",
        NetBuy: "当日增持净额",
        BankCustodyRatio: "银行配置席位占比",
        BrokerageRatio: "券商交易席位占比",
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
        长线资金主导: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        交易型热钱偏多: { bg: "bg-rose-500/15", text: "text-rose-400" },
        连续买入: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        资金流出: { bg: "bg-rose-500/15", text: "text-rose-400" },
        偏多: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
        偏空: { bg: "bg-rose-500/15", text: "text-rose-400" },
        暂无数据: { bg: "bg-slate-500/15", text: "text-slate-400" },
    };

    let symbol = $state("600519");
    let loading = $state(false);
    /** @type {any} */
    let data = $state(null);
    let error = $state("");

    // AI 预测相关状态
    let aiLoading = $state(false);
    /** @type {any[] | null} */
    let aiPredictions = $state(null);
    let aiError = $state("");

    async function loadAiPredictions() {
        aiLoading = true;
        aiError = "";
        aiPredictions = null;
        try {
            const res = await fetch(`/api/predict/${symbol}`);
            if (!res.ok) throw new Error(`模型未激活或拉取失败`);
            const json = await res.json();
            if (json.error) throw new Error(json.error);

            // 为了架构上支持用户未来新加的模型，封装为可遍历的列表
            // 此处保留当前默认的混合模型结果，并确保界面是以渲染数组的方式进行展示
            aiPredictions = [
                {
                    model_name: "Transformer+LSTM 混合架构",
                    trend: json.predicted_trend,
                    confidence: json.confidence || 0,
                },
            ];
            // 若未来接口支持多模型直接返回 Array，可在此处直接接收
        } catch (/** @type {any} */ e) {
            aiError = e.message;
        } finally {
            aiLoading = false;
        }
    }

    // LLM 全盘因子流式研判状态
    let llmAnalyzing = $state(false);
    let llmReport = $state("");
    let llmError = $state("");

    function startAiAnalysis() {
        if (!data || loading) return;
        llmAnalyzing = true;
        llmReport = "";
        llmError = "";

        const eventSource = new EventSource(`/api/factors/${symbol}/analyze`);

        eventSource.onmessage = (event) => {
            try {
                const parsed = JSON.parse(event.data);
                if (parsed.text !== undefined) {
                    llmReport += parsed.text;
                }
            } catch (e) {
                // 兼容后备：如果不是JSON也尝试直接追加（去除无谓的强制回车）
                llmReport += event.data;
            }
        };

        eventSource.onerror = (err) => {
            console.error("SSE Error:", err);
            // 只要不是空文本其实也算基本完成了，断开就行
            eventSource.close();
            llmAnalyzing = false;
        };

        // 监听结束或其他自定义事件（这里简化为后端主动关闭或出错关闭）
        eventSource.addEventListener("close", () => {
            eventSource.close();
            llmAnalyzing = false;
        });
    }

    async function loadFactors() {
        loading = true;
        error = "";
        data = null;
        llmReport = ""; // 切换股票时清空上一次的研报
        loadAiPredictions(); // 并行触发预测推演

        try {
            const res = await fetch(`/api/factors/${symbol}`);
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

        <!-- AI 模型预测卡片 (支持多模型横向扩展) -->
        <div class="mb-6 mt-6">
            <h3
                class="flex items-center gap-2 font-bold text-lg text-white mb-4"
            >
                <span>🤖 AI 智能前瞻预测</span>
                <span
                    class="text-[10px] font-normal text-white/40 bg-white/5 px-2 py-0.5 rounded-full border border-white/5"
                    >多模型并行支持</span
                >
            </h3>

            {#if aiLoading}
                <div
                    class="flex items-center justify-center py-6 bg-white/[0.02] border border-white/5 rounded-2xl"
                >
                    <div class="flex items-center gap-3">
                        <div
                            class="w-5 h-5 border-[1.5px] border-primary-500/30 border-t-primary-500 rounded-full animate-spin"
                        ></div>
                        <span class="text-white/40 text-sm"
                            >正在深度推演当前标的未来走势...</span
                        >
                    </div>
                </div>
            {:else if aiError}
                <div
                    class="flex items-center gap-4 bg-white/[0.01] border border-white/5 rounded-2xl p-5"
                >
                    <div class="text-3xl opacity-50 grayscale">😴</div>
                    <div>
                        <div class="text-white/60 text-sm mb-0.5 font-medium">
                            {aiError}
                        </div>
                        <div class="text-white/30 text-xs">
                            提示：您可以前往左侧“模型训练”页为该资产独立训练专属大脑
                        </div>
                    </div>
                </div>
            {:else if aiPredictions && aiPredictions.length > 0}
                <div
                    class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4"
                >
                    {#each aiPredictions as pred}
                        <div
                            class="bg-gradient-to-br from-white/[0.04] to-transparent border border-white/10 rounded-2xl p-4 hover:border-primary-500/30 transition-colors relative overflow-hidden group"
                        >
                            <!-- 光晕背景修饰 -->
                            <div
                                class="absolute -right-6 -top-6 w-24 h-24 bg-primary-500/5 rounded-full blur-2xl group-hover:bg-primary-500/10 transition-colors"
                            ></div>

                            <div
                                class="flex items-start justify-between relative z-10"
                            >
                                <div class="space-y-1">
                                    <div
                                        class="text-white/30 text-[10px] uppercase font-bold tracking-wider"
                                    >
                                        预测引擎 / Model
                                    </div>
                                    <h4
                                        class="text-white/90 font-medium text-sm"
                                    >
                                        {pred.model_name}
                                    </h4>
                                </div>
                                <div
                                    class="px-2.5 py-1 rounded-lg text-xs font-bold shadow-sm {pred.trend ===
                                    'UP'
                                        ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                                        : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'}"
                                >
                                    {pred.trend === "UP"
                                        ? "📈 看涨 (UP)"
                                        : "📉 看跌 (DOWN)"}
                                </div>
                            </div>

                            <div class="mt-5 relative z-10">
                                <div class="flex justify-between text-xs mb-2">
                                    <span class="text-white/40"
                                        >预测置信度 (Confidence)</span
                                    >
                                    <span class="text-white font-mono"
                                        >{pred.confidence.toFixed(1)}</span
                                    >
                                </div>
                                <div
                                    class="h-1.5 w-full bg-white/5 rounded-full overflow-hidden"
                                >
                                    <div
                                        class="h-full rounded-full transition-all duration-1000 {pred.trend ===
                                        'UP'
                                            ? 'bg-rose-500'
                                            : 'bg-emerald-500'}"
                                        style="width: {Math.min(
                                            pred.confidence / 10,
                                            100,
                                        )}%"
                                    ></div>
                                </div>
                            </div>
                        </div>
                    {/each}
                </div>
            {/if}
        </div>

        <!-- LLM 综合因子投资诊断大模型版块 -->
        <div
            class="mb-8 p-1 rounded-2xl bg-gradient-to-r from-primary-600/20 via-blue-500/20 to-purple-500/20"
        >
            <div
                class="bg-[#12141c] rounded-xl p-6 h-full border border-white/5"
            >
                <div class="flex items-center justify-between mb-4">
                    <h3
                        class="flex items-center gap-2 font-bold text-lg text-white"
                    >
                        <span class="text-xl">🦾</span>
                        <span
                            class="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent"
                            >全盘因子大模型深度诊断</span
                        >
                    </h3>

                    {#if !llmReport && !llmAnalyzing}
                        <button
                            onclick={startAiAnalysis}
                            class="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-white font-medium transition-all flex items-center gap-2"
                        >
                            <span>⚡</span> 召唤大模型进行综合分析
                        </button>
                    {/if}

                    {#if llmAnalyzing}
                        <div
                            class="flex items-center gap-2 px-3 py-1.5 bg-primary-500/10 border border-primary-500/20 rounded-lg text-primary-400 text-xs font-mono"
                        >
                            <span
                                class="w-2 h-2 rounded-full bg-primary-500 animate-pulse"
                            ></span>
                            AI 思维推演中...
                        </div>
                    {/if}
                </div>

                {#if llmReport || llmAnalyzing}
                    <div
                        class="p-5 rounded-lg bg-[#0d0f16] border border-white/5 relative overflow-hidden"
                    >
                        <!-- 装饰背景线条 -->
                        <div
                            class="absolute inset-0 opacity-[0.03] select-none"
                            style="background-image: repeating-linear-gradient(0deg, transparent, transparent 19px, #fff 19px, #fff 20px);"
                        ></div>

                        <div
                            class="relative z-10 prose prose-invert prose-sm max-w-none text-white/80 leading-relaxed font-sans"
                        >
                            <!-- Svelte 内置支持渲染简单的换行文本或直接解析。这里因 Markdown 解析需引入额外的包，直接使用 pre-wrap 保留大模型自带换行和加粗星号 -->
                            <div
                                class="whitespace-pre-wrap font-mono text-[13px]"
                            >
                                {llmReport}
                            </div>

                            {#if llmAnalyzing}
                                <span
                                    class="inline-block w-1.5 h-4 bg-primary-400 ml-1 animate-pulse align-middle"
                                ></span>
                            {/if}
                        </div>
                    </div>
                {/if}
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

                    {#if catKey === "news" && data[catKey] && data[catKey].items && data[catKey].items.length > 0}
                        <!-- 新闻因子专用渲染：情感概览 + 新闻列表 -->
                        <div class="space-y-4">
                            <!-- 情感概览 -->
                            <div
                                class="flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03]"
                            >
                                <div class="flex items-center gap-2">
                                    <span class="text-sm text-white/60"
                                        >舆情得分</span
                                    >
                                    <span
                                        class="text-lg font-bold text-white/90 tabular-nums"
                                        >{data[catKey].sentiment_score}</span
                                    >
                                </div>
                                <span
                                    class="px-2.5 py-0.5 rounded-full text-[11px] font-medium {getSignalStyle(
                                        data[catKey].sentiment_signal,
                                    ).bg} {getSignalStyle(
                                        data[catKey].sentiment_signal,
                                    ).text}"
                                >
                                    {data[catKey].sentiment_signal}
                                </span>
                            </div>
                            <!-- 新闻列表 -->
                            <div
                                class="space-y-1.5 max-h-[320px] overflow-y-auto pr-1 custom-scrollbar"
                            >
                                {#each data[catKey].items as news, i}
                                    <a
                                        href={news.url || "#"}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        class="group/news block px-3 py-2.5 rounded-lg hover:bg-white/[0.04] transition-colors border border-transparent hover:border-white/5"
                                    >
                                        <div class="flex items-start gap-2">
                                            <span
                                                class="text-[10px] text-white/15 mt-1 min-w-[16px]"
                                                >{i + 1}</span
                                            >
                                            <div class="flex-1 min-w-0">
                                                <p
                                                    class="text-sm text-white/75 group-hover/news:text-white/90 transition-colors leading-relaxed line-clamp-2"
                                                >
                                                    {news.title}
                                                </p>
                                                <div
                                                    class="flex items-center gap-3 mt-1.5"
                                                >
                                                    {#if news.source}
                                                        <span
                                                            class="text-[10px] text-white/25"
                                                            >{news.source}</span
                                                        >
                                                    {/if}
                                                    {#if news.time}
                                                        <span
                                                            class="text-[10px] text-white/20"
                                                            >{news.time}</span
                                                        >
                                                    {/if}
                                                </div>
                                            </div>
                                            <span
                                                class="text-[10px] text-white/10 group-hover/news:text-white/30 mt-1"
                                                >↗</span
                                            >
                                        </div>
                                    </a>
                                {/each}
                            </div>
                            <div
                                class="text-center text-[10px] text-white/15 pt-1"
                            >
                                共 {data[catKey].count} 条相关新闻
                            </div>
                        </div>
                    {:else if catKey === "alternative" && data[catKey] && typeof data[catKey] === "object" && Object.keys(data[catKey]).length > 0}
                        <div class="space-y-4">
                            <!-- 北向大盘横向拉通 -->
                            <div class="flex gap-4">
                                {#each ["HoldingRatio", "NetBuy"] as key}
                                    {#if data[catKey][key]}
                                        <div
                                            class="flex-1 bg-white/[0.02] rounded-xl p-3 border border-white/5"
                                        >
                                            <div
                                                class="text-xs text-white/50 mb-1"
                                            >
                                                {getName(key)}
                                            </div>
                                            <div
                                                class="flex items-end justify-between"
                                            >
                                                <div
                                                    class="font-mono text-lg font-bold"
                                                >
                                                    {fmtValue(
                                                        data[catKey][key].value,
                                                        data[catKey][key].unit,
                                                    )}
                                                </div>
                                                {#if data[catKey][key].signal}
                                                    <span
                                                        class="px-2 py-0.5 rounded-md text-[10px] font-medium {getSignalStyle(
                                                            data[catKey][key]
                                                                .signal,
                                                        ).bg} {getSignalStyle(
                                                            data[catKey][key]
                                                                .signal,
                                                        ).text}"
                                                    >
                                                        {data[catKey][key]
                                                            .signal}
                                                    </span>
                                                {/if}
                                            </div>
                                        </div>
                                    {/if}
                                {/each}
                            </div>
                            <!-- 资金席位对决双向条 -->
                            {#if data[catKey]["BankCustodyRatio"] && data[catKey]["BrokerageRatio"]}
                                <div
                                    class="px-4 py-4 bg-gradient-to-r from-blue-500/5 to-purple-500/5 border border-white/5 rounded-xl"
                                >
                                    <div
                                        class="flex justify-between text-xs font-medium mb-3"
                                    >
                                        <div class="flex items-center gap-2">
                                            <span
                                                class="w-1.5 h-1.5 rounded-full bg-blue-500"
                                            ></span>
                                            <span class="text-blue-300"
                                                >长线银行托管席位</span
                                            >
                                        </div>
                                        <div class="flex items-center gap-2">
                                            <span class="text-purple-300"
                                                >活跃券商交易席位</span
                                            >
                                            <span
                                                class="w-1.5 h-1.5 rounded-full bg-purple-500"
                                            ></span>
                                        </div>
                                    </div>
                                    <div
                                        class="flex h-2.5 w-full rounded-full overflow-hidden shadow-inner border border-white/5"
                                    >
                                        <div
                                            class="bg-blue-500 transition-all duration-1000"
                                            style="width: {data[catKey][
                                                'BankCustodyRatio'
                                            ].value}%"
                                        ></div>
                                        <div
                                            class="bg-purple-500 transition-all duration-1000"
                                            style="width: {data[catKey][
                                                'BrokerageRatio'
                                            ].value}%"
                                        ></div>
                                    </div>
                                    <div
                                        class="flex justify-between text-lg font-mono font-bold mt-2"
                                    >
                                        <span class="text-blue-400"
                                            >{data[catKey]["BankCustodyRatio"]
                                                .value}%</span
                                        >
                                        <span class="text-purple-400"
                                            >{data[catKey]["BrokerageRatio"]
                                                .value}%</span
                                        >
                                    </div>
                                    {#if data[catKey]["BrokerageRatio"].signal || data[catKey]["BankCustodyRatio"].signal}
                                        <div
                                            class="text-center mt-2 text-[10px] text-white/40"
                                        >
                                            特征判定: <span
                                                class="text-white/70"
                                                >{data[catKey][
                                                    "BankCustodyRatio"
                                                ].signal ||
                                                    data[catKey][
                                                        "BrokerageRatio"
                                                    ].signal}</span
                                            >
                                        </div>
                                    {/if}
                                </div>
                            {/if}
                        </div>
                    {:else if catKey !== "news" && catKey !== "alternative" && data[catKey] && typeof data[catKey] === "object" && Object.keys(data[catKey]).length > 0}
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
