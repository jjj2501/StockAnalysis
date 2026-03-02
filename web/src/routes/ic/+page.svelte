<script>
    import { onMount } from "svelte";

    // ── 状态 ──
    let symbol = $state("600519");
    let loading = $state(false);
    let error = $state("");
    let result = $state(null);
    let lookback = $state(500);
    let activeTab = $state("summary"); // summary | decay | corr

    // ── 因子颜色映射 ──
    const FACTOR_COLORS = [
        "#3b82f6",
        "#8b5cf6",
        "#06b6d4",
        "#10b981",
        "#f59e0b",
        "#ef4444",
        "#ec4899",
        "#84cc16",
        "#f97316",
    ];

    async function runAnalysis() {
        if (!symbol) return;
        loading = true;
        error = "";
        result = null;
        try {
            const res = await fetch(
                `/api/ic/${symbol}?lookback=${lookback}&window=60`,
            );
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "分析失败");
            result = data;
            // 稍等一帧再画图
            setTimeout(() => {
                if (activeTab === "decay") drawDecayChart();
                if (activeTab === "corr") drawCorrHeatmap();
            }, 50);
        } catch (e) {
            error = e.message;
        } finally {
            loading = false;
        }
    }

    // ── IC 排行榜：CSS 宽度驱动横向柱状图 ──
    function barWidth(icir) {
        const maxICIR = 2;
        return Math.min((Math.abs(icir) / maxICIR) * 100, 100).toFixed(1);
    }

    function icirColor(icir) {
        if (icir >= 0.5) return "bg-emerald-500";
        if (icir >= 0.2) return "bg-amber-400";
        if (icir <= -0.5) return "bg-rose-600";
        if (icir <= -0.2) return "bg-orange-400";
        return "bg-slate-500";
    }

    function icColor(ic) {
        if (ic > 0.04) return "text-emerald-400";
        if (ic > 0.02) return "text-amber-400";
        if (ic < -0.04) return "text-rose-400";
        if (ic < -0.02) return "text-orange-400";
        return "text-white/50";
    }

    // ── IC 衰减折线图（Canvas） ──
    function drawDecayChart() {
        if (!result?.ic_decay) return;
        const canvas = document.getElementById("decayCanvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const { forward_days, factors } = result.ic_decay;
        const W = canvas.width,
            H = canvas.height;
        const padL = 55,
            padR = 20,
            padT = 20,
            padB = 40;
        const plotW = W - padL - padR;
        const plotH = H - padT - padB;

        ctx.clearRect(0, 0, W, H);

        // 背景
        ctx.fillStyle = "#0d0f14";
        ctx.fillRect(0, 0, W, H);

        // 数据范围
        const allVals = Object.values(factors).flat();
        const maxV = Math.max(
            Math.abs(Math.max(...allVals)),
            Math.abs(Math.min(...allVals)),
            0.05,
        );

        // 零线
        const yZero = padT + plotH * (1 - (0 + maxV) / (2 * maxV));
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(padL, yZero);
        ctx.lineTo(padL + plotW, yZero);
        ctx.stroke();
        ctx.setLineDash([]);

        // Y 轴刻度
        ctx.fillStyle = "rgba(255,255,255,0.35)";
        ctx.font = "11px monospace";
        ctx.textAlign = "right";
        [-maxV, -maxV / 2, 0, maxV / 2, maxV].forEach((v) => {
            const y = padT + plotH * (1 - (v + maxV) / (2 * maxV));
            ctx.fillText(v.toFixed(2), padL - 6, y + 4);
            ctx.strokeStyle = "rgba(255,255,255,0.05)";
            ctx.beginPath();
            ctx.moveTo(padL, y);
            ctx.lineTo(padL + plotW, y);
            ctx.stroke();
        });

        // X 轴标签
        ctx.fillStyle = "rgba(255,255,255,0.35)";
        ctx.textAlign = "center";
        ctx.font = "11px monospace";
        forward_days.forEach((d, i) => {
            const x = padL + (i / (forward_days.length - 1)) * plotW;
            ctx.fillText(`${d}日`, x, H - padB + 16);
        });

        // 折线
        const factorNames = Object.keys(factors);
        factorNames.forEach((name, fi) => {
            const vals = factors[name];
            const color = FACTOR_COLORS[fi % FACTOR_COLORS.length];
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            vals.forEach((v, i) => {
                const x = padL + (i / (forward_days.length - 1)) * plotW;
                const y = padT + plotH * (1 - (v + maxV) / (2 * maxV));
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.stroke();

            // 数据点圆圈
            vals.forEach((v, i) => {
                const x = padL + (i / (forward_days.length - 1)) * plotW;
                const y = padT + plotH * (1 - (v + maxV) / (2 * maxV));
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();
            });
        });
    }

    // ── 因子相关热力图（Canvas） ──
    function drawCorrHeatmap() {
        if (!result?.factor_corr) return;
        const canvas = document.getElementById("corrCanvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const { labels, matrix } = result.factor_corr;
        const n = labels.length;
        if (n === 0) return;

        const W = canvas.width,
            H = canvas.height;
        const pad = 90;
        const cellSize = Math.floor(Math.min((W - pad) / n, (H - pad) / n));

        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "#0d0f14";
        ctx.fillRect(0, 0, W, H);

        // 绘制单元格
        matrix.forEach((row, ri) => {
            row.forEach((val, ci) => {
                const x = pad + ci * cellSize;
                const y = pad + ri * cellSize;
                // val 范围 [-1, 1]，映射颜色
                let r, g, b;
                if (val > 0) {
                    r = Math.round(val * 239);
                    g = Math.round(val * 68);
                    b = Math.round(val * 68);
                } else {
                    const av = Math.abs(val);
                    r = Math.round(av * 59);
                    g = Math.round(av * 130);
                    b = Math.round(av * 246);
                }
                ctx.fillStyle = `rgba(${r},${g},${b},${0.15 + Math.abs(val) * 0.75})`;
                ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

                // 数值文字
                ctx.fillStyle = "rgba(255,255,255,0.85)";
                ctx.font = `${Math.max(9, cellSize * 0.25)}px monospace`;
                ctx.textAlign = "center";
                ctx.fillText(
                    val.toFixed(2),
                    x + cellSize / 2,
                    y + cellSize / 2 + 4,
                );
            });
        });

        // 行/列标签
        ctx.fillStyle = "rgba(255,255,255,0.55)";
        ctx.font = "11px sans-serif";
        labels.forEach((label, i) => {
            const pos = pad + i * cellSize + cellSize / 2;
            // 顶部（列）
            ctx.save();
            ctx.translate(pos, pad - 6);
            ctx.rotate(-Math.PI / 4);
            ctx.textAlign = "left";
            ctx.fillText(label, 0, 0);
            ctx.restore();
            // 左侧（行）
            ctx.textAlign = "right";
            ctx.fillText(label, pad - 6, pad + i * cellSize + cellSize / 2 + 4);
        });
    }

    // Tab 切换后重绘
    $effect(() => {
        if (!result) return;
        if (activeTab === "decay") setTimeout(drawDecayChart, 30);
        if (activeTab === "corr") setTimeout(drawCorrHeatmap, 30);
    });

    onMount(() => {});
</script>

<svelte:head>
    <title>因子实验室 (IC 分析) - AlphaPulse</title>
</svelte:head>

<div class="p-6 md:p-8 max-w-6xl mx-auto space-y-6">
    <!-- 标题 -->
    <div
        class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6"
    >
        <div>
            <h1
                class="text-3xl font-bold tracking-tight text-white flex items-center gap-3"
            >
                <span class="text-4xl">🧪</span> 因子实验室 — IC 分析
            </h1>
            <p class="text-white/50 mt-2 text-sm leading-relaxed max-w-2xl">
                信息系数（IC）衡量每个技术因子对未来价格方向的真实预测能力。ICIR
                ≥ 0.5 且 |IC| ≥ 0.03 的因子值得重点关注。
            </p>
        </div>

        <!-- 控制栏 -->
        <div
            class="flex items-center gap-3 bg-surface-800/80 p-1.5 rounded-xl border border-white/5 backdrop-blur-xl shadow-xl"
        >
            <div class="relative">
                <span
                    class="absolute left-3 top-1/2 -translate-y-1/2 text-white/40"
                    >🎯</span
                >
                <input
                    type="text"
                    bind:value={symbol}
                    placeholder="股票代码 (如 600519)"
                    class="w-44 bg-[#0a0c10] border border-white/10 rounded-lg py-2.5 pl-9 pr-4 text-sm text-white placeholder:text-white/20 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/50 transition-all font-mono uppercase font-bold"
                    disabled={loading}
                    onkeydown={(e) => {
                        if (e.key === "Enter") runAnalysis();
                    }}
                />
            </div>
            <select
                bind:value={lookback}
                class="bg-[#0a0c10] border border-white/10 rounded-lg py-2.5 px-3 text-sm text-white/70 focus:outline-none focus:border-primary-500/50 transition-all"
                disabled={loading}
            >
                <option value={250}>回溯 1年</option>
                <option value={500}>回溯 2年</option>
                <option value={750}>回溯 3年</option>
            </select>
            <button
                onclick={runAnalysis}
                disabled={loading}
                class="px-5 py-2.5 bg-primary-600 hover:bg-primary-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-bold rounded-lg transition-all shadow-[0_0_15px_rgba(59,130,246,0.3)] hover:shadow-[0_0_25px_rgba(59,130,246,0.5)] flex items-center gap-2"
            >
                {#if loading}
                    <span
                        class="w-4 h-4 rounded-full border-2 border-white/30 border-t-white animate-spin"
                    ></span>
                    计算中...
                {:else}
                    <span>⚡</span> 开始分析
                {/if}
            </button>
        </div>
    </div>

    <!-- 错误提示 -->
    {#if error}
        <div
            class="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 text-rose-300 text-sm"
        >
            ⚠️ {error}
        </div>
    {/if}

    <!-- 空状态 -->
    {#if !result && !loading && !error}
        <div
            class="py-32 flex flex-col items-center justify-center text-center opacity-40 space-y-4"
        >
            <span class="text-7xl grayscale">🧪</span>
            <p class="text-white text-lg font-bold tracking-widest uppercase">
                等待因子检验
            </p>
            <p class="text-white/50 text-sm max-w-md leading-relaxed">
                输入股票代码，系统将从历史 K
                线数据中计算各技术因子对未来收益的预测能力（IC/ICIR）。
            </p>
        </div>
    {/if}

    <!-- 加载骨架屏 -->
    {#if loading}
        <div class="grid grid-cols-1 gap-4 animate-pulse">
            {#each Array(4) as _}
                <div class="h-16 rounded-xl bg-white/5"></div>
            {/each}
        </div>
    {/if}

    <!-- 结果区域 -->
    {#if result}
        <!-- 统计徽章 -->
        <div class="flex flex-wrap gap-3 mb-2">
            <span
                class="text-xs bg-white/5 border border-white/10 rounded-full px-3 py-1 text-white/60"
            >
                📊 历史数据：{result.total_bars} 条
            </span>
            <span
                class="text-xs bg-white/5 border border-white/10 rounded-full px-3 py-1 text-white/60"
            >
                🏷️ 分析标的：{result.symbol} ({result.market})
            </span>
            <span
                class="text-xs bg-emerald-500/10 border border-emerald-500/20 rounded-full px-3 py-1 text-emerald-400"
            >
                ✅ 推荐因子：{result.factor_summary?.filter((f) =>
                    f.suggestion.startsWith("✅"),
                ).length} 个
            </span>
        </div>

        <!-- Tab 导航 -->
        <div
            class="flex gap-1 bg-white/5 border border-white/10 rounded-xl p-1 w-fit"
        >
            {#each [{ key: "summary", label: "IC 排行榜" }, { key: "decay", label: "IC 衰减曲线" }, { key: "corr", label: "因子相关热力图" }] as tab}
                <button
                    onclick={() => (activeTab = tab.key)}
                    class="px-4 py-2 rounded-lg text-sm font-medium transition-all {activeTab ===
                    tab.key
                        ? 'bg-primary-600 text-white shadow'
                        : 'text-white/50 hover:text-white'}"
                >
                    {tab.label}
                </button>
            {/each}
        </div>

        <!-- ── Tab: IC 排行榜 ── -->
        {#if activeTab === "summary"}
            <div class="space-y-3">
                <!-- 表头 -->
                <div
                    class="grid grid-cols-12 gap-2 px-4 py-2 text-xs text-white/30 uppercase tracking-widest"
                >
                    <div class="col-span-3">因子</div>
                    <div class="col-span-2 text-center">IC 均值</div>
                    <div class="col-span-2 text-center">ICIR</div>
                    <div class="col-span-2 text-center">IC>0 率</div>
                    <div class="col-span-3">预测强度</div>
                </div>

                {#each result.factor_summary as f, i}
                    <div
                        class="bg-[#12141c]/90 border border-white/5 rounded-xl p-4 hover:border-white/10 transition-all animate-in slide-in-from-bottom-2 fade-in duration-300"
                        style="animation-delay: {i * 40}ms"
                    >
                        <div class="grid grid-cols-12 gap-2 items-center">
                            <!-- 因子名 -->
                            <div class="col-span-3">
                                <div
                                    class="font-mono font-bold text-white text-sm"
                                >
                                    {f.factor}
                                </div>
                                <div class="text-white/40 text-xs mt-0.5">
                                    {f.name}
                                </div>
                            </div>
                            <!-- IC 均值 -->
                            <div class="col-span-2 text-center">
                                <span
                                    class="font-mono font-bold {icColor(
                                        f.ic_mean,
                                    )}"
                                    >{f.ic_mean > 0 ? "+" : ""}{f.ic_mean}</span
                                >
                            </div>
                            <!-- ICIR -->
                            <div class="col-span-2 text-center">
                                <span
                                    class="font-mono font-bold {f.icir >= 0.5
                                        ? 'text-emerald-400'
                                        : f.icir <= -0.5
                                          ? 'text-rose-400'
                                          : 'text-white/60'}"
                                    >{f.icir > 0 ? "+" : ""}{f.icir}</span
                                >
                            </div>
                            <!-- IC>0 率 -->
                            <div class="col-span-2 text-center">
                                <span class="text-blue-300 font-mono"
                                    >{(f.ic_positive_rate * 100).toFixed(
                                        1,
                                    )}%</span
                                >
                            </div>
                            <!-- 进度条 + 建议 -->
                            <div class="col-span-3 space-y-1.5">
                                <div
                                    class="h-2 rounded-full bg-white/10 overflow-hidden"
                                >
                                    <div
                                        class="h-full rounded-full transition-all duration-700 {icirColor(
                                            f.icir,
                                        )}"
                                        style="width: {barWidth(f.icir)}%"
                                    ></div>
                                </div>
                                <span class="text-xs text-white/50"
                                    >{f.suggestion}</span
                                >
                            </div>
                        </div>
                    </div>
                {/each}
            </div>

            <!-- 说明卡 -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                {#each [{ color: "emerald", title: "✅ 推荐使用", desc: "ICIR ≥ 0.5 且 |IC| ≥ 0.03，预测信号稳定且显著" }, { color: "amber", title: "⚠️ 参考使用", desc: "|IC| ≥ 0.02，有一定预测能力，建议配合其它因子组合使用" }, { color: "rose", title: "❌ 预测力弱", desc: "|IC| < 0.02，在当前股票历史上缺乏统计意义" }] as card}
                    <div
                        class="bg-{card.color}-500/5 border border-{card.color}-500/20 rounded-xl p-4"
                    >
                        <div
                            class="font-bold text-{card.color}-400 text-sm mb-1"
                        >
                            {card.title}
                        </div>
                        <div class="text-xs text-white/50 leading-relaxed">
                            {card.desc}
                        </div>
                    </div>
                {/each}
            </div>
        {/if}

        <!-- ── Tab: IC 衰减曲线 ── -->
        {#if activeTab === "decay"}
            <div class="bg-[#0d0f14] border border-white/5 rounded-xl p-4">
                <div class="text-sm text-white/50 mb-4">
                    各因子在不同预测期（1日/3日/5日/10日/20日）的 IC 值。IC
                    随时间快速衰减说明因子更适合短线；缓慢衰减适合波段。
                </div>
                <canvas id="decayCanvas" width="860" height="320" class="w-full"
                ></canvas>
                <!-- 图例 -->
                <div class="flex flex-wrap gap-3 mt-4 justify-center">
                    {#each Object.keys(result.ic_decay.factors) as name, i}
                        <div
                            class="flex items-center gap-1.5 text-xs text-white/60"
                        >
                            <div
                                class="w-4 h-1 rounded-full"
                                style="background:{FACTOR_COLORS[
                                    i % FACTOR_COLORS.length
                                ]}"
                            ></div>
                            {name}
                        </div>
                    {/each}
                </div>
            </div>
        {/if}

        <!-- ── Tab: 因子相关热力图 ── -->
        {#if activeTab === "corr"}
            <div class="bg-[#0d0f14] border border-white/5 rounded-xl p-4">
                <div class="text-sm text-white/50 mb-4">
                    因子间的 Spearman
                    相关系数。深红色=高度正相关（冗余），深蓝色=高度负相关，接近
                    0=相互独立（组合价值高）。
                </div>
                <canvas
                    id="corrCanvas"
                    width="560"
                    height="560"
                    class="max-w-[560px] mx-auto block"
                ></canvas>
                <!-- 色阶 -->
                <div
                    class="flex items-center gap-3 mt-4 justify-center text-xs text-white/40"
                >
                    <span>-1.0 (负相关)</span>
                    <div
                        class="w-32 h-2 rounded-full"
                        style="background: linear-gradient(to right, rgb(59,130,246), #1e2130, rgb(239,68,68))"
                    ></div>
                    <span>+1.0 (正相关)</span>
                </div>
            </div>
        {/if}
    {/if}
</div>
