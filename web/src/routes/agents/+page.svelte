<script>
    import { onMount } from "svelte";

    let symbol = $state("AMAT.O");
    let analyzing = $state(false);
    let chatLog = $state([]);
    let llmProvider = "null";
    let modelName = "null";

    // 历史记忆面板状态
    let memoryPanelOpen = $state(false);
    let memoryHistory = $state(/** @type {any[]} */ ([]));
    let globalInsights = $state(/** @type {any[]} */ ([]));
    let memoryLoading = $state(false);

    // 辩论回合与共识评分
    let debateRound = $state(0);
    let crossReviewRound = $state(0);
    /** @type {number|null} */
    let consensusScore = $state(null);
    let showRawData = $state(null);

    // 阶段进度指示器
    let currentPhase = $state(0);
    let phaseLabel = $state("");
    let totalPhases = $state(6);

    onMount(() => {
        llmProvider = localStorage.getItem("llmProvider") || "null";
        modelName = localStorage.getItem("modelName") || "null";
    });

    const roleMeta = {
        "Data Engineer": {
            icon: "🗃️",
            color: "text-blue-400",
            bg: "bg-blue-500/10",
            border: "border-blue-500/20",
            name: "数据工程师",
        },
        "Macro Analyst": {
            icon: "🏛️",
            color: "text-purple-400",
            bg: "bg-purple-500/10",
            border: "border-purple-500/20",
            name: "宏观分析师",
        },
        "Quant Researcher": {
            icon: "🧮",
            color: "text-cyan-400",
            bg: "bg-cyan-500/10",
            border: "border-cyan-500/20",
            name: "量化研究员",
        },
        "Risk Control Agent": {
            icon: "🛡️",
            color: "text-rose-400",
            bg: "bg-rose-500/10",
            border: "border-rose-500/20",
            name: "首席风控官",
        },
        "Portfolio Manager": {
            icon: "💼",
            color: "text-amber-400",
            bg: "bg-amber-500/10",
            border: "border-amber-500/20",
            name: "策略基金经理 (Boss)",
        },
    };

    function startAnalysis() {
        if (!symbol) return;
        analyzing = true;
        chatLog = [];
        debateRound = 0;
        crossReviewRound = 0;
        consensusScore = null;
        currentPhase = 0;
        phaseLabel = "";
        totalPhases = 6;

        const es = new EventSource(
            `/api/agents/${symbol}/stream?provider=${llmProvider}&model=${modelName}`,
        );

        es.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                // 阶段进度更新（任务3）
                if (data.event === "phase_update") {
                    currentPhase = data.phase || 0;
                    phaseLabel = data.phase_label || "";
                    totalPhases = data.total_phases || 6;
                    return;
                }

                // 辩论回合分隔符
                if (
                    data.event === "round_start" ||
                    data.event === "debate_round"
                ) {
                    debateRound = data.round || 0;
                    chatLog.push({
                        role: "__divider__",
                        status: "done",
                        content:
                            data.event === "debate_round"
                                ? `⚔️ 第 ${(data.round || 0) + 1} 轮追加辩论 — 发现显著分歧，决策层点名复盘`
                                : `🔔 第 ${data.round || 1} 轮圆桌辩论开始`,
                        round: data.round,
                    });
                    return;
                }

                // 交叉评论阶段分隔符
                if (data.event === "cross_review_start") {
                    crossReviewRound = data.round || 1;
                    chatLog.push({
                        role: "__divider__",
                        status: "done",
                        content: `🔍 交叉评论第 ${data.round || 1} 轮 — 各位专家互相审阅同行观点`,
                        round: data.round,
                        dividerType: "cross_review",
                    });
                    return;
                }

                // 回应评论阶段分隔符
                if (data.event === "rebuttal_start") {
                    chatLog.push({
                        role: "__divider__",
                        status: "done",
                        content: `💬 回应评论阶段 — 各位专家回应同行的点评`,
                        round: data.round,
                        dividerType: "rebuttal",
                    });
                    return;
                }

                // 交叉评论气泡 — 带审阅对象标签
                if (data.event === "cross_comment" && data.reviewing) {
                    chatLog.push({
                        role: data.role,
                        status: data.status,
                        content: data.content,
                        reviewing: data.reviewing,
                        msgType: "cross_comment",
                        skills: [],
                    });
                    return;
                }

                // 回应评论气泡 — 带评论者标签 + 原始评论引用（任务5）
                if (data.event === "rebuttal" && data.commenters) {
                    chatLog.push({
                        role: data.role,
                        status: data.status,
                        content: data.content,
                        commenters: data.commenters,
                        commentQuotes: data.comment_quotes || [],
                        msgType: "rebuttal",
                        skills: [],
                    });
                    return;
                }

                let lastEntry = chatLog[chatLog.length - 1];

                // 带技能强化徽章的全新气泡
                if (
                    data.skills &&
                    data.skills.length > 0 &&
                    (!lastEntry ||
                        lastEntry.role !== data.role ||
                        lastEntry.role === "__divider__")
                ) {
                    chatLog.push({
                        role: data.role,
                        status: data.status,
                        content: data.content,
                        raw_data: data.raw_data || null,
                        skills: data.skills,
                    });
                    return;
                }

                if (
                    lastEntry &&
                    lastEntry.role === data.role &&
                    lastEntry.role !== "__divider__"
                ) {
                    lastEntry.content += data.content;
                    lastEntry.status = data.status;
                    if (data.raw_data) lastEntry.raw_data = data.raw_data;
                } else {
                    chatLog.push({
                        role: data.role,
                        status: data.status,
                        content: data.content,
                        raw_data: data.raw_data || null,
                        skills: data.skills || [],
                    });
                }

                // 提取共识评分
                if (
                    data.status === "done" &&
                    data.role === "Portfolio Manager" &&
                    typeof data.content === "string"
                ) {
                    const match = data.content.match(/(\d+)\/100/);
                    if (match) consensusScore = parseInt(match[1]);
                    es.close();
                    analyzing = false;
                    if (memoryPanelOpen) loadMemory();
                }
            } catch (e) {
                console.error("SSE parse error", e);
            }
        };

        es.onerror = () => {
            es.close();
            analyzing = false;
        };
    }

    async function loadMemory() {
        memoryLoading = true;
        try {
            const [mr, ir] = await Promise.all([
                fetch(`/api/agents/memory/${symbol}`),
                fetch(`/api/agents/insights`),
            ]);
            memoryHistory = (await mr.json()).history || [];
            globalInsights = (await ir.json()).insights || [];
        } catch {}
        memoryLoading = false;
    }

    function consensusColor(s) {
        if (s >= 65) return "text-green-400";
        if (s <= 35) return "text-rose-400";
        return "text-amber-400";
    }
</script>

<svelte:head>
    <title>多智能体作战室 - AlphaPulse</title>
</svelte:head>

<div class="p-6 md:p-8 max-w-5xl mx-auto space-y-6">
    <div
        class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8"
    >
        <div>
            <h1
                class="text-3xl font-bold tracking-tight text-white flex items-center gap-3"
            >
                <span class="text-4xl text-shadow-glow">🤖</span> 多智能体(Agents)推演沙盘
            </h1>
            <p class="text-white/50 mt-2 text-sm leading-relaxed max-w-2xl">
                AlphaPulse Safari Room 🚀
                让他们并肩为您撕裂盘面数据，从宏观、量化到合规风控，最终由投资组合经理统一裁量。
            </p>
        </div>

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
                    placeholder="输入资产代码 (如 AAPL)"
                    class="w-48 bg-[#0a0c10] border border-white/10 rounded-lg py-2.5 pl-9 pr-4 text-sm text-white placeholder:text-white/20 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/50 transition-all font-mono uppercase font-bold tracking-wider"
                    disabled={analyzing}
                    onkeydown={(e) => {
                        if (e.key === "Enter") startAnalysis();
                    }}
                />
            </div>

            <button
                onclick={startAnalysis}
                disabled={analyzing}
                class="px-5 py-2.5 bg-primary-600 hover:bg-primary-500 disabled:bg-primary-600/50 disabled:cursor-not-allowed text-white text-sm font-bold tracking-wide rounded-lg transition-all shadow-[0_0_15px_rgba(59,130,246,0.3)] hover:shadow-[0_0_25px_rgba(59,130,246,0.5)] flex items-center gap-2"
            >
                {#if analyzing}
                    <span
                        class="w-4 h-4 rounded-full border-2 border-white/30 border-t-white animate-spin"
                    ></span>
                    全线程推演中...
                {:else}
                    <span>⚡</span> 召唤投资委员会
                {/if}
            </button>
        </div>

        <!-- 共识评分 + 记忆面板入口 -->
        {#if consensusScore !== null}
            <div class="flex items-center gap-3 mt-2">
                <span class="text-xs text-white/40">多空共识评分:</span>
                <span class="font-bold text-lg {consensusColor(consensusScore)}"
                    >{consensusScore}/100</span
                >
                <div
                    class="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden"
                >
                    <div
                        class="h-full bg-gradient-to-r from-rose-500 via-amber-400 to-green-500 transition-all"
                        style="width:{consensusScore}%"
                    ></div>
                </div>
            </div>
        {/if}
    </div>

    <!-- 阶段进度指示器（任务3）— 推演进行中时显示 -->
    {#if analyzing && currentPhase > 0}
        <div
            class="flex items-center gap-3 mb-4 bg-surface-800/60 rounded-xl border border-white/5 px-4 py-3 backdrop-blur-md"
        >
            <span class="text-xs text-white/40 shrink-0"
                >Phase {currentPhase}/{totalPhases}</span
            >
            <div class="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                    class="h-full bg-gradient-to-r from-primary-500 to-indigo-500 rounded-full transition-all duration-700 ease-out"
                    style="width:{Math.round(
                        (currentPhase / totalPhases) * 100,
                    )}%"
                ></div>
            </div>
            <span
                class="text-xs font-bold text-primary-400 shrink-0 tracking-wide"
                >{phaseLabel}</span
            >
        </div>
    {/if}
    <!-- 历史记忆侧边抽屉 -->
    {#if memoryPanelOpen}
        <div
            class="fixed inset-y-0 right-0 w-80 z-50 bg-[#0d0f14] border-l border-white/10 shadow-2xl flex flex-col"
        >
            <div
                class="flex items-center justify-between p-4 border-b border-white/10"
            >
                <h3 class="font-bold text-white text-sm">📚 历史推演洞察</h3>
                <button
                    onclick={() => (memoryPanelOpen = false)}
                    class="text-white/40 hover:text-white transition-colors"
                    >✕</button
                >
            </div>
            <div
                class="flex-1 overflow-y-auto p-4 space-y-4 text-xs text-white/70"
            >
                {#if memoryLoading}
                    <div class="flex justify-center py-8">
                        <div
                            class="w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full animate-spin"
                        ></div>
                    </div>
                {:else}
                    {#if memoryHistory.length > 0}
                        <div
                            class="font-semibold text-white/50 uppercase tracking-wider mb-2"
                        >
                            📂 {symbol} 历史推演
                        </div>
                        {#each memoryHistory as h}
                            <div class="bg-white/5 rounded-lg p-3 space-y-1">
                                <div class="text-white/40">
                                    {(h.timestamp || "").slice(0, 10)}
                                </div>
                                <div
                                    class="{consensusColor(
                                        h.consensus?.score || 50,
                                    )} font-bold"
                                >
                                    {h.consensus?.score || "-"}/100 {h.consensus
                                        ?.verdict || ""}
                                </div>
                                <div class="text-white/60 leading-relaxed">
                                    {(h.verdict_summary || "").slice(0, 120)}...
                                </div>
                            </div>
                        {/each}
                    {:else}
                        <div class="text-white/30 text-center py-6">
                            暂无历史推演记录
                        </div>
                    {/if}
                    {#if globalInsights.length > 0}
                        <div
                            class="font-semibold text-white/50 uppercase tracking-wider mt-4 mb-2"
                        >
                            🧠 全局智慧库
                        </div>
                        {#each globalInsights as ins}
                            <div
                                class="bg-amber-500/10 border border-amber-500/20 rounded-lg p-2.5"
                            >
                                <div class="text-amber-300/80">
                                    {ins.content}
                                </div>
                                <div class="text-white/20 mt-1">
                                    {(ins.timestamp || "").slice(0, 10)}
                                </div>
                            </div>
                        {/each}
                    {/if}
                {/if}
            </div>
        </div>
        <!-- 遮罩 -->
        <button
            class="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm"
            onclick={() => (memoryPanelOpen = false)}
        ></button>
    {/if}
    <!-- 顶部功能：记忆面板入口 -->
    <div class="flex items-center gap-3 -mt-3 mb-2">
        <button
            onclick={() => {
                memoryPanelOpen = true;
                loadMemory();
            }}
            class="text-xs text-white/40 hover:text-white/80 transition-colors flex items-center gap-1.5 border border-white/10 hover:border-white/20 rounded-lg px-3 py-1.5"
        >
            📚 历史洞察
        </button>
        {#if debateRound > 0}
            <span
                class="text-xs text-amber-400 border border-amber-500/30 rounded-lg px-3 py-1.5"
            >
                ⚔️ 已进行 {debateRound} 轮追加辩论
            </span>
        {/if}
        {#if crossReviewRound > 0}
            <span
                class="text-xs text-indigo-400 border border-indigo-500/30 rounded-lg px-3 py-1.5"
            >
                🔍 已进行 {crossReviewRound} 轮交叉评论
            </span>
        {/if}
    </div>

    <!-- 瀑布流沙盘区 -->
    <div class="space-y-4 pb-20 relative">
        <!-- 虚线轨道轴 -->
        <div
            class="absolute left-[33px] top-0 bottom-0 w-px border-l-2 border-dashed border-white/5 -z-10"
        ></div>

        {#if chatLog.length === 0 && !analyzing}
            <div
                class="py-24 flex flex-col items-center justify-center text-center opacity-40"
            >
                <span class="text-6xl mb-6 grayscale">🎙️</span>
                <p
                    class="text-white text-lg font-bold tracking-widest uppercase"
                >
                    委员会坐席已就绪
                </p>
                <p class="text-white/50 text-sm mt-3 max-w-md leading-relaxed">
                    按下右上方按钮，数据工程师将瞬间唤醒整条量化大动脉。为您带来真正原汁原味的华尔街战情会议室体验。
                </p>
            </div>
        {/if}

        {#each chatLog as log}
            {#if log.role === "__divider__"}
                <!-- 辩论回合/交叉评论分隔符（使用静态类名映射，避免 Tailwind JIT 编译遗漏） -->
                {@const divStyles =
                    log.dividerType === "cross_review"
                        ? {
                              line: "bg-indigo-500/20",
                              text: "text-indigo-400 bg-indigo-500/10 border-indigo-500/30",
                          }
                        : log.dividerType === "rebuttal"
                          ? {
                                line: "bg-cyan-500/20",
                                text: "text-cyan-400 bg-cyan-500/10 border-cyan-500/30",
                            }
                          : {
                                line: "bg-amber-500/20",
                                text: "text-amber-400 bg-amber-500/10 border-amber-500/30",
                            }}
                <div
                    class="flex items-center gap-3 my-4 animate-in fade-in duration-500"
                >
                    <div class="flex-1 h-px {divStyles.line}"></div>
                    <span
                        class="text-xs font-bold {divStyles.text} border rounded-full px-4 py-1.5 tracking-wide"
                        >{log.content}</span
                    >
                    <div class="flex-1 h-px {divStyles.line}"></div>
                </div>
            {:else}
                {@const meta = roleMeta[log.role] || {
                    icon: "👤",
                    color: "text-white",
                    bg: "bg-white/10",
                    border: "border-white/20",
                    name: log.role,
                }}
                <div
                    class="flex gap-5 group animate-in slide-in-from-bottom-4 fade-in duration-500"
                >
                    <!-- 头像列 (带发光光晕) -->
                    <div class="shrink-0 flex flex-col items-center">
                        <div
                            class="w-16 h-16 rounded-2xl {meta.bg} {meta.border} border-2 flex items-center justify-center text-3xl shadow-lg relative overflow-hidden transition-all duration-300 {log.status ===
                            'typing'
                                ? 'shadow-[0_0_20px_var(--tw-shadow-color)] ' +
                                  meta.color.replace('text-', 'shadow-')
                                : ''}"
                        >
                            {#if log.status === "typing"}
                                <div
                                    class="absolute inset-0 bg-white/10 animate-pulse"
                                ></div>
                            {/if}
                            <span class="relative z-10">{meta.icon}</span>
                        </div>
                    </div>

                    <!-- 内容气泡列 -->
                    <div class="flex-1 pt-1 mb-8 w-0">
                        <div class="flex items-center gap-3 mb-2.5 flex-wrap">
                            <span
                                class="font-bold text-base {meta.color} tracking-wide"
                                >{meta.name}</span
                            >
                            <span
                                class="text-white/30 text-xs font-mono px-2 py-0.5 rounded bg-white/5"
                                >{log.role}</span
                            >
                            <!-- 交叉评论标签：审阅对象 -->
                            {#if log.msgType === "cross_comment" && log.reviewing}
                                <span
                                    class="text-xs bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-2 py-0.5 rounded-full flex items-center gap-1"
                                >
                                    🔍 审阅
                                    {#each log.reviewing as peer, i}
                                        <span class="font-bold"
                                            >{roleMeta[peer]?.name ||
                                                peer}</span
                                        >{#if i < log.reviewing.length - 1}<span
                                                >、</span
                                            >{/if}
                                    {/each}
                                </span>
                            {/if}
                            <!-- 回应评论标签：评论者 -->
                            {#if log.msgType === "rebuttal" && log.commenters}
                                <span
                                    class="text-xs bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 px-2 py-0.5 rounded-full flex items-center gap-1"
                                >
                                    💬 回应
                                    {#each log.commenters as commenter, i}
                                        <span class="font-bold"
                                            >{roleMeta[commenter]?.name ||
                                                commenter}</span
                                        >{#if i < log.commenters.length - 1}<span
                                                >、</span
                                            >{/if}
                                    {/each}
                                </span>
                            {/if}
                            <!-- 技能强化徽章 -->
                            {#each log.skills || [] as skill}
                                <span
                                    class="text-xs bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-2 py-0.5 rounded-full"
                                    >⚡ {skill}</span
                                >
                            {/each}
                            {#if log.status === "typing"}
                                <div
                                    class="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/5 border border-white/10 shadow-inner"
                                >
                                    <span
                                        class="w-1.5 h-1.5 rounded-full {meta.bg.replace(
                                            '/10',
                                            '',
                                        )} animate-bounce"
                                        style="animation-delay: 0ms"
                                    ></span>
                                    <span
                                        class="w-1.5 h-1.5 rounded-full {meta.bg.replace(
                                            '/10',
                                            '',
                                        )} animate-bounce"
                                        style="animation-delay: 150ms"
                                    ></span>
                                    <span
                                        class="w-1.5 h-1.5 rounded-full {meta.bg.replace(
                                            '/10',
                                            '',
                                        )} animate-bounce"
                                        style="animation-delay: 300ms"
                                    ></span>
                                    <span
                                        class="text-[11px] font-bold text-white/50 ml-1"
                                        >语音输出中...</span
                                    >
                                </div>
                            {:else if log.status === "error"}
                                <span
                                    class="px-2 py-0.5 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs font-bold shadow-sm"
                                    >⚠️ 节点被熔断</span
                                >
                            {:else}
                                <span
                                    class="px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-bold shadow-sm flex items-center gap-1"
                                    ><span>✓</span> 发言完毕</span
                                >
                            {/if}
                        </div>

                        <!-- 气泡框体 -->
                        <div class="prose prose-invert prose-sm max-w-none">
                            <div
                                class="bg-[#12141c]/90 backdrop-blur-md border {meta.border} rounded-2xl rounded-tl-sm p-6 shadow-xl relative leading-relaxed text-white/90 font-sans transition-all hover:bg-[#151821]
                            prose-blockquote:border-l-4 prose-blockquote:border-emerald-500/50 prose-blockquote:bg-emerald-500/5 prose-blockquote:px-4 prose-blockquote:py-2 prose-blockquote:rounded-r-lg prose-blockquote:not-italic prose-blockquote:my-3 prose-blockquote:text-emerald-400/90 hover:prose-blockquote:bg-emerald-500/10 transition-all
                            prose-code:text-amber-300 prose-code:bg-amber-500/10 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:before:content-none prose-code:after:content-none"
                            >
                                <!-- 回应评论时的原始评论引用折叠框（任务5） -->
                                {#if log.msgType === "rebuttal" && log.commentQuotes && log.commentQuotes.length > 0}
                                    <details
                                        class="mb-4 border border-white/10 rounded-lg overflow-hidden bg-black/30"
                                    >
                                        <summary
                                            class="cursor-pointer text-xs font-bold text-white/40 hover:text-white/70 px-3 py-2 flex items-center gap-2 select-none transition-colors"
                                        >
                                            <span>💬</span> 查看原始评论 ({log
                                                .commentQuotes.length} 条)
                                        </summary>
                                        <div class="px-3 pb-3 space-y-2">
                                            {#each log.commentQuotes as quote}
                                                <div
                                                    class="border-l-2 border-cyan-500/40 bg-cyan-500/5 rounded-r-lg px-3 py-2 text-xs text-white/60"
                                                >
                                                    <span
                                                        class="font-bold text-cyan-400/80"
                                                        >{roleMeta[quote.from]
                                                            ?.name ||
                                                            quote.from}：</span
                                                    >
                                                    <span class="italic"
                                                        >{quote.text}</span
                                                    >
                                                </div>
                                            {/each}
                                        </div>
                                    </details>
                                {/if}

                                <!-- 发言内容（任务6：超过 300 字自动折叠） -->
                                {#if log.content && log.content.length > 300 && log.status === "done"}
                                    {@const contentId = `bubble-${log.role}-${chatLog.indexOf(log)}`}
                                    <div id={contentId}>
                                        <div
                                            class="whitespace-pre-wrap font-mono text-[14px] leading-7"
                                        >
                                            {log.content.slice(0, 300)}...
                                        </div>
                                        <button
                                            onclick={(e) => {
                                                const container =
                                                    e.currentTarget
                                                        .parentElement;
                                                const textDiv =
                                                    container.querySelector(
                                                        ".truncated-text",
                                                    );
                                                const fullDiv =
                                                    container.querySelector(
                                                        ".full-text",
                                                    );
                                                if (textDiv && fullDiv) {
                                                    textDiv.classList.toggle(
                                                        "hidden",
                                                    );
                                                    fullDiv.classList.toggle(
                                                        "hidden",
                                                    );
                                                    e.currentTarget.textContent =
                                                        fullDiv.classList.contains(
                                                            "hidden",
                                                        )
                                                            ? "📖 展开全文"
                                                            : "📕 收起";
                                                } else {
                                                    container.innerHTML = "";
                                                    const full =
                                                        document.createElement(
                                                            "div",
                                                        );
                                                    full.className =
                                                        "whitespace-pre-wrap font-mono text-[14px] leading-7";
                                                    full.textContent =
                                                        log.content;
                                                    container.appendChild(full);
                                                }
                                            }}
                                            class="mt-2 text-xs text-primary-400 hover:text-primary-300 transition-colors underline underline-offset-2"
                                            >📖 展开全文</button
                                        >
                                    </div>
                                {:else}
                                    <div
                                        class="whitespace-pre-wrap font-mono text-[14px] leading-7"
                                    >
                                        {log.content}
                                    </div>
                                {/if}

                                <!-- 尾部闪烁的光标 -->
                                {#if log.status === "typing"}
                                    <span
                                        class="inline-block w-2.5 h-5 {meta.bg.replace(
                                            '/10',
                                            '',
                                        )} ml-1 animate-pulse align-middle rounded-sm"
                                    ></span>
                                {/if}

                                <!-- 底层数据查阅抽屉 (仅当存在 raw_data 时渲染) -->
                                {#if log.raw_data}
                                    <details
                                        class="mt-6 border border-white/10 rounded-xl p-4 bg-black/40 group/details custom-scrollbar"
                                    >
                                        <summary
                                            class="cursor-pointer text-xs font-bold text-white/50 hover:text-white transition-colors flex items-center gap-2 select-none"
                                        >
                                            <span
                                                class="group-open/details:rotate-90 transition-transform"
                                                >▶</span
                                            >
                                            <span>📊</span> 点击展开可读化的决策底层数据卡片
                                        </summary>
                                        <div
                                            class="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pb-2"
                                        >
                                            {#each ["technical", "fundamental", "sentiment", "alternative", "macro"] as category}
                                                {#if log.raw_data[category] && Object.keys(log.raw_data[category]).length > 0}
                                                    <div
                                                        class="bg-white/5 rounded-lg border border-white/10 overflow-hidden shadow-inner"
                                                    >
                                                        <div
                                                            class="px-3 py-2 bg-white/10 text-[11px] font-bold text-white/70 uppercase tracking-wider border-b border-white/10"
                                                        >
                                                            {category.toUpperCase()}
                                                            提取指标簇
                                                        </div>
                                                        <div
                                                            class="p-2 space-y-1 bg-black/20"
                                                        >
                                                            {#each Object.entries(log.raw_data[category]) as [k, v]}
                                                                <div
                                                                    class="flex justify-between items-center text-xs px-2 py-1 hover:bg-white/5 rounded transition-colors"
                                                                >
                                                                    <span
                                                                        class="text-white/40"
                                                                        >{k}</span
                                                                    >
                                                                    <span
                                                                        class="text-white font-mono"
                                                                        >{typeof v ===
                                                                        "number"
                                                                            ? v.toFixed(
                                                                                  2,
                                                                              )
                                                                            : String(
                                                                                  v,
                                                                              )}</span
                                                                    >
                                                                </div>
                                                            {/each}
                                                        </div>
                                                    </div>
                                                {/if}
                                            {/each}
                                        </div>
                                    </details>
                                {/if}
                            </div>
                        </div>
                    </div>
                </div>
            {/if}
        {/each}

        {#if chatLog.length > 0 && chatLog[chatLog.length - 1].role === "Portfolio Manager" && chatLog[chatLog.length - 1].status === "done"}
            <div
                class="py-12 flex justify-center animate-in zoom-in fade-in duration-1000"
            >
                <div
                    class="px-8 py-4 bg-gradient-to-r from-amber-500/20 via-orange-500/20 to-rose-500/20 text-amber-500 font-extrabold text-lg tracking-widest uppercase rounded-full border border-amber-500/40 flex items-center gap-3 shadow-[0_0_40px_rgba(245,158,11,0.25)] backdrop-blur-xl"
                >
                    <span class="text-2xl animate-bounce">🔥</span> 组合经理已下达最终操盘指令
                </div>
            </div>

            <div class="flex justify-center -mt-6 pb-12">
                <button
                    onclick={startAnalysis}
                    class="text-white/40 hover:text-white transition-colors text-sm underline underline-offset-4"
                >
                    重新发启对于 {symbol} 的研讨
                </button>
            </div>
        {/if}
    </div>
</div>
