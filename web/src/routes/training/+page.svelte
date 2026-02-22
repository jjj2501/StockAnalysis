<script>
    import { onMount, onDestroy } from "svelte";
    import Card from "$lib/components/Card.svelte";
    import { Chart, registerables } from "chart.js";
    Chart.register(...registerables);

    let symbol = $state("600519");
    let training = $state(false);
    let error = $state("");

    /** @type {any[]} */
    let trainingHistory = $state([]);
    /** @type {any} */
    let currentTask = $state(null);
    /** @type {ReturnType<typeof setInterval> | null} */
    let pollTimer = $state(null);

    // 设备信息
    /** @type {any} */
    let gpuStatus = $state(null);

    // 损失曲线 chart
    /** @type {any} */
    let lossChart = $state(null);

    onMount(async () => {
        // 加载设备和历史训练任务
        try {
            const [statusRes, historyRes] = await Promise.all([
                fetch("/api/gpu/status"),
                fetch("/api/gpu/training/status"),
            ]);
            gpuStatus = await statusRes.json();
            trainingHistory = await historyRes.json();

            // 检查是否有正在进行的训练
            const active = trainingHistory.find(
                (/** @type {any} */ t) => t.status === "training",
            );
            if (active) {
                currentTask = active;
                training = true;
                startPolling();
            }
        } catch (/** @type {any} */ e) {
            console.error("加载失败:", e);
        }
    });

    onDestroy(() => {
        if (pollTimer) clearInterval(pollTimer);
        if (lossChart) lossChart.destroy();
    });

    async function startTraining() {
        training = true;
        error = "";
        currentTask = {
            symbol,
            status: "training",
            progress: 0,
            message: "正在提交...",
        };

        try {
            const res = await fetch("/api/gpu/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ symbol }),
            });
            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.detail || "请求失败");
            }
            startPolling();
        } catch (/** @type {any} */ e) {
            error = e.message;
            training = false;
            currentTask = null;
        }
    }

    function startPolling() {
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(async () => {
            try {
                const res = await fetch(
                    `/api/gpu/training/status?symbol=${symbol}`,
                );
                const tasks = await res.json();
                if (tasks.length > 0) {
                    // 获取最新的训练任务
                    const latest = tasks[tasks.length - 1];
                    currentTask = latest;

                    if (
                        latest.status === "completed" ||
                        latest.status === "failed"
                    ) {
                        if (pollTimer) clearInterval(pollTimer);
                        pollTimer = null;
                        training = false;

                        // 刷新历史
                        const histRes = await fetch("/api/gpu/training/status");
                        trainingHistory = await histRes.json();

                        // 如果训练完成，绘制损失曲线
                        if (
                            latest.status === "completed" &&
                            latest.result?.train_losses
                        ) {
                            setTimeout(
                                () => renderLossChart(latest.result),
                                100,
                            );
                        }
                    }
                }
            } catch (/** @type {any} */ e) {
                console.error("轮询失败:", e);
            }
        }, 2000);
    }

    function renderLossChart(/** @type {any} */ result) {
        const canvas = /** @type {HTMLCanvasElement | null} */ (
            document.getElementById("lossChart")
        );
        if (!canvas) return;
        if (lossChart) lossChart.destroy();

        const trainLosses = result.train_losses || [];
        const valLosses = result.val_losses || [];
        const labels = trainLosses.map(
            (/** @type {any} */ _v, /** @type {number} */ i) => `${i + 1}`,
        );

        lossChart = new Chart(canvas, {
            type: "line",
            data: {
                labels,
                datasets: [
                    {
                        label: "训练损失",
                        data: trainLosses,
                        borderColor: "rgba(59,130,246,0.8)",
                        backgroundColor: "rgba(59,130,246,0.05)",
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        borderWidth: 2,
                    },
                    {
                        label: "验证损失",
                        data: valLosses,
                        borderColor: "rgba(249,115,22,0.8)",
                        backgroundColor: "rgba(249,115,22,0.05)",
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        borderWidth: 2,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: "rgba(255,255,255,0.6)",
                            font: { size: 11 },
                        },
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "训练轮次",
                            color: "rgba(255,255,255,0.3)",
                        },
                        ticks: {
                            color: "rgba(255,255,255,0.3)",
                            maxTicksLimit: 10,
                        },
                        grid: { color: "rgba(255,255,255,0.03)" },
                    },
                    y: {
                        title: {
                            display: true,
                            text: "损失值 (MSE)",
                            color: "rgba(255,255,255,0.3)",
                        },
                        ticks: { color: "rgba(255,255,255,0.3)" },
                        grid: { color: "rgba(255,255,255,0.05)" },
                    },
                },
            },
        });
    }

    function fmtTime(/** @type {number} */ sec) {
        if (sec < 60) return sec.toFixed(1) + " 秒";
        return (sec / 60).toFixed(1) + " 分钟";
    }

    function statusColor(/** @type {string} */ s) {
        if (s === "completed") return "text-emerald-400";
        if (s === "failed") return "text-rose-400";
        if (s === "training") return "text-blue-400";
        return "text-white/40";
    }

    function statusLabel(/** @type {string} */ s) {
        if (s === "completed") return "✅ 训练完成";
        if (s === "failed") return "❌ 训练失败";
        if (s === "training") return "⏳ 训练中";
        return s;
    }
</script>

<svelte:head>
    <title>模型训练 - AlphaPulse</title>
</svelte:head>

<div class="space-y-8">
    <!-- 标题 -->
    <div
        class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4"
    >
        <div>
            <h2 class="text-2xl font-bold">模型训练</h2>
            <p class="text-white/40 mt-1">
                训练 Transformer+LSTM 混合模型预测股票趋势
            </p>
        </div>
        <!-- 设备信息摘要 -->
        {#if gpuStatus}
            <div
                class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/5 text-xs"
            >
                <span
                    class="w-2 h-2 rounded-full {gpuStatus.gpu_available
                        ? 'bg-emerald-400'
                        : 'bg-amber-400'}"
                ></span>
                <span class="text-white/50">
                    {gpuStatus.gpu_available
                        ? gpuStatus.gpu_info?.device_name
                        : "CPU 模式"}
                </span>
            </div>
        {/if}
    </div>

    <!-- 训练启动 -->
    <Card title="🚀 启动训练">
        <div class="flex flex-col sm:flex-row gap-4 items-start sm:items-end">
            <div class="flex-1">
                <label class="block text-xs text-white/40 mb-1.5 font-medium"
                    >股票代码</label
                >
                <input
                    type="text"
                    bind:value={symbol}
                    placeholder="例: 600519"
                    disabled={training}
                    class="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-primary-500/50 transition-all disabled:opacity-50"
                />
            </div>
            <button
                onclick={startTraining}
                disabled={training || !symbol}
                class="px-6 py-2.5 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 text-white text-sm font-medium rounded-xl transition-all shadow-lg shadow-primary-600/20 whitespace-nowrap"
            >
                {training ? "⏳ 训练中..." : "🎯 开始训练"}
            </button>
        </div>

        <!-- 快捷按钮 -->
        {#if !training}
            <div class="flex flex-wrap gap-2 mt-4">
                {#each [{ code: "600519", name: "贵州茅台" }, { code: "000858", name: "五粮液" }, { code: "300750", name: "宁德时代" }, { code: "002594", name: "比亚迪" }, { code: "600036", name: "招商银行" }] as stock}
                    <button
                        onclick={() => (symbol = stock.code)}
                        class="px-2.5 py-1 rounded-lg bg-white/5 border border-white/5 text-xs text-white/40 hover:bg-primary-600/10 hover:text-primary-500 hover:border-primary-500/20 transition-all"
                    >
                        {stock.name}
                    </button>
                {/each}
            </div>
        {/if}

        {#if error}
            <div
                class="mt-4 bg-rose-500/10 border border-rose-500/20 text-rose-400 px-4 py-2.5 rounded-xl text-sm"
            >
                ⚠️ {error}
            </div>
        {/if}
    </Card>

    <!-- 训练进度 -->
    {#if currentTask}
        <Card title="📊 训练进度">
            <div class="space-y-4">
                <!-- 股票代码和状态 -->
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-3">
                        <div
                            class="w-10 h-10 rounded-xl bg-primary-600/20 flex items-center justify-center text-primary-500 font-bold text-sm"
                        >
                            {currentTask.symbol?.slice(-2) || "A"}
                        </div>
                        <div>
                            <div class="text-sm font-semibold text-white">
                                {currentTask.symbol}
                            </div>
                            <div class="text-xs text-white/30">
                                {currentTask.start_time
                                    ? new Date(
                                          currentTask.start_time,
                                      ).toLocaleString("zh-CN")
                                    : ""}
                            </div>
                        </div>
                    </div>
                    <span
                        class="text-sm font-medium {statusColor(
                            currentTask.status,
                        )}"
                    >
                        {statusLabel(currentTask.status)}
                    </span>
                </div>

                <!-- 进度条 -->
                <div>
                    <div class="flex items-center justify-between mb-1.5">
                        <span class="text-xs text-white/40"
                            >{currentTask.message || "准备中..."}</span
                        >
                        <span class="text-xs font-mono text-white/50"
                            >{(currentTask.progress ?? 0).toFixed(1)}%</span
                        >
                    </div>
                    <div class="h-3 bg-white/5 rounded-full overflow-hidden">
                        <div
                            class="h-full rounded-full transition-all duration-500 ease-out
                            {currentTask.status === 'completed'
                                ? 'bg-emerald-500'
                                : currentTask.status === 'failed'
                                  ? 'bg-rose-500'
                                  : 'bg-primary-500'}"
                            style="width: {currentTask.progress ?? 0}%"
                        ></div>
                    </div>
                </div>

                <!-- 训练完成的结果 -->
                {#if currentTask.status === "completed" && currentTask.result}
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
                        <div class="bg-white/[0.03] rounded-xl p-3 text-center">
                            <div class="text-xs text-white/30 mb-1">
                                最佳验证损失
                            </div>
                            <div class="text-lg font-bold text-emerald-400">
                                {currentTask.result.best_val_loss?.toFixed(6)}
                            </div>
                        </div>
                        <div class="bg-white/[0.03] rounded-xl p-3 text-center">
                            <div class="text-xs text-white/30 mb-1">
                                训练耗时
                            </div>
                            <div class="text-lg font-bold text-white/90">
                                {fmtTime(currentTask.result.training_time ?? 0)}
                            </div>
                        </div>
                        <div class="bg-white/[0.03] rounded-xl p-3 text-center">
                            <div class="text-xs text-white/30 mb-1">
                                使用设备
                            </div>
                            <div
                                class="text-lg font-bold {currentTask.result
                                    .gpu_used
                                    ? 'text-emerald-400'
                                    : 'text-amber-400'}"
                            >
                                {currentTask.result.gpu_used ? "GPU" : "CPU"}
                            </div>
                        </div>
                        <div class="bg-white/[0.03] rounded-xl p-3 text-center">
                            <div class="text-xs text-white/30 mb-1">总轮次</div>
                            <div class="text-lg font-bold text-white/90">
                                {currentTask.result.train_losses?.length ?? "—"}
                            </div>
                        </div>
                    </div>

                    <!-- 损失曲线图 -->
                    <div class="pt-4">
                        <div class="text-xs text-white/40 font-medium mb-3">
                            📉 训练 / 验证损失曲线
                        </div>
                        <div class="h-64 bg-white/[0.02] rounded-xl p-3">
                            <canvas id="lossChart"></canvas>
                        </div>
                    </div>
                {/if}
            </div>
        </Card>
    {/if}

    <!-- 训练历史 -->
    {#if trainingHistory.length > 0}
        <Card title="📜 训练记录">
            <div class="space-y-2">
                {#each [...trainingHistory].reverse() as task}
                    <div
                        class="flex items-center justify-between py-2.5 px-3 rounded-xl bg-white/[0.02] hover:bg-white/[0.04] transition-colors"
                    >
                        <div class="flex items-center gap-3">
                            <span class="text-sm font-medium text-white/80"
                                >{task.symbol}</span
                            >
                            <span class="text-xs {statusColor(task.status)}"
                                >{statusLabel(task.status)}</span
                            >
                        </div>
                        <div
                            class="flex items-center gap-4 text-xs text-white/30"
                        >
                            {#if task.result?.training_time}
                                <span
                                    >耗时 {fmtTime(
                                        task.result.training_time,
                                    )}</span
                                >
                            {/if}
                            {#if task.result?.best_val_loss}
                                <span
                                    >损失 {task.result.best_val_loss.toFixed(
                                        6,
                                    )}</span
                                >
                            {/if}
                            {#if task.start_time}
                                <span
                                    >{new Date(task.start_time).toLocaleString(
                                        "zh-CN",
                                        {
                                            month: "2-digit",
                                            day: "2-digit",
                                            hour: "2-digit",
                                            minute: "2-digit",
                                        },
                                    )}</span
                                >
                            {/if}
                        </div>
                    </div>
                {/each}
            </div>
        </Card>
    {/if}
</div>
