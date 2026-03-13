<script>
    import { onMount } from "svelte";
    import Card from "$lib/components/Card.svelte";

    // 自动检测的设备信息
    /** @type {any} */
    let gpuStatus = $state(null);
    /** @type {any} */
    let config = $state(null);
    let loading = $state(true);
    let saving = $state(false);
    let saveMsg = $state("");
    let testingConnection = $state(false);
    let testMsg = $state("");

    // 可编辑参数
    let batchSize = $state(32);
    let epochs = $state(50);
    let learningRate = $state(0.001);
    let hiddenDim = $state(64);
    let numLayers = $state(2);
    let seqLength = $state(60);

    // AI 模型配置
    let llmProvider = $state("ollama"); // 设置页这里的下拉框可以暂定为 default 毕竟如果真的用外部 API 这里用户会重选
    let modelName = $state("qwen3:1.7b");
    let llmApiKey = $state("");
    let llmBaseUrl = $state("");

    onMount(async () => {
        // 从浏览器本地恢复 LLM 设置偏好
        llmProvider = localStorage.getItem("llmProvider") || "ollama";
        modelName = localStorage.getItem("modelName") || "qwen3:1.7b";

        await loadDeviceInfo();
    });

    async function loadDeviceInfo() {
        loading = true;
        try {
            const [statusRes, configRes, llmRes] = await Promise.all([
                fetch("/api/gpu/status"),
                fetch("/api/gpu/config/current"),
                fetch("/api/config/llm"),
            ]);
            gpuStatus = await statusRes.json();
            config = await configRes.json();
            const llmConfig = await llmRes.json();

            // 用后端真实配置填充表单
            if (config) {
                batchSize = config.batch_size ?? 32;
                epochs = config.epochs ?? 50;
                learningRate = config.learning_rate ?? 0.001;
                hiddenDim = config.model_hidden_dim ?? 64;
                numLayers = config.model_num_layers ?? 2;
                seqLength = config.sequence_length ?? 60;
            }
            if (llmConfig) {
                llmApiKey = llmConfig.api_key || "";
                llmBaseUrl = llmConfig.base_url || "";
                // 优先使用后端持久化的设置，而不是仅靠 localStorage
                if (llmConfig.provider) llmProvider = llmConfig.provider;
                if (llmConfig.model) modelName = llmConfig.model;
            }
        } catch (/** @type {any} */ e) {
            console.error("加载设备信息失败:", e);
        } finally {
            loading = false;
        }
    }

    async function testConnection() {
        testingConnection = true;
        testMsg = "⏳ 测试中...";
        try {
            const res = await fetch("/api/config/llm/test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    provider: llmProvider,
                    model_name: modelName,
                    api_key: llmApiKey,
                    base_url: llmBaseUrl,
                }),
            });
            const data = await res.json();
            if (res.ok && data.status === "success") {
                testMsg = "✅ " + data.message;
            } else {
                testMsg = "❌ " + (data.message || "测试失败");
            }
        } catch (/** @type {any} */ e) {
            testMsg = "❌ 网络错误: " + e.message;
        } finally {
            testingConnection = false;
        }
    }

    async function saveConfig() {
        saving = true;
        saveMsg = "";

        // 保存前端本地偏好
        localStorage.setItem("llmProvider", llmProvider);
        localStorage.setItem("modelName", modelName);

        try {
            const [res, llmRes] = await Promise.all([
                fetch("/api/gpu/config/update", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        batch_size: batchSize,
                        epochs: epochs,
                        learning_rate: learningRate,
                        model_hidden_dim: hiddenDim,
                        model_num_layers: numLayers,
                        sequence_length: seqLength,
                    }),
                }),
                fetch("/api/config/llm", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        api_key: llmApiKey,
                        base_url: llmBaseUrl,
                        provider: llmProvider,
                        model: modelName,
                    }),
                }),
            ]);
            if (res.ok && llmRes.ok) {
                saveMsg = "✅ 配置已保存成功";
            } else {
                saveMsg = "❌ 保存失败";
            }
        } catch (/** @type {any} */ e) {
            saveMsg = "❌ 网络错误: " + e.message;
        } finally {
            saving = false;
            setTimeout(() => (saveMsg = ""), 3000);
        }
    }

    // 设备类型和颜色
    function deviceLabel(/** @type {string} */ device) {
        if (device?.includes("cuda"))
            return {
                text: "NVIDIA GPU (CUDA)",
                color: "text-emerald-400",
                dot: "bg-emerald-400",
            };
        if (device?.includes("mps"))
            return {
                text: "Apple Silicon (MPS)",
                color: "text-purple-400",
                dot: "bg-purple-400",
            };
        return {
            text: "CPU 模式",
            color: "text-amber-400",
            dot: "bg-amber-400",
        };
    }

    function fmtMem(/** @type {number} */ mb) {
        return mb >= 1024
            ? (mb / 1024).toFixed(2) + " GB"
            : mb.toFixed(1) + " MB";
    }
</script>

<svelte:head>
    <title>系统设置 - AlphaPulse</title>
</svelte:head>

<div class="space-y-8">
    <div>
        <h2 class="text-2xl font-bold">系统设置</h2>
        <p class="text-white/40 mt-1">管理账户、AI模型与计算资源</p>
    </div>

    <!-- 个人信息 -->
    <Card title="👤 个人信息">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div>
                <label class="block text-xs text-white/40 mb-1.5 font-medium"
                    >用户名</label
                >
                <input
                    type="text"
                    value="admin"
                    disabled
                    class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white/50"
                />
            </div>
            <div>
                <label class="block text-xs text-white/40 mb-1.5 font-medium"
                    >邮箱</label
                >
                <input
                    type="email"
                    value="admin@example.com"
                    class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all"
                />
            </div>
        </div>
    </Card>

    <!-- AI 模型配置 -->
    <Card title="🤖 AI 模型配置">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div>
                <label class="block text-xs text-white/40 mb-1.5 font-medium"
                    >LLM 提供商</label
                >
                <select
                    bind:value={llmProvider}
                    class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all appearance-none"
                >
                    <option value="ollama" class="bg-surface-900"
                        >Ollama (本地)</option
                    >
                    <option value="openai" class="bg-surface-900"
                        >OpenAI 兼容 (例如 DeepSeek 等外部 API)</option
                    >
                </select>
            </div>
            <div>
                <label class="block text-xs text-white/40 mb-1.5 font-medium"
                    >模型名称</label
                >
                <input
                    type="text"
                    bind:value={modelName}
                    class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all"
                />
            </div>
            {#if llmProvider === "openai"}
                <div
                    class="col-span-1 md:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-5 mt-2 p-4 bg-emerald-500/5 rounded-xl border border-emerald-500/20"
                >
                    <div>
                        <label
                            class="block text-xs text-emerald-400 mb-1.5 font-medium flex items-center gap-1"
                        >
                            <span>🔑</span> OpenAI API Key (密钥)
                        </label>
                        <input
                            type="password"
                            bind:value={llmApiKey}
                            placeholder="sk-..."
                            class="w-full bg-[#0a0c10] border border-emerald-500/30 rounded-xl px-3 py-2.5 text-sm text-white placeholder:text-white/20 focus:outline-none focus:border-emerald-500/80 transition-all"
                        />
                    </div>
                    <div>
                        <label
                            class="block text-xs text-emerald-400 mb-1.5 font-medium flex items-center gap-1"
                        >
                            <span>🌐</span> Base URL (中转站/DeepSeek)
                        </label>
                        <input
                            type="text"
                            bind:value={llmBaseUrl}
                            placeholder="例如: https://api.deepseek.com/v1"
                            class="w-full bg-[#0a0c10] border border-emerald-500/30 rounded-xl px-3 py-2.5 text-sm text-white placeholder:text-white/20 focus:outline-none focus:border-emerald-500/80 transition-all"
                        />
                    </div>
                </div>
            {/if}
        </div>

        <!-- 测试连接区块 -->
        <div
            class="mt-6 flex flex-col sm:flex-row items-center justify-between border-t border-white/5 pt-5"
        >
            <div
                class="text-sm {testMsg.includes('✅')
                    ? 'text-emerald-400'
                    : testMsg.includes('❌')
                      ? 'text-rose-400'
                      : 'text-white/40'}"
            >
                {testMsg || "在保存前可以先测试模型服务是否连通"}
            </div>
            <button
                onclick={testConnection}
                disabled={testingConnection || loading}
                class="mt-3 sm:mt-0 px-4 py-2 bg-white/5 hover:bg-white/10 disabled:opacity-50 border border-white/10 rounded-xl text-sm font-medium text-white transition-all focus:outline-none"
            >
                {testingConnection ? "🔌 连接中..." : "⚡ 测试连接"}
            </button>
        </div>
    </Card>

    <!-- 计算资源 - 自动检测 -->
    <Card title="🖥️ 计算资源（自动检测）">
        {#if loading}
            <div class="flex items-center gap-3 py-4">
                <div
                    class="w-5 h-5 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin"
                ></div>
                <span class="text-sm text-white/40">正在检测硬件环境...</span>
            </div>
        {:else if gpuStatus}
            <!-- 设备状态卡片 -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <!-- 设备类型 -->
                <div
                    class="bg-white/[0.03] border border-white/5 rounded-xl p-4"
                >
                    <div class="text-xs text-white/30 mb-2">训练设备</div>
                    <div class="flex items-center gap-2">
                        <span
                            class="w-2.5 h-2.5 rounded-full {deviceLabel(
                                gpuStatus.device,
                            ).dot} shadow-lg"
                        ></span>
                        <span
                            class="text-sm font-semibold {deviceLabel(
                                gpuStatus.device,
                            ).color}"
                        >
                            {deviceLabel(gpuStatus.device).text}
                        </span>
                    </div>
                    <div class="text-xs text-white/20 mt-1">
                        {gpuStatus.device}
                    </div>
                </div>

                <!-- GPU 信息 -->
                <div
                    class="bg-white/[0.03] border border-white/5 rounded-xl p-4"
                >
                    <div class="text-xs text-white/30 mb-2">
                        {gpuStatus.gpu_available ? "GPU 型号" : "硬件状态"}
                    </div>
                    {#if gpuStatus.gpu_available && gpuStatus.gpu_info?.device_name}
                        <div class="text-sm font-semibold text-white/90">
                            {gpuStatus.gpu_info.device_name}
                        </div>
                        <div class="text-xs text-white/20 mt-1">
                            CUDA {gpuStatus.gpu_info.cuda_version}
                        </div>
                    {:else}
                        <div class="text-sm text-white/50">CUDA 不可用</div>
                        <div class="text-xs text-white/20 mt-1">
                            未检测到 NVIDIA GPU
                        </div>
                    {/if}
                </div>

                <!-- 显存 -->
                <div
                    class="bg-white/[0.03] border border-white/5 rounded-xl p-4"
                >
                    <div class="text-xs text-white/30 mb-2">显存</div>
                    {#if gpuStatus.gpu_available && gpuStatus.gpu_info?.memory_total}
                        <div class="text-sm font-semibold text-white/90">
                            {gpuStatus.gpu_info.memory_total.toFixed(1)} GB
                        </div>
                        <div class="text-xs text-white/20 mt-1">
                            已用 {fmtMem(
                                gpuStatus.gpu_info.memory_allocated ?? 0,
                            )}
                        </div>
                        <!-- 显存使用条 -->
                        <div
                            class="mt-2 h-1.5 bg-white/5 rounded-full overflow-hidden"
                        >
                            <div
                                class="h-full bg-emerald-500/70 rounded-full transition-all"
                                style="width: {Math.min(
                                    ((gpuStatus.gpu_info.memory_allocated ??
                                        0) /
                                        (gpuStatus.gpu_info.memory_total *
                                            1024)) *
                                        100,
                                    100,
                                )}%"
                            ></div>
                        </div>
                    {:else}
                        <div class="text-sm text-white/50">—</div>
                        <div class="text-xs text-white/20 mt-1">
                            使用系统内存
                        </div>
                    {/if}
                </div>
            </div>

            <!-- 训练参数 -->
            <div class="border-t border-white/5 pt-5">
                <div class="text-xs text-white/40 font-medium mb-4">
                    训练超参数
                </div>
                <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div>
                        <label class="block text-xs text-white/30 mb-1.5"
                            >批大小 (Batch Size)</label
                        >
                        <input
                            type="number"
                            bind:value={batchSize}
                            min="1"
                            class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all"
                        />
                    </div>
                    <div>
                        <label class="block text-xs text-white/30 mb-1.5"
                            >训练轮数 (Epochs)</label
                        >
                        <input
                            type="number"
                            bind:value={epochs}
                            min="1"
                            class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all"
                        />
                    </div>
                    <div>
                        <label class="block text-xs text-white/30 mb-1.5"
                            >学习率 (LR)</label
                        >
                        <input
                            type="number"
                            bind:value={learningRate}
                            step="0.0001"
                            min="0.00001"
                            class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all"
                        />
                    </div>
                    <div>
                        <label class="block text-xs text-white/30 mb-1.5"
                            >隐藏层维度</label
                        >
                        <input
                            type="number"
                            bind:value={hiddenDim}
                            min="16"
                            class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all"
                        />
                    </div>
                    <div>
                        <label class="block text-xs text-white/30 mb-1.5"
                            >网络层数</label
                        >
                        <input
                            type="number"
                            bind:value={numLayers}
                            min="1"
                            max="8"
                            class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all"
                        />
                    </div>
                    <div>
                        <label class="block text-xs text-white/30 mb-1.5"
                            >序列长度</label
                        >
                        <input
                            type="number"
                            bind:value={seqLength}
                            min="10"
                            class="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500/50 transition-all"
                        />
                    </div>
                </div>
            </div>
        {:else}
            <div class="text-center py-6 text-white/30 text-sm">
                ⚠️ 无法连接后端服务，请确认服务已启动
            </div>
        {/if}
    </Card>

    <!-- 保存按钮 -->
    <div class="flex items-center justify-end gap-4">
        {#if saveMsg}
            <span
                class="text-sm {saveMsg.includes('✅')
                    ? 'text-emerald-400'
                    : 'text-rose-400'}">{saveMsg}</span
            >
        {/if}
        <button
            onclick={saveConfig}
            disabled={saving || loading}
            class="px-6 py-3 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 text-white font-medium rounded-xl transition-all shadow-lg shadow-primary-600/20"
        >
            {saving ? "⏳ 保存中..." : "💾 保存更改"}
        </button>
    </div>
</div>
