<script>
    import { onMount } from "svelte";
    import Card from "$lib/components/Card.svelte";

    let files = [];
    let loading = true;
    let actionLoading = false;
    let errorMsg = "";

    // 手工上传状态
    let fileInput;
    let uploadSymbol = "";
    let uploadMarket = "CN";
    let showUploadModal = false;

    async function loadCache() {
        loading = true;
        errorMsg = "";
        try {
            const res = await fetch("/api/data/cache");
            if (!res.ok) throw new Error("加载缓存列表失败");
            files = await res.json();
        } catch (err) {
            errorMsg = err.message || "加载缓存列表失败";
        } finally {
            loading = false;
        }
    }

    async function handleDelete(filename) {
        if (!confirm(`确定要删除缓存文件 ${filename} 吗？`)) return;

        actionLoading = true;
        try {
            const res = await fetch(`/api/data/cache/${filename}`, {
                method: "DELETE",
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "删除失败");
            }
            await loadCache();
        } catch (err) {
            alert(`删除失败: ${err.message}`);
        } finally {
            actionLoading = false;
        }
    }

    async function handleForceFetch(symbol, market) {
        if (
            !confirm(
                `确定要强制重新拉取并在本地覆盖 ${symbol} 的数据吗？单次请求可能需要 5-15 秒。`,
            )
        )
            return;

        actionLoading = true;
        try {
            const res = await fetch("/api/data/fetch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ symbol, market }),
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "强制更新失败");
            }
            const data = await res.json();
            alert(`更新成功！共拉取 ${data.rows} 条最新数据。`);
            await loadCache();
        } catch (err) {
            alert(`强制更新失败: ${err.message}`);
        } finally {
            actionLoading = false;
        }
    }

    async function handleUploadSubmit() {
        if (!fileInput.files.length) {
            alert("请选择要上传的 CSV 文件");
            return;
        }
        if (!uploadSymbol.trim()) {
            alert("请输入要覆盖的标的代码");
            return;
        }

        actionLoading = true;
        try {
            const formData = new FormData();
            formData.append("symbol", uploadSymbol);
            formData.append("market", uploadMarket);
            formData.append("file", fileInput.files[0]);

            // 因为使用了 FormData，不能用普通的 application/json apiFetch
            // 我们直接用原生 fetch
            const token = localStorage.getItem("token");
            const res = await fetch("/api/data/upload", {
                method: "POST",
                headers: token
                    ? {
                          Authorization: `Bearer ${token}`,
                      }
                    : {},
                body: formData,
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "上传处理失败");
            }

            const data = await res.json();
            alert(`上传合并成功！共覆盖写入 ${data.rows} 条数据。`);
            showUploadModal = false;
            await loadCache();
        } catch (err) {
            alert(`上传覆写失败: ${err.message}`);
        } finally {
            actionLoading = false;
        }
    }

    onMount(() => {
        loadCache();
    });

    const marketConfig = {
        CN: {
            label: "A股",
            class: "bg-red-500/10 text-red-500 border-red-500/20",
        },
        US: {
            label: "美股",
            class: "bg-blue-500/10 text-blue-500 border-blue-500/20",
        },
        HK: {
            label: "港股",
            class: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20",
        },
        CRYPTO: {
            label: "数字资产",
            class: "bg-purple-500/10 text-purple-500 border-purple-500/20",
        },
    };
</script>

{#snippet headerActions()}
    <div class="flex items-center gap-3">
        <button
            on:click={() => {
                uploadSymbol = "";
                showUploadModal = true;
            }}
            class="px-4 py-2 bg-primary-600/20 text-primary-400 hover:bg-primary-600/30 hover:text-white rounded-xl transition-all font-medium text-sm flex items-center gap-2 border border-primary-500/20"
        >
            <span>📥 手工账单覆盖</span>
        </button>
        <button
            on:click={loadCache}
            disabled={loading || actionLoading}
            class="px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-xl transition-all font-medium text-sm flex items-center gap-2 border border-white/5 disabled:opacity-50"
        >
            <span class={loading ? "animate-spin" : ""}>🔄</span>
            <span>刷新列表</span>
        </button>
    </div>
{/snippet}

<div class="space-y-6 animate-in">
    <!-- 头部信息 -->
    <div class="flex flex-col gap-2">
        <h1 class="text-3xl font-bold tracking-tight">
            AlphaPulse 数据库中心 / Data Center
        </h1>
        <p class="text-white/60">
            底层的资产历史缓存都在这里。本地零延迟加载依赖这些数据。你可以直接对异常的脏数据进行接管、强制洗盘清洗，或者手工喂食
            CSV。
        </p>
    </div>

    <!-- 列表展示区 -->
    <Card
        title="Parquet 离线缓存池 ({files.length} 个资产)"
        header={headerActions}
    >
        {#if errorMsg}
            <div
                class="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm mb-4"
            >
                {errorMsg}
            </div>
        {/if}

        {#if loading && files.length === 0}
            <div
                class="py-12 flex flex-col items-center justify-center text-white/40"
            >
                <div
                    class="w-8 h-8 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin mb-4"
                ></div>
                <p>扫描庞大的底层资产库...</p>
            </div>
        {:else if files.length === 0}
            <div
                class="py-12 text-center text-white/40 border border-dashed border-white/10 rounded-xl bg-white/5"
            >
                <p class="text-4xl mb-3">📭</p>
                <p>缓存池深不可测，但是现在底空空如也。</p>
                <p class="text-sm mt-2">前往因子分析或尝试强制刷新一个标的！</p>
            </div>
        {:else}
            <div class="overflow-x-auto rounded-xl border border-white/5">
                <table class="w-full text-left text-sm whitespace-nowrap">
                    <thead class="bg-white/5 text-white/60">
                        <tr>
                            <th class="px-6 py-4 font-medium">市场</th>
                            <th class="px-6 py-4 font-medium">标的代码</th>
                            <th class="px-6 py-4 font-medium">物理体积</th>
                            <th class="px-6 py-4 font-medium"
                                >最后同步/洗盘时间</th
                            >
                            <th class="px-6 py-4 font-medium text-right"
                                >人工干预</th
                            >
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-white/5">
                        {#each files as file (file.filename)}
                            <tr class="hover:bg-white/[0.02] transition-colors">
                                <td class="px-6 py-4">
                                    {#if marketConfig[file.market]}
                                        <span
                                            class="px-2.5 py-1 rounded-md text-xs font-medium border {marketConfig[
                                                file.market
                                            ].class}"
                                        >
                                            {marketConfig[file.market].label}
                                        </span>
                                    {:else}
                                        <span
                                            class="px-2.5 py-1 rounded-md text-xs font-medium border bg-white/10 text-white/70 border-white/20"
                                        >
                                            {file.market}
                                        </span>
                                    {/if}
                                </td>
                                <td class="px-6 py-4">
                                    <span
                                        class="font-bold text-white tracking-widest"
                                        >{file.symbol}</span
                                    >
                                </td>
                                <td class="px-6 py-4 text-white/70">
                                    {file.size_kb} KB
                                </td>
                                <td class="px-6 py-4 text-white/50">
                                    {file.modified_time}
                                </td>
                                <td class="px-6 py-4 text-right">
                                    <div
                                        class="flex items-center justify-end gap-2"
                                    >
                                        <button
                                            on:click={() =>
                                                handleForceFetch(
                                                    file.symbol,
                                                    file.market,
                                                )}
                                            disabled={actionLoading}
                                            class="p-2 bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 rounded-lg transition-colors group"
                                            title="强制重新从外部拉取并清洗数据"
                                        >
                                            <span
                                                class="group-hover:rotate-180 transition-transform duration-500 inline-block"
                                                >🔄</span
                                            >
                                        </button>
                                        <button
                                            on:click={() =>
                                                handleDelete(file.filename)}
                                            disabled={actionLoading}
                                            class="p-2 bg-red-500/10 text-red-400 hover:bg-red-500/20 hover:text-red-300 rounded-lg transition-colors"
                                            title="删库扔弃（强制核销）"
                                        >
                                            🗑️
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        {/if}
    </Card>
</div>

<!-- 手工上传模态框 -->
{#if showUploadModal}
    <div
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4"
    >
        <div
            class="bg-surface-800 border border-white/10 rounded-2xl w-full max-w-md shadow-2xl animate-in slide-in-from-bottom-4 relative"
        >
            <div
                class="p-6 border-b border-white/5 flex justify-between items-center"
            >
                <h3 class="text-xl font-bold">📥 手工覆盖底层数据黑盒</h3>
                <button
                    class="text-white/40 hover:text-white"
                    on:click={() => (showUploadModal = false)}>✕</button
                >
            </div>

            <div class="p-6 space-y-5">
                <div class="space-y-2">
                    <label class="block text-sm font-medium text-white/70"
                        >资产代码 (Symbol)</label
                    >
                    <input
                        type="text"
                        bind:value={uploadSymbol}
                        placeholder="例如: AAPL 或 600519"
                        class="w-full bg-black/30 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all"
                    />
                </div>

                <div class="space-y-2">
                    <label class="block text-sm font-medium text-white/70"
                        >所属市场 (Market Segment)</label
                    >
                    <select
                        bind:value={uploadMarket}
                        class="w-full bg-black/30 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all appearance-none"
                    >
                        <option value="CN">🇨🇳 大陆 A 股 (CN)</option>
                        <option value="US">🇺🇸 美股 (US)</option>
                        <option value="HK">🇭🇰 港股 (HK)</option>
                        <option value="CRYPTO">🪙 加密货币 (CRYPTO)</option>
                    </select>
                </div>

                <div class="space-y-2 pt-2">
                    <label class="block text-sm font-medium text-white/70"
                        >干净的 CSV 药方文件</label
                    >
                    <div
                        class="relative border-2 border-dashed border-white/20 rounded-xl p-8 hover:bg-white/5 hover:border-primary-500/50 transition-colors text-center group"
                    >
                        <input
                            type="file"
                            accept=".csv"
                            bind:this={fileInput}
                            class="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        />
                        <span
                            class="text-3xl mb-2 inline-block group-hover:-translate-y-1 transition-transform"
                            >📄</span
                        >
                        <p class="text-sm text-white/60">
                            点击或拖拽您的 .csv 洗盘文件<br />(需含
                            <b>date, open, close, high, low, volume</b>)
                        </p>
                    </div>
                </div>
            </div>

            <div class="p-6 border-t border-white/5 flex gap-3">
                <button
                    on:click={() => (showUploadModal = false)}
                    class="flex-1 py-3 px-4 rounded-xl border border-white/10 text-white hover:bg-white/5 transition-colors font-medium"
                >
                    取消
                </button>
                <button
                    on:click={handleUploadSubmit}
                    disabled={actionLoading}
                    class="flex-1 py-3 px-4 rounded-xl bg-primary-600 text-white hover:bg-primary-500 transition-colors font-medium shadow-lg shadow-primary-500/20 disabled:opacity-70 flex items-center justify-center gap-2"
                >
                    {#if actionLoading}
                        <span
                            class="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"
                        ></span>
                        <span>覆写注入中...</span>
                    {:else}
                        <span>注入主缓冲池并覆盖</span>
                    {/if}
                </button>
            </div>
        </div>
    </div>
{/if}
