<script>
    let { children } = $props();
    import "./layout.css";
    import { page } from "$app/stores";
    import Sidebar from "$lib/components/Sidebar.svelte";

    // 路由 → 页面标题映射
    const pageTitles = {
        "/": "首页概览",
        "/factors": "因子分析",
        "/backtest": "量化回测",
        "/training": "模型训练",
        "/portfolio": "投资组合",
        "/watchlist": "智能预警",
        "/settings": "系统设置",
        "/agents": "多智能体作战室",
        "/data": "底层数据中心",
    };

    // 根据当前路径获取页面标题
    /** @type {Record<string, string>} */
    const _titles = pageTitles;
    let currentTitle = $derived(_titles[$page.url.pathname] || "控制台");

    // 判断是否为认证页面（不显示侧边栏和顶栏）
    let isAuthPage = $derived($page.url.pathname.startsWith("/auth"));
</script>

{#if isAuthPage}
    <!-- 认证页面使用独立全屏布局 -->
    <div class="min-h-screen bg-surface-950 text-white">
        {@render children()}
    </div>
{:else}
    <div
        class="flex min-h-screen bg-surface-950 text-white selection:bg-primary-500/30"
    >
        <Sidebar />

        <main class="flex-1 flex flex-col min-w-0 overflow-hidden">
            <header
                class="h-16 flex items-center justify-between px-8 border-b border-white/5 glass sticky top-0 z-10"
            >
                <div class="flex items-center gap-3">
                    <h1 class="text-lg font-semibold text-white/90">
                        {currentTitle}
                    </h1>
                </div>

                <div class="flex items-center gap-6">
                    <div
                        class="flex items-center gap-3 bg-white/5 px-3 py-1.5 rounded-full border border-white/5"
                    >
                        <span
                            class="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"
                        ></span>
                        <span class="text-xs font-medium text-white/70"
                            >引擎在线</span
                        >
                    </div>

                    <button
                        class="w-8 h-8 rounded-full bg-primary-600 flex items-center justify-center text-sm font-bold shadow-lg shadow-primary-600/20 hover:scale-105 transition-transform"
                    >
                        B
                    </button>
                </div>
            </header>

            <section class="flex-1 overflow-y-auto p-4 md:p-8">
                <div class="max-w-7xl mx-auto">
                    {@render children()}
                </div>
            </section>
        </main>
    </div>
{/if}
