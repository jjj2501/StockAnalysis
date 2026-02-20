<script>
    let { children } = $props();
    import "./layout.css";
    import Sidebar from "$lib/components/Sidebar.svelte";
</script>

<div
    class="flex min-h-screen bg-surface-950 text-white selection:bg-primary-500/30"
>
    <Sidebar />

    <main class="flex-1 flex flex-col min-w-0 overflow-hidden">
        <header
            class="h-16 flex items-center justify-between px-8 border-b border-white/5 glass sticky top-0 z-10"
        >
            <div class="flex items-center gap-4">
                <h1 class="text-lg font-semibold gradient-text">AlphaPulse</h1>
            </div>

            <div class="flex items-center gap-6">
                <!-- 顶部工具栏占位 -->
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
