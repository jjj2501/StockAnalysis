<script>
    import { page } from "$app/stores";

    const menuItems = [
        { name: "首页概览", path: "/", icon: "📊" },
        { name: "因子分析", path: "/factors", icon: "🔍" },
        { name: "量化回测", path: "/backtest", icon: "⚙️" },
        { name: "模型训练", path: "/training", icon: "🧠" },
        { name: "投资组合", path: "/portfolio", icon: "💼" },
        { name: "智能预警", path: "/watchlist", icon: "🔔" },
    ];

    const settingsItems = [
        { name: "数据管理", path: "/data", icon: "💽" },
        { name: "系统设置", path: "/settings", icon: "🛠️" },
        { name: "退出登录", path: "/auth/login", icon: "🚪" },
    ];

    // 当前活跃路由
    $: activePath = $page.url.pathname;
</script>

<aside
    class="w-64 border-r border-white/5 bg-surface-900/50 backdrop-blur-xl flex flex-col shrink-0"
>
    <div class="h-16 flex items-center px-6 border-b border-white/5">
        <img
            src="/logo.png"
            alt="AlphaPulse Logo"
            class="w-8 h-8 rounded-lg object-contain"
        />
        <span class="ml-3 font-bold text-lg tracking-tight">AlphaPulse</span>
    </div>

    <nav class="flex-1 py-6 px-4 space-y-1">
        {#each menuItems as item}
            <a
                href={item.path}
                class="flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group
                {$page.url.pathname === item.path
                    ? 'bg-primary-600/10 text-primary-500 font-medium border border-primary-500/20'
                    : 'text-white/60 hover:text-white hover:bg-white/5'}"
            >
                <span class="text-xl group-hover:scale-110 transition-transform"
                    >{item.icon}</span
                >
                <span class="text-sm">{item.name}</span>
                {#if $page.url.pathname === item.path}
                    <div
                        class="ml-auto w-1.5 h-1.5 rounded-full bg-primary-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]"
                    ></div>
                {/if}
            </a>
        {/each}
    </nav>

    <div class="p-4 border-t border-white/5 space-y-1">
        {#each settingsItems as item}
            <a
                href={item.path}
                class="flex items-center gap-3 px-3 py-2 rounded-xl text-white/50 hover:text-white hover:bg-white/5 transition-all text-sm"
            >
                <span>{item.icon}</span>
                <span>{item.name}</span>
            </a>
        {/each}
    </div>
</aside>
