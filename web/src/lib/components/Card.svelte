<script>
    let { children, title = "", class: className = "" } = $props();
</script>

<div
    class="glass rounded-2xl p-6 transition-all duration-300 hover:border-white/10 overflow-hidden relative group {className}"
>
    {#if title}
        <div class="flex items-center justify-between mb-6">
            <h3 class="font-semibold text-white/90">{title}</h3>
            <div class="w-1.5 h-1.5 rounded-full bg-primary-500/50"></div>
        </div>
    {/if}

    {@render children()}

    <!-- 装饰性光晕 -->
    <div
        class="absolute -right-10 -top-10 w-32 h-32 bg-primary-600/5 rounded-full blur-3xl group-hover:bg-primary-600/10 transition-colors"
    ></div>
</div>
