<script>
    import "../layout.css";
    let { children } = $props();
</script>

<!-- 认证页面不使用侧边栏布局 -->
<div class="min-h-screen bg-surface-950 text-white">
    {@render children()}
</div>
