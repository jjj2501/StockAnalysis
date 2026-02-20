<script>
    import { goto } from "$app/navigation";

    let username = $state("");
    let password = $state("");
    let showPassword = $state(false);
    let remember = $state(false);
    let loading = $state(false);
    let error = $state("");
    let success = $state("");

    async function handleLogin(/** @type {Event} */ e) {
        e.preventDefault();
        loading = true;
        error = "";
        try {
            const res = await fetch("http://localhost:8000/api/auth/login", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: new URLSearchParams({ username, password }),
            });
            const data = await res.json();
            if (!res.ok) {
                const msg =
                    typeof data.detail === "string"
                        ? data.detail
                        : Array.isArray(data.detail)
                          ? data.detail
                                .map((/** @type {any} */ d) => d.msg)
                                .join(", ")
                          : JSON.stringify(data.detail);
                throw new Error(msg || "登录失败");
            }
            if (data.access_token) {
                localStorage.setItem("access_token", data.access_token);
                localStorage.setItem(
                    "user_info",
                    JSON.stringify(data.user || {}),
                );
                success = "登录成功，正在跳转...";
                setTimeout(() => goto("/"), 1000);
            }
        } catch (err) {
            error = /** @type {Error} */ (err).message;
        } finally {
            loading = false;
        }
    }
</script>

<svelte:head>
    <title>登录 - AlphaPulse</title>
</svelte:head>

<!-- 登录页面使用全屏布局，不走侧边栏 -->
<div class="min-h-screen flex items-center justify-center p-4 -m-8">
    <div class="w-full max-w-md">
        <!-- Logo -->
        <div class="text-center mb-10">
            <div
                class="inline-flex w-16 h-16 rounded-2xl bg-primary-600 items-center justify-center text-2xl font-bold mb-4 shadow-2xl shadow-primary-600/30"
            >
                S
            </div>
            <h1 class="text-2xl font-bold">AlphaPulse</h1>
            <p class="text-white/40 text-sm mt-1">智能投资分析平台</p>
        </div>

        <!-- 登录卡片 -->
        <div class="glass rounded-2xl p-8">
            {#if error}
                <div
                    class="bg-rose-500/10 border border-rose-500/20 text-rose-400 px-4 py-3 rounded-xl text-sm mb-6"
                >
                    ✕ {error}
                </div>
            {/if}
            {#if success}
                <div
                    class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl text-sm mb-6"
                >
                    ✓ {success}
                </div>
            {/if}

            <form onsubmit={handleLogin} class="space-y-5">
                <div>
                    <label
                        class="block text-sm text-white/60 mb-1.5 font-medium"
                        >用户名或邮箱</label
                    >
                    <input
                        type="text"
                        bind:value={username}
                        required
                        autocomplete="username"
                        placeholder="请输入用户名或邮箱"
                        class="w-full bg-white/5 border border-white/10 px-4 py-3 rounded-xl text-sm text-white placeholder:text-white/25 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/20 transition-all"
                    />
                </div>

                <div>
                    <div class="flex justify-between items-center mb-1.5">
                        <label class="text-sm text-white/60 font-medium"
                            >密码</label
                        >
                        <button
                            type="button"
                            class="text-xs text-primary-500 hover:text-primary-400 transition-colors"
                            >忘记密码?</button
                        >
                    </div>
                    <div class="relative">
                        <input
                            type={showPassword ? "text" : "password"}
                            bind:value={password}
                            required
                            autocomplete="current-password"
                            placeholder="请输入密码"
                            class="w-full bg-white/5 border border-white/10 px-4 py-3 rounded-xl text-sm text-white placeholder:text-white/25 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/20 transition-all pr-10"
                        />
                        <button
                            type="button"
                            onclick={() => (showPassword = !showPassword)}
                            class="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 transition-colors text-sm"
                        >
                            {showPassword ? "🙈" : "👁"}
                        </button>
                    </div>
                </div>

                <div class="flex items-center gap-2">
                    <input
                        type="checkbox"
                        bind:checked={remember}
                        id="remember"
                        class="rounded accent-primary-600"
                    />
                    <label for="remember" class="text-sm text-white/40"
                        >记住我</label
                    >
                </div>

                <button
                    type="submit"
                    disabled={loading}
                    class="w-full py-3 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 text-white font-medium rounded-xl transition-all shadow-lg shadow-primary-600/20"
                >
                    {loading ? "登录中..." : "登录"}
                </button>
            </form>

            <div class="mt-6 flex items-center gap-3">
                <div class="flex-1 h-px bg-white/5"></div>
                <span class="text-xs text-white/20">或者</span>
                <div class="flex-1 h-px bg-white/5"></div>
            </div>

            <div class="text-center mt-4 text-sm text-white/40">
                新用户? <a
                    href="/auth/register"
                    class="text-primary-500 hover:text-primary-400 font-medium transition-colors"
                    >创建账户</a
                >
            </div>
        </div>

        <div class="text-center mt-6 text-xs text-white/20 space-y-1">
            <p>基于 Transformers + LSTM 与 LLM 的智能选股助手</p>
            <p>支持多用户 · 数据隔离 · 本地部署</p>
        </div>
    </div>
</div>
