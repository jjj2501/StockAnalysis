<script>
    import { goto } from "$app/navigation";

    let username = $state("");
    let email = $state("");
    let password = $state("");
    let confirmPassword = $state("");
    let showPassword = $state(false);
    let loading = $state(false);
    let error = $state("");
    let success = $state("");

    async function handleRegister(/** @type {Event} */ e) {
        e.preventDefault();
        if (password !== confirmPassword) {
            error = "两次输入的密码不一致";
            return;
        }
        loading = true;
        error = "";
        try {
            const res = await fetch("http://localhost:8000/api/auth/register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, email, password }),
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
                throw new Error(msg || "注册失败");
            }
            success = "注册成功! 正在跳转到登录页...";
            setTimeout(() => goto("/auth/login"), 1500);
        } catch (err) {
            error = /** @type {Error} */ (err).message;
        } finally {
            loading = false;
        }
    }
</script>

<svelte:head>
    <title>注册 - AlphaPulse</title>
</svelte:head>

<div class="min-h-screen flex items-center justify-center p-4 -m-8">
    <div class="w-full max-w-md">
        <div class="text-center mb-10">
            <div
                class="inline-flex w-16 h-16 rounded-2xl bg-primary-600 items-center justify-center text-2xl font-bold mb-4 shadow-2xl shadow-primary-600/30"
            >
                S
            </div>
            <h1 class="text-2xl font-bold">创建账户</h1>
            <p class="text-white/40 text-sm mt-1">
                加入 AlphaPulse，开启智能投资分析之旅
            </p>
        </div>

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

            <form onsubmit={handleRegister} class="space-y-5">
                <div>
                    <label
                        class="block text-sm text-white/60 mb-1.5 font-medium"
                        >用户名</label
                    >
                    <input
                        type="text"
                        bind:value={username}
                        required
                        placeholder="请输入用户名"
                        class="w-full bg-white/5 border border-white/10 px-4 py-3 rounded-xl text-sm text-white placeholder:text-white/25 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/20 transition-all"
                    />
                </div>

                <div>
                    <label
                        class="block text-sm text-white/60 mb-1.5 font-medium"
                        >邮箱</label
                    >
                    <input
                        type="email"
                        bind:value={email}
                        required
                        placeholder="请输入邮箱地址"
                        class="w-full bg-white/5 border border-white/10 px-4 py-3 rounded-xl text-sm text-white placeholder:text-white/25 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/20 transition-all"
                    />
                </div>

                <div>
                    <label
                        class="block text-sm text-white/60 mb-1.5 font-medium"
                        >密码</label
                    >
                    <div class="relative">
                        <input
                            type={showPassword ? "text" : "password"}
                            bind:value={password}
                            required
                            placeholder="设置密码（至少 8 位）"
                            minlength="8"
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

                <div>
                    <label
                        class="block text-sm text-white/60 mb-1.5 font-medium"
                        >确认密码</label
                    >
                    <input
                        type="password"
                        bind:value={confirmPassword}
                        required
                        placeholder="再次输入密码"
                        class="w-full bg-white/5 border border-white/10 px-4 py-3 rounded-xl text-sm text-white placeholder:text-white/25 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/20 transition-all"
                    />
                </div>

                <button
                    type="submit"
                    disabled={loading}
                    class="w-full py-3 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 text-white font-medium rounded-xl transition-all shadow-lg shadow-primary-600/20"
                >
                    {loading ? "注册中..." : "创建账户"}
                </button>
            </form>

            <div class="mt-6 flex items-center gap-3">
                <div class="flex-1 h-px bg-white/5"></div>
                <span class="text-xs text-white/20">或者</span>
                <div class="flex-1 h-px bg-white/5"></div>
            </div>

            <div class="text-center mt-4 text-sm text-white/40">
                已有账号? <a
                    href="/auth/login"
                    class="text-primary-500 hover:text-primary-400 font-medium transition-colors"
                    >立即登录</a
                >
            </div>
        </div>
    </div>
</div>
