"""
多智能体调度引擎 v2.0 (AgentOrchestrator)

核心升级:
  1. 多轮辩论共识引擎 (DebateEngine) — Portfolio Manager 可点名追加辩论
  2. 角色专属技能系统 (Skills Registry) — 每位 Agent 挂载其专属工具包
  3. 主动学习与知识积累 (AgentMemoryStore) — 推演前加载历史记忆，推演后持久化
"""

import json
import time
import logging

from backend.core.data import DataFetcher
from backend.core.llm import get_llm_client
from backend.core.agents.prompts import (
    MACRO_ANALYST_PROMPT,
    QUANT_RESEARCHER_PROMPT,
    RISK_CONTROL_PROMPT,
    PORTFOLIO_MANAGER_PROMPT
)
from backend.core.agents.base import ReActAgent
from backend.core.agents.debate import DebateEngine
from backend.core.agents.memory import AgentMemoryStore
from backend.core.agents.memory_fs import (
    write_session_to_daily, promote_to_long_term,
    read_recent_daily, read_long_term, read_global_memory
)
from backend.core.agents.context_loader import build_memory_injection_message
from backend.core.agents.skills import ROLE_SKILLS

logger = logging.getLogger(__name__)

# 追加辩论时，若 PM 初评包含这些词则触发追加辩论
_DEBATE_TRIGGER_PHRASES = [
    "存在严重分歧", "请", "重新论证", "需要补充", "值得重新核查",
    "数据矛盾", "追加调查", "请提供",
]


class AgentOrchestrator:
    def __init__(self, provider="ollama", model_name="qwen3:1.7b", max_debate_rounds: int = 2):
        self.fetcher = DataFetcher()
        self.llm = get_llm_client(provider=provider, model_name=model_name)
        self.max_debate_rounds = max_debate_rounds

        # 实例化各位专家 Agent，注入角色专属技能
        self.agents = [
            ReActAgent(
                name="Macro Analyst",
                role_prompt=MACRO_ANALYST_PROMPT,
                llm=self.llm,
                max_turns=3,
                extra_skills=ROLE_SKILLS.get("Macro Analyst", [])
            ),
            ReActAgent(
                name="Quant Researcher",
                role_prompt=QUANT_RESEARCHER_PROMPT,
                llm=self.llm,
                max_turns=3,
                extra_skills=ROLE_SKILLS.get("Quant Researcher", [])
            ),
            ReActAgent(
                name="Risk Control Agent",
                role_prompt=RISK_CONTROL_PROMPT,
                llm=self.llm,
                max_turns=3,
                extra_skills=ROLE_SKILLS.get("Risk Control Agent", [])
            ),
        ]

    def run_safari(self, symbol: str):
        """
        全链路流式 SSE 生成器。
        推演结构:
          Phase 0: 加载历史记忆并注入上下文
          Phase 1: 数据工程师 — 提取全维因子底牌
          Phase 2: 第 1 轮分析师接力辩论
          Phase 3: Portfolio Manager 初评 → 如有分歧，触发追加辩论（最多 2 轮）
          Phase 4: Portfolio Manager 最终裁定 + 知识提炼写库
        """
        # ─── SSE 打包辅助函数 ───
        def _sse(role, status, content, is_chunk=False, raw_data=None, extra=None):
            payload = {"role": role, "status": status, "content": content, "is_chunk": is_chunk}
            if raw_data:
                payload["raw_data"] = raw_data
            if extra:
                payload.update(extra)
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        session_id = str(int(time.time()))
        debate = DebateEngine(max_debate_rounds=self.max_debate_rounds)

        # ==========================================
        # Phase 0: 加载历史记忆（OpenClaw 分级加载）
        # ==========================================
        # 分级加载：常驻 = 长期 memory.md + 近两日短期 .md + 全局规律
        memory_msg = build_memory_injection_message(symbol)
        long_term_text = read_long_term(symbol)
        recent_text    = read_recent_daily(symbol, days=2)
        global_text    = read_global_memory()

        has_memory = bool(long_term_text or recent_text)
        if has_memory:
            yield _sse(
                "Data Engineer", "typing",
                f"📂 **检索历史记忆（Markdown 档案）...** "
                f"发现 {symbol} 长期规律 + 近期推演摘要，正在注入参考上下文。\n\n",
                is_chunk=True
            )

        # ==========================================
        # Phase 1: 数据工程师 — 提取因子底牌
        # ==========================================
        yield _sse("Data Engineer", "typing", f"正在构建 **{symbol}** 全维市场雷达探针矩阵...\n\n")

        try:
            factors = self.fetcher.get_comprehensive_factors(symbol)
        except Exception as e:
            logger.error(f"Data Engineer 抓取崩溃: {e}")
            yield _sse("Data Engineer", "error", f"❌ 致命错误：获取 {symbol} 因子矩阵失败 ({e})")
            return

        yield _sse("Data Engineer", "done", f"✅ 成功提取 **{symbol}** 全维面板。[{len(factors)} 个维度] 交由投研委员会研讨。\n", raw_data=factors)

        # 中央议事大厅记忆体（注入因子底牌 + Markdown 记忆上下文）
        context_messages = []

        # 若有历史记忆，作为首条 user message 注入（兼容 OpenAI message 列表格式）
        if memory_msg:
            context_messages.append(memory_msg)

        context_messages.append({
            "role": "user",
            "content": (
                f"本次研讨标的: **{symbol}**。\n"
                f"数据工程师初采底层面板（供所有成员参考）:\n{json.dumps(factors, ensure_ascii=False)}"
            )
        })

        # ==========================================
        # Phase 2: 第 1 轮 — 分析师接力发言
        # ==========================================
        yield _sse(
            "Portfolio Manager", "typing",
            f"🔔 **第 {debate.current_round + 1} 轮圆桌辩论开始** — 请各位专家依次发表研判。\n\n",
            extra={"event": "round_start", "round": debate.current_round + 1}
        )

        agent_speeches = {}

        for agent in self.agents:
            yield _sse(agent.name, "typing", "\n", extra={"skills": [s["function"]["name"] for s in agent.extra_skills]})

            final_content = ""
            for step in agent.run(context_messages, symbol):
                stype = step.get("type")
                scontent = step.get("content", "")

                if stype == "thought":
                    yield _sse(agent.name, "typing", scontent + "\n\n", is_chunk=True)
                    final_content += scontent + "\n\n"
                elif stype == "action":
                    tool = step.get("tool")
                    args_str = json.dumps(step.get("args", {}), ensure_ascii=False)
                    msg = f"> ⚡ **调用专属技能**: `{tool}({args_str})`\n\n"
                    yield _sse(agent.name, "typing", msg, is_chunk=True)
                    final_content += msg
                elif stype == "observation":
                    msg = f"> 📡 **探针响应**: `{scontent}`\n\n"
                    yield _sse(agent.name, "typing", msg, is_chunk=True)
                    final_content += msg
                elif stype == "final_answer":
                    yield _sse(agent.name, "typing", scontent + "\n", is_chunk=True)
                    final_content += scontent + "\n"
                elif stype == "error":
                    yield _sse(agent.name, "error", scontent, is_chunk=True)
                elif stype == "system_warning":
                    yield _sse(agent.name, "typing", f"\n[系统插断]: *{scontent}*\n", is_chunk=True)

            yield _sse(agent.name, "done", "")

            # 写入公共黑板
            debate.record_speech(agent.name, final_content)
            agent_speeches[agent.name] = final_content
            context_messages.append({"role": "assistant", "name": agent.name, "content": final_content})

        # ==========================================
        # Phase 3: Portfolio Manager 初评 + 追加辩论
        # ==========================================
        debate_round = 0
        while debate_round <= self.max_debate_rounds:
            # 检测是否有分歧（第 1 次也要检测）
            if debate_round > 0 and not debate.should_trigger_debate():
                break

            pm_action = "初步评估" if debate_round == 0 else f"追加辩论 第 {debate_round} 轮"
            
            if debate_round > 0:
                # 宣告进入追加辩论轮
                new_round = debate.start_next_round()
                yield _sse(
                    "Portfolio Manager", "typing",
                    f"\n\n⚔️ **第 {new_round + 1} 轮追加辩论** — 发现显著分歧，就以下问题追加核查：\n\n",
                    extra={"event": "debate_round", "round": new_round}
                )
                # 让各方 Agent 再次回应
                for agent in self.agents:
                    yield _sse(agent.name, "typing", "\n", extra={"skills": [s["function"]["name"] for s in agent.extra_skills]})
                    extra_content = ""
                    debate_prompt = debate.get_board_summary()
                    extra_msgs = context_messages + [
                        {"role": "user", "content": f"Portfolio Manager 要求你就以下黑板内容补充或反驳：\n{debate_prompt[-1000:]}"}
                    ]
                    for step in agent.run(extra_msgs, symbol):
                        stype = step.get("type")
                        scontent = step.get("content", "")
                        if stype in ("thought", "final_answer"):
                            yield _sse(agent.name, "typing", scontent + "\n", is_chunk=True)
                            extra_content += scontent + "\n"
                        elif stype == "action":
                            tool = step.get("tool")
                            yield _sse(agent.name, "typing", f"> ⚡ **追加调阅**: `{tool}`\n\n", is_chunk=True)
                        elif stype == "observation":
                            yield _sse(agent.name, "typing", f"> 📡 `{scontent}`\n\n", is_chunk=True)
                            extra_content += scontent + "\n"
                    yield _sse(agent.name, "done", "")
                    debate.record_speech(agent.name, extra_content)
                    context_messages.append({"role": "assistant", "name": agent.name, "content": extra_content})

            # PM 做（初评或追加轮次后的）评审
            yield _sse("Portfolio Manager", "typing", f"📋 **{pm_action}** — 综合各方观点...\n\n")

            boss_context = debate.get_board_summary()
            pm_eval_prompt = (
                f"你是 Portfolio Manager，对标的 {symbol} 进行{pm_action}。\n"
                f"以下是所有与会专家在公共黑板上的发言记录：\n\n{boss_context}\n\n"
                f"请判断：当前台上是否存在严重分歧？若存在，请明确点出是哪两方的哪条观点产生了矛盾，并说明需要哪个专家追加何种数据来化解分歧。"
                if debate_round == 0 else
                f"综合{pm_action}后的黑板记录，给出更新后的共识评估。"
            )

            pm_interim = ""
            try:
                for chunk in self.llm.stream_generate(pm_eval_prompt):
                    yield _sse("Portfolio Manager", "typing", chunk, is_chunk=True)
                    pm_interim += chunk
                yield _sse("Portfolio Manager", "done", "")
            except Exception as e:
                yield _sse("Portfolio Manager", "error", f"初评失败: {e}")
                break

            # 写入黑板
            debate.record_speech("Portfolio Manager", pm_interim)
            context_messages.append({"role": "assistant", "name": "Portfolio Manager", "content": pm_interim})

            # 判断是否还应该继续辩论
            if debate_round >= self.max_debate_rounds:
                break
            if not debate.should_trigger_debate():
                break

            debate_round += 1

        # ==========================================
        # Phase 4: 最终裁定 + 知识提炼
        # ==========================================
        yield _sse("Portfolio Manager", "typing", "\n\n## 📜 **最终裁定书** — 全体投票归票完毕\n\n")

        all_debate_recap = debate.get_board_summary()
        pm_final_prompt = PORTFOLIO_MANAGER_PROMPT.format(
            base_data=f"资产代码: {symbol}",
            macro_opinion=all_debate_recap,
            quant_opinion="",
            risk_opinion=""
        )

        final_verdict = ""
        try:
            for chunk in self.llm.stream_generate(pm_final_prompt):
                yield _sse("Portfolio Manager", "typing", chunk, is_chunk=True)
                final_verdict += chunk
        except Exception as e:
            yield _sse("Portfolio Manager", "error", f"最终裁定生成失败: {e}")

        yield _sse("Portfolio Manager", "done", "\n\n---\n✅ **全体智能体全链路推演完毕**\n")

        # ── OpenClaw 记忆蒸馏：将本次推演压缩写入 Markdown ──
        consensus = debate.get_final_consensus()
        yield _sse("Portfolio Manager", "typing",
                   "\n\n🧠 **正在蒸馏推演记忆并写入 Markdown 档案...**\n\n", is_chunk=True)

        # 1. 生成结构化摘要（用于短期记忆）
        distill_prompt = (
            f"请将以下关于 {symbol} 的多智能体投研讨论，"
            f"压缩为 3-5 条 Markdown 要点（每条以 - 开头）。"
            f"如果发现有值得长期记录的跨标的规律，请在最末以 '【长期洞见】:' 开头单独列出一条。\n\n"
            f"辩论摘要：\n{debate.get_board_summary()[-1500:]}\n"
            f"最终裁定：\n{final_verdict[:500]}"
        )
        distill_text = ""
        try:
            for chunk in self.llm.stream_generate(distill_prompt):
                distill_text += chunk
        except Exception:
            distill_text = f"最终裁定：{final_verdict[:300]}"

        # 2. 写入短期记忆（今日 .md 文件）
        try:
            write_session_to_daily(symbol, distill_text)
        except Exception as e:
            logger.warning(f"[OpenClaw] 写入短期记忆失败: {e}")

        # 3. 若包含长期洞见，提炼并写入 memory.md
        if "【长期洞见】" in distill_text:
            long_term_candidate = distill_text.split("【长期洞见】:")[-1].strip().split("\n")[0]
            if long_term_candidate:
                try:
                    promote_to_long_term(symbol, long_term_candidate)
                    promote_to_long_term("GLOBAL", long_term_candidate, is_global=True)
                    yield _sse(
                        "Portfolio Manager", "typing",
                        f"> 💡 **新长期洞见已写入 memory.md**: {long_term_candidate[:80]}\n\n",
                        is_chunk=True
                    )
                except Exception as e:
                    logger.warning(f"[OpenClaw] 写入长期记忆失败: {e}")

        # 4. 兼容性写入旧 JSON 存储（保持 API /api/agents/memory 接口可用）
        try:
            AgentMemoryStore.save_session(
                symbol=symbol,
                session_id=session_id,
                debate_history=consensus["history"],
                final_verdict=final_verdict,
                consensus=consensus["consensus"]
            )
        except Exception as mem_err:
            logger.error(f"[记忆持久化] JSON 备份保存失败: {mem_err}")

        # 5. 刷新 BM25 索引（让刚写入的 Markdown 记忆立即可被下次推演检索到）
        try:
            from backend.core.agents.rag_retriever import refresh_index
            refresh_index()
        except Exception:
            pass  # 索引刷新不影响主流程

        # 6. L1 Prompt 自进化（推演后自动优化 Agent 角色 Prompt 并 Git 提交）
        try:
            from backend.core.agents.evolution.prompt_evolver import evolve_prompt
            score = consensus["consensus"].get("score", 50)
            board_summary = debate.get_board_summary()

            # 选择一个 Agent 进行进化（简单策略：轮流选择，避免偏向）
            agent_names = ["Macro Analyst", "Quant Researcher", "Risk Control Agent"]
            target_agent = agent_names[int(session_id) % len(agent_names)]

            yield _sse(
                "Portfolio Manager", "typing",
                f"\n\n🧬 **启动 L1 自进化引擎** — 正在为 {target_agent} 提炼经验...\n\n",
                is_chunk=True
            )

            evo_result = evolve_prompt(
                llm=self.llm,
                agent_name=target_agent,
                symbol=symbol,
                debate_summary=board_summary[-1000:],
                verdict=final_verdict[:500],
                consensus_score=score
            )

            if evo_result["success"]:
                insight = evo_result["insight"]
                git_msg = evo_result.get("git_result", {}).get("message", "")
                branch = evo_result.get("git_result", {}).get("branch", "")
                yield _sse(
                    "Portfolio Manager", "typing",
                    f"> 🧬 **{target_agent} 进化完成**: {insight}\n"
                    f"> 📦 Git: {git_msg}\n\n",
                    is_chunk=True
                )
            else:
                yield _sse(
                    "Portfolio Manager", "typing",
                    f"> ⏭️ 本轮进化跳过: {evo_result.get('insight', '条件不足')}\n\n",
                    is_chunk=True
                )
        except Exception as evo_err:
            import traceback
            tb = traceback.format_exc()
            logger.warning(f"[L1进化] 进化失败（不影响主流程）: {evo_err}\n{tb}")
            yield _sse(
                "Portfolio Manager", "typing",
                f"\n\n> ⚠️ **L1 自进化引擎异常**: {evo_err}\n\n",
                is_chunk=True
            )

        yield _sse(
            "Portfolio Manager", "done",
            f"📊 共识评分: {consensus['consensus']['score']}/100 ({consensus['consensus']['verdict']})"
        )
