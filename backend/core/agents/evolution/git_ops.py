"""
安全 Git 操作封装 (GitOps)

安全规则:
1. 分支名强制前缀 agent-evolution/
2. 只允许 add 白名单文件
3. 不允许修改 evolution/ 目录自身
4. commit message 带有 🧬 标记
"""

import os
import logging
import subprocess
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# 项目根目录
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

# 安全白名单：只允许 Agent 修改这些文件
ALLOWED_FILES = [
    "backend/core/agents/prompts.py",
]

# 绝对禁止修改的路径
FORBIDDEN_PATHS = [
    "backend/core/agents/evolution",
    "backend/config.py",
    ".env",
    "backend/main.py",
]

# 分支前缀
BRANCH_PREFIX = "agent-evolution"


def _run_git(args: list, cwd: str = None) -> tuple:
    """
    执行 Git 命令并返回 (成功?, 输出文本)。
    """
    cmd = ["git"] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or _PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8"
        )
        output = result.stdout.strip() + result.stderr.strip()
        return result.returncode == 0, output
    except Exception as e:
        return False, str(e)


def _validate_file_path(relative_path: str) -> bool:
    """检查文件路径是否在白名单内且不在禁止列表中"""
    normalized = relative_path.replace("\\", "/")

    # 检查禁止列表
    for forbidden in FORBIDDEN_PATHS:
        if normalized.startswith(forbidden):
            logger.error(f"[GitOps][安全] 拒绝修改禁止路径: {normalized}")
            return False

    # 检查白名单
    if normalized not in ALLOWED_FILES:
        logger.error(f"[GitOps][安全] 拒绝修改非白名单文件: {normalized}")
        return False

    return True


def get_current_branch() -> str:
    """获取当前 Git 分支名"""
    ok, output = _run_git(["branch", "--show-current"])
    return output if ok else "unknown"


def create_evolution_branch(description: str) -> Optional[str]:
    """
    创建 agent-evolution/ 前缀的新分支。

    Args:
        description: 进化描述（用于分支名后缀，如 "macro-analyst-tune"）

    Returns:
        分支名，失败返回 None
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    # 清理分支名（去掉不合法字符）
    safe_desc = "".join(c if c.isalnum() or c == "-" else "-" for c in description)[:30]
    branch_name = f"{BRANCH_PREFIX}/{safe_desc}-{timestamp}"

    # 直接 checkout -b，Git 会自动把工作区中未暂存的修改带到新分支
    # 注意：不能先 stash，否则 prompts.py 的修改会被暂存掉，后面 commit 时没有变更
    ok, output = _run_git(["checkout", "-b", branch_name])
    if not ok:
        logger.error(f"[GitOps] 创建分支失败: {output}")
        return None

    logger.info(f"[GitOps] 已创建进化分支: {branch_name}")
    return branch_name


def commit_evolution(file_path: str, agent_name: str, insight: str) -> bool:
    """
    将指定文件的修改提交到当前分支。

    Args:
        file_path: 相对于项目根目录的文件路径
        agent_name: 进化的 Agent 角色名
        insight: 进化的经验摘要（用于 commit message）

    Returns:
        是否成功
    """
    if not _validate_file_path(file_path):
        return False

    # git add
    ok, output = _run_git(["add", file_path])
    if not ok:
        logger.error(f"[GitOps] git add 失败: {output}")
        return False

    # git commit
    short_insight = insight[:50].replace('"', "'")
    commit_msg = f'🧬 agent-evolution: {agent_name} 新增经验 "{short_insight}"'
    ok, output = _run_git(["commit", "-m", commit_msg])
    if not ok:
        logger.error(f"[GitOps] git commit 失败: {output}")
        return False

    logger.info(f"[GitOps] 已提交进化: {commit_msg}")
    return True


def push_evolution(branch_name: str) -> bool:
    """
    推送进化分支到远程仓库。

    安全检查: 分支名必须以 agent-evolution/ 开头
    """
    if not branch_name.startswith(BRANCH_PREFIX):
        logger.error(f"[GitOps][安全] 拒绝推送非进化分支: {branch_name}")
        return False

    ok, output = _run_git(["push", "origin", branch_name])
    if not ok:
        logger.warning(f"[GitOps] push 失败（可能是网络或权限问题）: {output}")
        return False

    logger.info(f"[GitOps] 已推送到远程: origin/{branch_name}")
    return True


def switch_back_to_dev() -> bool:
    """切回 dev 分支"""
    ok, _ = _run_git(["checkout", "dev"])
    return ok


def full_evolution_cycle(file_path: str, agent_name: str, insight: str) -> dict:
    """
    完整的进化 Git 工作流：创建分支 → commit → push → 切回 dev。

    Returns:
        {"success": bool, "branch": str, "message": str}
    """
    result = {"success": False, "branch": "", "message": ""}

    # 1. 创建进化分支
    branch = create_evolution_branch(agent_name.lower().replace(" ", "-"))
    if not branch:
        result["message"] = "创建进化分支失败"
        return result

    result["branch"] = branch

    # 2. 提交
    if not commit_evolution(file_path, agent_name, insight):
        result["message"] = "提交进化失败"
        switch_back_to_dev()
        return result

    # 3. 推送
    pushed = push_evolution(branch)
    result["message"] = f"已推送到 origin/{branch}" if pushed else "本地提交成功，但推送失败（已保留在本地分支）"

    # 4. 切回 dev 并将经验 cherry-pick 回来（让经验在工作分支上也生效）
    switch_back_to_dev()

    # 尝试将进化分支上的经验提交 cherry-pick 回来
    ok, cherry_output = _run_git(["cherry-pick", branch, "--no-commit"])
    if ok:
        # cherry-pick 成功，直接 reset（不提交，保留文件修改在工作区）
        _run_git(["reset", "HEAD"])
        logger.info(f"[GitOps] 已将经验 cherry-pick 回 dev 工作区")
    else:
        # cherry-pick 冲突或失败，回退 cherry-pick 并手动重新注入经验
        _run_git(["cherry-pick", "--abort"])
        logger.warning(f"[GitOps] cherry-pick 失败，经验仅存在于进化分支: {cherry_output}")

    result["success"] = True
    return result
