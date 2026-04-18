from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from treehouse_lab.runtime_settings import load_llm_settings

try:
    import requests
except ImportError:  # pragma: no cover - optional at runtime
    requests = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional at runtime
    OpenAI = None

DEFAULT_ADVISOR_PROVIDER = "ollama"
DEFAULT_ADVISOR_QUESTION = "What should I do next and why?"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_KEY_ENV_VARS = ("OLLAMA_API_KEY", "OLLAMA_CLOUD_KEY", "VIOLAAMA_CLOUD_KEY")
DEFAULT_AGENT_CLI = "codex"
DEFAULT_OPENAI_COMPATIBLE_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


@dataclass(slots=True)
class TextGenerationResult:
    status: str
    provider: str
    model: str | None
    text: str | None = None
    message: str | None = None


@dataclass(slots=True)
class AdvisorResponse:
    status: str
    provider: str
    model: str | None
    question: str
    answer: str | None = None
    message: str | None = None
    grounding: dict[str, Any] = field(default_factory=dict)
    recommended_proposal: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProposalSelectionResponse:
    status: str
    provider: str
    model: str | None
    selected_proposal_id: str | None
    rationale: str | None = None
    message: str | None = None
    raw_output: str | None = None
    candidate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_research_advice(context: dict[str, Any], question: str | None = None) -> AdvisorResponse:
    question_text = (question or DEFAULT_ADVISOR_QUESTION).strip() or DEFAULT_ADVISOR_QUESTION
    grounding = _build_grounding(context)
    result = _generate_text(
        system_prompt=_developer_prompt(),
        user_prompt=_user_prompt(context, question_text),
        context=context,
        purpose="research coach",
    )
    return AdvisorResponse(
        status=result.status,
        provider=result.provider,
        model=result.model,
        question=question_text,
        answer=result.text if result.status == "available" else None,
        message=result.message,
        grounding=grounding,
    )


def llm_loop_selection_enabled(project_root: str | Path | None = None) -> bool:
    if project_root is None:
        project_root = Path.cwd()
    project_root = Path(project_root)
    setting = load_llm_settings(project_root).get("loop_llm_selection", False)
    if isinstance(setting, bool):
        return setting
    return _truthy(str(setting))


def select_bounded_proposal(context: dict[str, Any], candidates: list[dict[str, Any]]) -> ProposalSelectionResponse:
    project_root = _project_root_from_context(context)
    if not candidates:
        return ProposalSelectionResponse(
            status="unavailable",
            provider=_active_provider(project_root),
            model=None,
            selected_proposal_id=None,
            message="No candidates were available for LLM review.",
            candidate_count=0,
        )

    result = _generate_text(
        system_prompt=_proposal_selection_system_prompt(),
        user_prompt=_proposal_selection_user_prompt(context, candidates),
        context=context,
        purpose="bounded proposal selection",
    )
    if result.status != "available" or not result.text:
        return ProposalSelectionResponse(
            status=result.status,
            provider=result.provider,
            model=result.model,
            selected_proposal_id=None,
            message=result.message,
            raw_output=result.text,
            candidate_count=len(candidates),
        )

    payload = _extract_json_object(result.text)
    candidate_ids = {str(candidate["proposal_id"]) for candidate in candidates}
    selected_proposal_id = None if payload is None else payload.get("selected_proposal_id")
    if not isinstance(selected_proposal_id, str) or selected_proposal_id not in candidate_ids:
        return ProposalSelectionResponse(
            status="error",
            provider=result.provider,
            model=result.model,
            selected_proposal_id=None,
            message="LLM selection did not return a valid candidate proposal_id.",
            raw_output=result.text,
            candidate_count=len(candidates),
        )

    rationale = payload.get("rationale")
    return ProposalSelectionResponse(
        status="available",
        provider=result.provider,
        model=result.model,
        selected_proposal_id=selected_proposal_id,
        rationale=None if rationale is None else str(rationale),
        raw_output=result.text,
        candidate_count=len(candidates),
    )


def _generate_text(system_prompt: str, user_prompt: str, context: dict[str, Any], purpose: str) -> TextGenerationResult:
    project_root = _project_root_from_context(context)
    provider = _active_provider(project_root)
    if provider == "ollama":
        return _request_text_via_ollama(system_prompt, user_prompt, purpose, project_root)
    if provider == "agent_cli":
        return _request_text_via_agent_cli(system_prompt, user_prompt, purpose, context, project_root)
    if provider == "openai_compatible":
        return _request_text_via_openai_compatible(system_prompt, user_prompt, purpose, project_root)
    if provider == "openai":
        return _request_text_via_openai(system_prompt, user_prompt, purpose, project_root)
    return TextGenerationResult(
        status="unavailable",
        provider=provider,
        model=None,
        message=(
            f"Unsupported LLM provider: {provider}. "
            "Supported providers are `ollama`, `agent_cli`, `openai_compatible`, and `openai`."
        ),
    )


def _request_text_via_ollama(system_prompt: str, user_prompt: str, purpose: str, project_root: str) -> TextGenerationResult:
    model = _setting("model", project_root, DEFAULT_OLLAMA_MODEL) or DEFAULT_OLLAMA_MODEL
    base_url = _setting("ollama_base_url", project_root, DEFAULT_OLLAMA_BASE_URL) or DEFAULT_OLLAMA_BASE_URL
    endpoint = f"{base_url.rstrip('/')}/api/chat"

    if requests is None:
        return TextGenerationResult(
            status="unavailable",
            provider="ollama",
            model=model,
            message="The optional `requests` dependency is not installed. Install the `llm` extra to enable Ollama-backed requests.",
        )

    headers = {"Content-Type": "application/json"}
    if _is_ollama_cloud(base_url):
        api_key = _setting("ollama_api_key", project_root)
        if not api_key:
            return TextGenerationResult(
                status="unavailable",
                provider="ollama",
                model=model,
                message=(
                    f"Set one of {', '.join(OLLAMA_API_KEY_ENV_VARS)} to use Ollama cloud directly for {purpose}, "
                    "or point TREEHOUSE_LAB_OLLAMA_BASE_URL at a local Ollama host."
                ),
            )
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
    except Exception as exc:
        return TextGenerationResult(
            status="error",
            provider="ollama",
            model=model,
            message=f"Ollama request failed during {purpose}: {exc}",
        )

    text = _extract_ollama_text(response.json())
    if not text:
        return TextGenerationResult(
            status="error",
            provider="ollama",
            model=model,
            message=f"Ollama returned no assistant text during {purpose}.",
        )
    return TextGenerationResult(status="available", provider="ollama", model=model, text=text)


def _request_text_via_agent_cli(
    system_prompt: str,
    user_prompt: str,
    purpose: str,
    context: dict[str, Any],
    project_root: str,
) -> TextGenerationResult:
    cli_name = (_setting("agent_cli", project_root, DEFAULT_AGENT_CLI) or DEFAULT_AGENT_CLI).lower()
    model = _setting("model", project_root) or None
    project_root = _project_root_from_context(context)

    if shutil.which(cli_name) is None:
        return TextGenerationResult(
            status="unavailable",
            provider=f"agent_cli:{cli_name}",
            model=model,
            message=f"The configured agent CLI `{cli_name}` is not installed or not on PATH.",
        )

    if cli_name == "codex":
        return _run_codex_cli(project_root, system_prompt, user_prompt, model=model, purpose=purpose)
    if cli_name == "claude":
        return _run_claude_cli(project_root, system_prompt, user_prompt, model=model, purpose=purpose)
    return TextGenerationResult(
        status="unavailable",
        provider=f"agent_cli:{cli_name}",
        model=model,
        message="Unsupported agent CLI. Supported values are `codex` and `claude`.",
    )


def _request_text_via_openai_compatible(
    system_prompt: str,
    user_prompt: str,
    purpose: str,
    project_root: str,
) -> TextGenerationResult:
    model = _setting("model", project_root, DEFAULT_OPENAI_COMPATIBLE_MODEL) or DEFAULT_OPENAI_COMPATIBLE_MODEL
    base_url = _setting("openai_compatible_base_url", project_root)

    if OpenAI is None:
        return TextGenerationResult(
            status="unavailable",
            provider="openai_compatible",
            model=model,
            message="The optional OpenAI SDK is not installed. Install the `llm` extra to enable OpenAI-compatible backends.",
        )
    if not base_url:
        return TextGenerationResult(
            status="unavailable",
            provider="openai_compatible",
            model=model,
            message="Set TREEHOUSE_LAB_OPENAI_COMPATIBLE_BASE_URL to use an OpenAI-compatible API.",
        )

    api_key = _setting("openai_compatible_api_key", project_root)
    if not api_key and not _is_local_base_url(base_url):
        return TextGenerationResult(
            status="unavailable",
            provider="openai_compatible",
            model=model,
            message=(
                "Set TREEHOUSE_LAB_OPENAI_COMPATIBLE_API_KEY for remote OpenAI-compatible APIs, "
                "or point the base URL at a local host."
            ),
        )

    try:
        client = OpenAI(base_url=base_url, api_key=api_key or "openai-compatible")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        return TextGenerationResult(
            status="error",
            provider="openai_compatible",
            model=model,
            message=f"OpenAI-compatible request failed during {purpose}: {exc}",
        )

    text = _extract_chat_completions_text(response)
    if not text:
        return TextGenerationResult(
            status="error",
            provider="openai_compatible",
            model=model,
            message=f"The OpenAI-compatible API returned no text output during {purpose}.",
        )
    return TextGenerationResult(status="available", provider="openai_compatible", model=model, text=text)


def _request_text_via_openai(system_prompt: str, user_prompt: str, purpose: str, project_root: str) -> TextGenerationResult:
    model = _setting("model", project_root, DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL

    if OpenAI is None:
        return TextGenerationResult(
            status="unavailable",
            provider="openai",
            model=model,
            message="The optional OpenAI SDK is not installed. Install the `llm` extra to enable the OpenAI-backed provider.",
        )
    api_key = _setting("openai_api_key", project_root)
    if not api_key:
        return TextGenerationResult(
            status="unavailable",
            provider="openai",
            model=model,
            message="Set OPENAI_API_KEY to use the OpenAI-backed provider.",
        )

    try:
        try:
            client = OpenAI(api_key=api_key)
        except TypeError:
            client = OpenAI()
        response = client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        return TextGenerationResult(
            status="error",
            provider="openai",
            model=model,
            message=f"OpenAI request failed during {purpose}: {exc}",
        )

    text = _extract_openai_text(response)
    if not text:
        return TextGenerationResult(
            status="error",
            provider="openai",
            model=model,
            message=f"The OpenAI provider returned no text output during {purpose}.",
        )
    return TextGenerationResult(status="available", provider="openai", model=model, text=text)


def _run_codex_cli(project_root: str, system_prompt: str, user_prompt: str, model: str | None, purpose: str) -> TextGenerationResult:
    prompt = f"{system_prompt}\n\n{user_prompt}"
    with tempfile.TemporaryDirectory(prefix="treehouse-lab-codex-") as temp_dir:
        output_path = Path(temp_dir) / "last_message.txt"
        cmd = [
            "codex",
            "exec",
            "-C",
            project_root,
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--output-last-message",
            str(output_path),
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.append(prompt)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
        except Exception as exc:
            return TextGenerationResult(
                status="error",
                provider="agent_cli:codex",
                model=model,
                message=f"Codex CLI request failed during {purpose}: {exc}",
            )

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            return TextGenerationResult(
                status="error",
                provider="agent_cli:codex",
                model=model,
                message=f"Codex CLI exited with code {result.returncode}: {detail}",
            )

        text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else result.stdout.strip()
        if not text:
            return TextGenerationResult(
                status="error",
                provider="agent_cli:codex",
                model=model,
                message=f"Codex CLI returned no final message during {purpose}.",
            )
        return TextGenerationResult(status="available", provider="agent_cli:codex", model=model, text=text)


def _run_claude_cli(project_root: str, system_prompt: str, user_prompt: str, model: str | None, purpose: str) -> TextGenerationResult:
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "text",
        "--permission-mode",
        "plan",
        "--tools",
        "",
        "--bare",
        "--add-dir",
        project_root,
        "--system-prompt",
        system_prompt,
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append(user_prompt)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
    except Exception as exc:
        return TextGenerationResult(
            status="error",
            provider="agent_cli:claude",
            model=model,
            message=f"Claude Code request failed during {purpose}: {exc}",
        )

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return TextGenerationResult(
            status="error",
            provider="agent_cli:claude",
            model=model,
            message=f"Claude Code exited with code {result.returncode}: {detail}",
        )

    text = result.stdout.strip()
    if not text:
        return TextGenerationResult(
            status="error",
            provider="agent_cli:claude",
            model=model,
            message=f"Claude Code returned no text output during {purpose}.",
        )
    return TextGenerationResult(status="available", provider="agent_cli:claude", model=model, text=text)


def _developer_prompt() -> str:
    return (
        "You are the Treehouse Lab research coach.\n"
        "Explain the current state of a tabular ML experiment using only the supplied context.\n"
        "Rules:\n"
        "- Never recommend actions outside the declared mutation templates or search space.\n"
        "- Never suggest touching the held-out test set or silently changing split policy.\n"
        "- Ground every recommendation in the provided metrics, diagnosis, proposal, and recent journal history.\n"
        "- If the recent history shows repeated rejected mutation families or diminishing deltas, say so plainly.\n"
        "- Keep the answer concise and auditable.\n"
        "- Use exactly these sections and labels:\n"
        "Current state:\n"
        "Next step:\n"
        "Watchouts:"
    )


def _proposal_selection_system_prompt() -> str:
    return (
        "You are selecting the next bounded Treehouse Lab experiment.\n"
        "Choose exactly one candidate proposal from the supplied list.\n"
        "Rules:\n"
        "- Only select from the candidate proposal_ids provided.\n"
        "- Never invent a new mutation, parameter, or feature-generation step.\n"
        "- Prefer small, attributable changes that fit the diagnosis and recent history.\n"
        "- If recent runs repeated a family without beating the promotion threshold, say that clearly.\n"
        "- Output strict JSON only, no markdown fences.\n"
        'Format: {"selected_proposal_id":"<candidate id>","rationale":"<short explanation>"}'
    )


def _user_prompt(context: dict[str, Any], question: str) -> str:
    return f"Question: {question}\n\nContext JSON:\n{json.dumps(context, indent=2, sort_keys=True)}"


def _proposal_selection_user_prompt(context: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    payload = {
        "dataset_key": context.get("dataset_key"),
        "incumbent": context.get("incumbent"),
        "diagnosis": context.get("diagnosis"),
        "promote_threshold": context.get("promote_threshold"),
        "recent_entries": context.get("recent_entries", []),
        "candidates": candidates,
    }
    return f"Select the next bounded proposal.\n\nCandidate Context JSON:\n{json.dumps(payload, indent=2, sort_keys=True)}"


def _build_grounding(context: dict[str, Any]) -> dict[str, Any]:
    recent_entries = list(context.get("recent_entries", []))
    recent_mutations = [entry.get("name") for entry in recent_entries if entry.get("name")]
    return {
        "dataset_key": context.get("dataset_key"),
        "journal_count": context.get("journal_count", len(recent_entries)),
        "recent_mutations": recent_mutations,
        "diagnosis_tag": context.get("diagnosis_preview", {}).get("diagnosis", {}).get("primary_tag")
        or context.get("diagnosis", {}).get("primary_tag"),
    }


def _project_root_from_context(context: dict[str, Any]) -> str:
    project_root = context.get("project_root")
    if project_root:
        return str(project_root)
    config_path = context.get("config", {}).get("path")
    if config_path:
        return str(Path(str(config_path)).resolve().parents[2])
    return os.getcwd()


def _active_provider(project_root: str) -> str:
    return (_setting("provider", project_root, DEFAULT_ADVISOR_PROVIDER) or DEFAULT_ADVISOR_PROVIDER).lower()


def _setting(name: str, project_root: str, default: str = "") -> str:
    settings = load_llm_settings(Path(project_root))
    setting_value = str(settings.get(name, "")).strip()
    if setting_value:
        return setting_value

    env_map: dict[str, str | tuple[str, ...]] = {
        "provider": "TREEHOUSE_LAB_LLM_PROVIDER",
        "model": "TREEHOUSE_LAB_LLM_MODEL",
        "ollama_base_url": "TREEHOUSE_LAB_OLLAMA_BASE_URL",
        "ollama_api_key": OLLAMA_API_KEY_ENV_VARS,
        "agent_cli": "TREEHOUSE_LAB_AGENT_CLI",
        "openai_compatible_base_url": "TREEHOUSE_LAB_OPENAI_COMPATIBLE_BASE_URL",
        "openai_compatible_api_key": "TREEHOUSE_LAB_OPENAI_COMPATIBLE_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
    }
    env_name = env_map.get(name)
    if isinstance(env_name, tuple):
        env_value = _resolve_first_env(env_name)
    elif isinstance(env_name, str):
        env_value = os.getenv(env_name, "").strip()
    else:
        env_value = ""
    if env_value:
        return env_value
    return default.strip()


def _extract_openai_text(response: Any) -> str:
    direct_text = getattr(response, "output_text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    payload: dict[str, Any] | None = None
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif isinstance(response, dict):
        payload = response

    if not payload:
        return ""

    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if content.get("type") in {"output_text", "text"} and isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n\n".join(chunks).strip()


def _extract_chat_completions_text(response: Any) -> str:
    payload: dict[str, Any] | None = None
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif isinstance(response, dict):
        payload = response

    if not payload:
        return ""

    choices = payload.get("choices", [])
    if not choices:
        return ""

    content = choices[0].get("message", {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item.get("text", "").strip() for item in content if isinstance(item, dict) and item.get("text")]
        return "\n\n".join(part for part in parts if part).strip()
    return ""


def _extract_ollama_text(payload: dict[str, Any]) -> str:
    message = payload.get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    return ""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()

    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _is_ollama_cloud(base_url: str) -> bool:
    normalized = base_url.rstrip("/").lower()
    return normalized == "https://ollama.com"


def _is_local_base_url(base_url: str) -> bool:
    normalized = base_url.rstrip("/").lower()
    return normalized.startswith("http://localhost") or normalized.startswith("http://127.0.0.1")


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_first_env(env_names: tuple[str, ...]) -> str:
    for env_name in env_names:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return env_value
    return ""
