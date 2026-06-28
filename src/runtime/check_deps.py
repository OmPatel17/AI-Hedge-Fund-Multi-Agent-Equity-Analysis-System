from __future__ import annotations


def _check_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def check_llm_deps() -> None:
    missing: list[str] = []

    required_modules = [
        "langchain",
        "langchain_core",
        "langchain_openai",
        "langchain_anthropic",
        "langgraph",
        "anthropic",
        "openai",
        "questionary",
    ]

    optional_provider_modules = [
        "langchain_deepseek",
        "langchain_google_genai",
        "langchain_groq",
        "langchain_gigachat",
        "langchain_ollama",
        "langchain_xai",
    ]

    for module_name in required_modules:
        if not _check_import(module_name):
            missing.append(module_name)

    # Optional providers — warn but do not block startup.
    missing_optional = [module_name for module_name in optional_provider_modules if not _check_import(module_name)]
    if missing_optional:
        import logging
        logging.getLogger(__name__).warning(
            "Optional LLM providers not installed (safe to ignore if unused): %s",
            sorted(missing_optional),
        )

    if missing:
        raise RuntimeError(f"Missing LLM dependencies: {sorted(set(missing))}")
