from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_HF_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_HF_PROVIDER = "deepinfra"
DEFAULT_MAX_TOKENS = 1600
DEFAULT_TEMPERATURE = 0.25
DEFAULT_TOP_P = 0.95


@dataclass
class HFConfig:
    token: str
    model: str = DEFAULT_HF_MODEL
    provider: str = DEFAULT_HF_PROVIDER
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    timeout: float = 90.0


def parse_model_provider(value: str | None) -> tuple[str, str]:
    """Support both HF style split config and shorthand like
    `google/gemma-4-26B-A4B-it:deepinfra`.

    A Hugging Face model id itself contains one slash but not a provider suffix.
    When the final colon suffix exists, treat it as provider name.
    """
    raw = (value or "").strip()
    if not raw:
        return DEFAULT_HF_MODEL, DEFAULT_HF_PROVIDER

    # Windows paths are not expected here. Keep parsing strict to model:provider.
    m = re.match(r"^([^\s:]+/[^\s:]+):([A-Za-z0-9_.-]+)$", raw)
    if m:
        return m.group(1), m.group(2)
    return raw, DEFAULT_HF_PROVIDER


def get_hf_config(
    token: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    timeout: float | None = None,
) -> HFConfig:
    env_model = os.getenv("HF_MODEL", "").strip()
    model_from_env, provider_from_model = parse_model_provider(model or env_model or DEFAULT_HF_MODEL)
    provider_final = (provider or os.getenv("HF_PROVIDER") or provider_from_model or DEFAULT_HF_PROVIDER).strip()

    return HFConfig(
        token=(token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip(),
        model=model_from_env,
        provider=provider_final,
        max_tokens=int(max_tokens or os.getenv("HF_MAX_TOKENS") or DEFAULT_MAX_TOKENS),
        temperature=float(temperature or os.getenv("HF_TEMPERATURE") or DEFAULT_TEMPERATURE),
        top_p=float(top_p or os.getenv("HF_TOP_P") or DEFAULT_TOP_P),
        timeout=float(timeout or os.getenv("HF_TIMEOUT") or 90.0),
    )


def _load_client(config: HFConfig):
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise RuntimeError("huggingface_hub가 설치되어 있지 않습니다. `pip install -r requirements.txt`를 다시 실행하세요.") from exc

    if not config.token:
        raise ValueError("HF_TOKEN이 없습니다. .env, Streamlit Secrets, 또는 사이드바 입력란에 HF_TOKEN을 넣어 주세요.")

    # token and api_key are both accepted by recent huggingface_hub versions.
    # token is kept for compatibility with the official HF examples.
    return InferenceClient(provider=config.provider, token=config.token, timeout=config.timeout)


def build_voc_messages(query: str, similar_rows: Iterable[dict[str, Any]], base_draft: str) -> list[dict[str, str]]:
    examples_text_parts: list[str] = []
    for i, row in enumerate(similar_rows, start=1):
        title = str(row.get("title", ""))[:300]
        question = str(row.get("question_text", row.get("question", "")))[:1200]
        answer = str(row.get("answer", ""))[:1600]
        author = str(row.get("reply_author", "담당자"))[:80]
        similarity = row.get("similarity", "")
        examples_text_parts.append(
            f"[유사사례 {i}]\n"
            f"- 유사도: {similarity}\n"
            f"- 제목: {title}\n"
            f"- VOC: {question}\n"
            f"- 기존 담당자: {author}\n"
            f"- 기존 답변: {answer}"
        )

    examples_text = "\n\n".join(examples_text_parts) if examples_text_parts else "유사 사례 없음"

    system_prompt = (
        "너는 삼성 멤버스 커뮤니티의 VOC 담당자 답변 초안을 작성하는 한국어 업무 보조자다. "
        "새 VOC와 유사 기존 사례를 참고하되, 확인되지 않은 원인이나 확정적 결론을 만들지 마라. "
        "증상 재현 정보, 발생 직후 로그, 재현 동영상, 단말/앱/소프트웨어 버전 확인이 필요한 경우 명확하게 요청하라. "
        "답변은 정중한 공식 고객 응대 톤으로 작성하고, 개인정보나 내부 정보는 포함하지 마라. "
        "최종 답변만 작성하라."
    )

    user_prompt = f"""새 VOC:
{query}

유사 기존 VOC/담당자 답변:
{examples_text}

기본 답변 초안:
{base_draft}

작성 요구사항:
1. '안녕하세요, 고객님.'으로 시작
2. 먼저 불편에 대한 사과 포함
3. 새 VOC에 맞게 증상명을 자연스럽게 반영
4. 원인 단정 금지
5. 필요한 경우 삼성 멤버스 앱 → 도움받기 → 질문/오류 보내기 → 오류 보내기 경로 안내
6. 마지막은 감사 인사로 종료
7. 마크다운 표 없이 일반 문단으로 작성
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_with_gemma(
    query: str,
    similar_rows: Iterable[dict[str, Any]],
    base_draft: str,
    config: HFConfig,
) -> str:
    client = _load_client(config)
    messages = build_voc_messages(query, similar_rows, base_draft)

    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Gemma 응답이 비어 있습니다.")
    return str(content).strip()


def test_hf_connection(config: HFConfig) -> str:
    client = _load_client(config)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": "한국어로 짧고 정확하게 답하는 테스트 도우미다."},
            {"role": "user", "content": "삼성멤버스 VOC 답변 시스템 연결 테스트입니다. 한 문장으로 응답해줘."},
        ],
        max_tokens=80,
        temperature=0.1,
        top_p=0.9,
    )
    content = response.choices[0].message.content
    return str(content or "").strip()
