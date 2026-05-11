from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.db import DB_PATH, clear_db, export_joined, fetch_posts, fetch_replies, init_db, upsert_post, upsert_replies, Post, Reply
from src.scraper import crawl_keyword, import_url_list, post_id_from_url, now_iso
from src.voc_model import generate_answer, model_available, train_retrieval_model
from src.hf_llm import DEFAULT_HF_MODEL, DEFAULT_HF_PROVIDER, generate_with_gemma, get_hf_config, parse_model_provider, test_hf_connection

load_dotenv()

# Streamlit Cloud secrets → 환경변수 반영
for key in ["SAMSUNG_USER_AGENT", "HF_TOKEN", "HF_MODEL", "HF_PROVIDER", "HF_MAX_TOKENS", "HF_TEMPERATURE", "HF_TOP_P"]:
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

st.set_page_config(page_title="삼성멤버스 VOC 수집·자동답변", page_icon="📱", layout="wide")
init_db(DB_PATH)

st.title("📱 삼성멤버스 VOC 수집 · 담당자 답변 학습 · 자동 답변 초안")
st.caption("검색어 기반 게시글 수집, 담당자/Moderator 답글 저장, 유사 VOC 검색, 답변 초안 생성을 한 화면에서 관리합니다.")

with st.sidebar:
    st.header("설정")
    db_path = st.text_input("SQLite DB 경로", value=str(DB_PATH))
    st.caption("Streamlit Cloud는 파일 시스템이 영구 저장소가 아닐 수 있습니다. 중요한 결과는 CSV로 내려받으세요.")
    if st.button("DB 초기화", type="secondary"):
        clear_db(db_path)
        st.success("DB를 초기화했습니다.")
    st.divider()
    st.subheader("Gemma 4 / Hugging Face")
    st.caption("Hugging Face Inference Providers에서 DeepInfra provider로 Gemma 4 26B A4B IT를 호출합니다.")

    default_model_text = os.getenv("HF_MODEL", f"{DEFAULT_HF_MODEL}:{os.getenv('HF_PROVIDER', DEFAULT_HF_PROVIDER)}")
    hf_model_provider = st.text_input("HF 모델:Provider", value=default_model_text, help="예: google/gemma-4-26B-A4B-it:deepinfra")
    parsed_model, parsed_provider = parse_model_provider(hf_model_provider)
    hf_provider = st.text_input("Provider", value=os.getenv("HF_PROVIDER", parsed_provider or DEFAULT_HF_PROVIDER))
    hf_token_input = st.text_input(
        "HF_TOKEN",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        help="로컬은 .env, Streamlit Cloud는 Secrets에 넣는 것을 권장합니다. 사이드바 입력도 가능합니다.",
    )
    hf_max_tokens = st.number_input("최대 생성 토큰", min_value=256, max_value=4096, value=int(os.getenv("HF_MAX_TOKENS", "1600")), step=128)
    hf_temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=float(os.getenv("HF_TEMPERATURE", "0.25")), step=0.05)
    hf_top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=float(os.getenv("HF_TOP_P", "0.95")), step=0.05)
    use_gemma = st.toggle("Gemma 4로 답변 초안 다듬기", value=True)

    hf_config = get_hf_config(
        token=hf_token_input,
        model=parsed_model,
        provider=hf_provider,
        max_tokens=int(hf_max_tokens),
        temperature=float(hf_temperature),
        top_p=float(hf_top_p),
    )
    st.caption(f"현재 설정: model=`{hf_config.model}`, provider=`{hf_config.provider}`")
    if st.button("HF / DeepInfra 연결 테스트"):
        try:
            test_text = test_hf_connection(hf_config)
            st.success("연결 성공")
            st.write(test_text)
        except Exception as e:
            st.error(f"연결 실패: {e}")


def rows_to_df(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def add_sample_data(db_path: str):
    sample_url = "https://r1.community.samsung.com/t5/galaxy-s/sample/m-p/00000001"
    post = Post(
        post_id=post_id_from_url(sample_url),
        url=sample_url,
        title="갤럭시 S26 울트라 화면 깜빡거림",
        author="sample_user",
        created_at="2026-03-30 10:00",
        board="갤럭시 S",
        content="갤럭시 S26 울트라 사용 중 특정 앱 실행 후 화면이 깜빡거리고 터치가 늦게 반응합니다. 재부팅하면 잠깐 괜찮다가 다시 발생합니다.",
        keyword="갤럭시 s26",
        fetched_at=now_iso(),
    )
    reply = Reply(
        reply_id="sample-reply-00000001",
        post_id=post.post_id,
        url=sample_url,
        author="GPU담당",
        role="Moderator",
        created_at="2026-03-30 13:42",
        content=(
            "안녕하세요, 고객님. GPU담당입니다. 먼저 이용에 불편을 드려 죄송합니다. "
            "고객님께서 겪고 계신 증상에 대한 정확한 파악을 위해 재현 동영상과 이슈가 발생한 직후의 로그가 필요합니다. "
            "삼성 멤버스 앱 → 도움받기 → 질문/오류 보내기 → 오류 보내기에서 시스템 로그 데이터 보내기를 체크하신 후 "
            "재현 동영상과 세부 증상을 함께 전달해 주시면 확인 후 답변드리겠습니다."
        ),
        is_moderator=1,
    )
    upsert_post(post, db_path=db_path)
    upsert_replies([reply], db_path=db_path)


tab1, tab2, tab3, tab4, tab5 = st.tabs(["① 수집", "② 리스트 관리", "③ 학습/색인", "④ 자동 답변", "⑤ 배포 가이드"])

with tab1:
    st.subheader("① 삼성멤버스 검색어 기반 수집")
    st.info("과도한 요청은 피하고, 사이트 이용 약관과 robots 정책을 준수하세요. 요청 간 지연 시간을 1초 이상 두는 것을 권장합니다.")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        keyword = st.text_input("검색어", value="갤럭시 s26")
    with col2:
        pages = st.number_input("검색 페이지 수", min_value=1, max_value=10, value=1)
    with col3:
        max_posts = st.number_input("최대 게시글", min_value=1, max_value=100, value=10)
    with col4:
        delay = st.number_input("요청 지연초", min_value=0.2, max_value=10.0, value=1.0, step=0.2)

    if st.button("검색어로 수집 시작", type="primary"):
        with st.spinner("수집 중입니다. 잠시만 기다려 주세요."):
            result = crawl_keyword(keyword, int(pages), int(max_posts), float(delay), db_path=db_path)
        st.success(f"수집 완료: URL {result.discovered_urls}개 발견, 게시글 {result.saved_posts}개 저장, 답글 {result.saved_replies}개 저장")
        if result.errors:
            st.warning("일부 오류가 있었습니다.")
            st.code("\n".join(result.errors[:20]))

    st.divider()
    st.subheader("개별 게시글 URL 직접 수집")
    urls_text = st.text_area(
        "게시글 URL을 한 줄에 하나씩 입력",
        value="https://r1.community.samsung.com/t5/갤럭시-s/갤럭시-s26-울트라-깜빡거림/m-p/37409116",
        height=120,
    )
    manual_keyword = st.text_input("직접 URL 수집용 키워드", value="manual")
    if st.button("URL 직접 수집"):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        with st.spinner("URL 수집 중입니다."):
            result = import_url_list(urls, keyword=manual_keyword, delay_sec=float(delay), db_path=db_path)
        st.success(f"직접 수집 완료: 게시글 {result.saved_posts}개, 답글 {result.saved_replies}개")
        if result.errors:
            st.warning("오류 내용")
            st.code("\n".join(result.errors[:20]))

    st.divider()
    if st.button("테스트용 샘플 Q&A 1건 추가"):
        add_sample_data(db_path)
        st.success("샘플 데이터를 추가했습니다. '학습/색인' 탭에서 학습을 실행해 보세요.")

with tab2:
    st.subheader("② 저장된 VOC 및 담당자 답글 리스트")
    filter_kw = st.text_input("리스트 필터", value="")
    posts_df = rows_to_df(fetch_posts(filter_kw or None, db_path=db_path))
    replies_df = rows_to_df(fetch_replies(only_moderator=True, db_path=db_path))
    c1, c2, c3 = st.columns(3)
    c1.metric("저장 게시글", len(posts_df))
    c2.metric("담당자 답글", len(replies_df))
    c3.metric("학습 가능 Q&A", len(export_joined(db_path)))

    st.markdown("#### 게시글")
    if posts_df.empty:
        st.caption("아직 저장된 게시글이 없습니다.")
    else:
        show_cols = [c for c in ["keyword", "board", "title", "author", "created_at", "url"] if c in posts_df.columns]
        st.dataframe(posts_df[show_cols], use_container_width=True, hide_index=True)

    st.markdown("#### 담당자/Moderator 답글")
    if replies_df.empty:
        st.caption("아직 저장된 담당자 답글이 없습니다.")
    else:
        show_cols = [c for c in ["post_id", "author", "role", "created_at", "content", "url"] if c in replies_df.columns]
        st.dataframe(replies_df[show_cols], use_container_width=True, hide_index=True)

    joined = rows_to_df(export_joined(db_path))
    if not joined.empty:
        csv = joined.to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSV로 내보내기", data=csv, file_name="samsung_members_voc_export.csv", mime="text/csv")

with tab3:
    st.subheader("③ 유사 VOC 검색 모델 학습/색인 생성")
    st.write("수집된 원글 + 담당자 답글 쌍을 이용해 TF‑IDF 유사도 검색 모델을 생성합니다. 이 단계는 API 없이 로컬에서 작동합니다.")
    n_clusters = st.slider("군집 수", min_value=1, max_value=20, value=5)
    if st.button("학습/색인 생성", type="primary"):
        result = train_retrieval_model(db_path=db_path, n_clusters=int(n_clusters))
        if result.ok:
            st.success(result.message)
        else:
            st.warning(result.message)
    st.write("검색 모델 상태:", "✅ 생성됨" if model_available() else "❌ 아직 없음")
    st.info("이 단계는 기존 Q&A를 빠르게 찾기 위한 검색 색인입니다. Gemma 4는 자동 답변 탭에서 유사 사례와 기본 초안을 받아 최종 답변 문장을 생성합니다.")

with tab4:
    st.subheader("④ 새 VOC 입력 → 유사 사례 검색 → 답변 초안")
    query = st.text_area(
        "새로 올라온 VOC 내용",
        value="갤럭시 S26 울트라에서 게임 실행 중 화면이 깜빡이고 끊깁니다. 업데이트 후 더 심해진 것 같습니다.",
        height=160,
    )
    top_k = st.slider("유사 사례 개수", 1, 10, 5)
    if st.button("예상 답변 작성", type="primary"):
        if not model_available():
            st.warning("먼저 '학습/색인' 탭에서 모델을 생성하세요. 데이터가 없으면 수집 또는 샘플 추가가 필요합니다.")
        else:
            base_answer, similar = generate_answer(query, top_k=top_k)
            final_answer = base_answer
            if use_gemma:
                try:
                    with st.spinner("Gemma 4 26B A4B IT / DeepInfra로 답변 초안을 생성 중입니다."):
                        final_answer = generate_with_gemma(
                            query=query,
                            similar_rows=similar.to_dict(orient="records"),
                            base_draft=base_answer,
                            config=hf_config,
                        )
                    st.success("Gemma 4 답변 생성 완료")
                except Exception as e:
                    st.warning(f"Gemma 4 호출 실패. 기본 템플릿 초안을 표시합니다: {e}")

            st.markdown("#### 답변 초안")
            st.text_area("초안", value=final_answer, height=420)
            with st.expander("기본 검색 기반 초안 보기"):
                st.text_area("기본 초안", value=base_answer, height=260)
            st.markdown("#### 유사 VOC")
            view_cols = [c for c in ["similarity", "keyword", "title", "reply_author", "reply_role", "post_url", "answer"] if c in similar.columns]
            st.dataframe(similar[view_cols], use_container_width=True, hide_index=True)

with tab5:
    st.subheader("⑤ VS Code 실행 및 Streamlit Cloud 배포")
    st.markdown(
        """
### 로컬 VS Code 실행
```powershell
cd samsung_members_voc_app
py -3.11 -m venv venv
.\\venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### GitHub 업로드
```powershell
git init
git add .
git commit -m "Initial commit: Samsung Members VOC app"
git branch -M main
git remote add origin https://github.com/사용자명/samsung_members_voc_app.git
git push -u origin main
```

### Streamlit Cloud 배포
1. Streamlit Cloud에서 New app 선택
2. GitHub repo 선택
3. Main file path에 `app.py` 입력
4. Python 버전은 `runtime.txt`의 `python-3.11` 사용
5. Gemma 4 / Hugging Face를 쓸 경우 Secrets에 아래처럼 입력
```toml
SAMSUNG_USER_AGENT="Mozilla/5.0 ..."
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
HF_MODEL="google/gemma-4-26B-A4B-it"
HF_PROVIDER="deepinfra"
HF_MAX_TOKENS="1600"
HF_TEMPERATURE="0.25"
HF_TOP_P="0.95"
```

### 주의
- Streamlit Cloud의 파일 저장은 영구 DB 용도로 적합하지 않을 수 있으므로 CSV export를 자주 사용하세요.
- 삼성멤버스 페이지 구조가 바뀌거나 크롤링이 차단되면 URL 직접 수집 또는 CSV 업로드 방식으로 보완하세요.
- 고객 개인정보가 포함될 수 있으므로 외부 API 전송 전 반드시 마스킹/검토가 필요합니다.
"""
    )
