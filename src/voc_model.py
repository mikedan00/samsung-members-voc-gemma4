from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from .db import connect, init_db

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = MODEL_DIR / "tfidf_matrix.joblib"
QA_PATH = MODEL_DIR / "qa_records.joblib"
CLUSTER_PATH = MODEL_DIR / "clusters.joblib"

PHONE_RE = re.compile(r"(?:\+?82[-\s]?)?0?1[016789][-\s]?\d{3,4}[-\s]?\d{4}")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://\S+")
LONG_NUM_RE = re.compile(r"\b\d{6,}\b")


def mask_pii(text: str | None) -> str:
    if not text:
        return ""
    text = EMAIL_RE.sub("[이메일]", text)
    text = PHONE_RE.sub("[전화번호]", text)
    text = URL_RE.sub("[URL]", text)
    text = LONG_NUM_RE.sub("[식별번호]", text)
    return text


def normalize(text: str | None) -> str:
    if not text:
        return ""
    text = mask_pii(text)
    text = text.replace("\u200e", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_training_records(db_path: str = "data/voc.db") -> pd.DataFrame:
    init_db(db_path)
    sql = """
    SELECT
        p.post_id,
        p.keyword,
        p.board,
        p.title,
        p.content AS question,
        p.url AS post_url,
        r.author AS reply_author,
        r.role AS reply_role,
        r.content AS answer,
        r.url AS reply_url
    FROM posts p
    JOIN replies r ON p.post_id = r.post_id
    WHERE r.is_moderator = 1
      AND LENGTH(COALESCE(p.content, '')) > 10
      AND LENGTH(COALESCE(r.content, '')) > 10
    """
    with connect(db_path) as con:
        df = pd.read_sql_query(sql, con)
    if df.empty:
        return df
    df["question_text"] = (df["title"].fillna("") + "\n" + df["question"].fillna("")).map(normalize)
    df["answer"] = df["answer"].map(normalize)
    df = df.drop_duplicates(subset=["post_id", "answer"]).reset_index(drop=True)
    return df


@dataclass
class TrainResult:
    ok: bool
    rows: int
    clusters: int
    message: str


def train_retrieval_model(db_path: str = "data/voc.db", n_clusters: int = 5) -> TrainResult:
    df = load_training_records(db_path)
    if df.empty:
        return TrainResult(False, 0, 0, "담당자 답글이 포함된 학습 데이터가 없습니다. 먼저 게시글을 수집하세요.")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=40000,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["question_text"].tolist())

    k = max(1, min(n_clusters, len(df)))
    clusters = np.zeros(len(df), dtype=int)
    cluster_model = None
    if k >= 2:
        cluster_model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        clusters = cluster_model.fit_predict(X)
    df["cluster"] = clusters

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(X, MATRIX_PATH)
    joblib.dump(cluster_model, CLUSTER_PATH)
    joblib.dump(df, QA_PATH)
    return TrainResult(True, len(df), k, f"학습 완료: {len(df)}개 Q&A, {k}개 군집")


def model_available() -> bool:
    return VECTORIZER_PATH.exists() and MATRIX_PATH.exists() and QA_PATH.exists()


def load_model():
    if not model_available():
        raise FileNotFoundError("학습된 검색 모델이 없습니다. 먼저 '학습/색인 생성'을 실행하세요.")
    vectorizer = joblib.load(VECTORIZER_PATH)
    X = joblib.load(MATRIX_PATH)
    df = joblib.load(QA_PATH)
    cluster_model = joblib.load(CLUSTER_PATH) if CLUSTER_PATH.exists() else None
    return vectorizer, X, df, cluster_model


def find_similar_vocs(query: str, top_k: int = 5) -> pd.DataFrame:
    vectorizer, X, df, cluster_model = load_model()
    q = normalize(query)
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, X).ravel()
    order = sims.argsort()[::-1][:top_k]
    result = df.iloc[order].copy()
    result["similarity"] = sims[order]
    return result.reset_index(drop=True)


def compact_answer(answer: str, max_chars: int = 1800) -> str:
    answer = normalize(answer)
    if len(answer) <= max_chars:
        return answer
    return answer[:max_chars].rstrip() + " ..."


def build_template_answer(query: str, similar: pd.DataFrame) -> str:
    if similar.empty:
        return (
            "안녕하세요, 고객님. 담당자입니다.\n"
            "먼저 이용에 불편을 드려 죄송합니다.\n\n"
            "말씀해 주신 증상은 추가 확인이 필요합니다. 증상이 발생한 직후 삼성 멤버스 앱의 "
            "도움받기 → 질문/오류 보내기 → 오류 보내기를 통해 시스템 로그와 재현 동영상, 발생 조건을 함께 전달해 주시면 "
            "확인 후 답변드리겠습니다.\n\n"
            "감사합니다."
        )

    best = similar.iloc[0]
    best_answer = compact_answer(str(best.get("answer", "")))
    author = str(best.get("reply_author", "담당자")) or "담당자"
    sim = float(best.get("similarity", 0.0))

    # 기존 담당자 답글의 톤을 유지하되, 새 VOC에 맞춘 초안으로 재구성합니다.
    return f"""안녕하세요, 고객님. {author}입니다.
먼저 이용에 불편을 드려 죄송합니다.

말씀해 주신 증상은 기존 유사 VOC와 비교했을 때 약 {sim:.0%} 수준으로 유사한 사례가 확인됩니다. 정확한 원인 파악을 위해 증상이 발생한 직후의 단말 로그와 재현 동영상, 발생 조건이 필요합니다.

아래는 가장 유사한 기존 담당자 답변을 바탕으로 작성한 답변 초안입니다.

{best_answer}

※ 실제 고객 응대 전에는 모델명, 소프트웨어 버전, 발생 앱/화면, 재현 경로, 개인정보 포함 여부를 담당자가 반드시 검토해 주세요."""



def generate_answer(query: str, top_k: int = 5) -> tuple[str, pd.DataFrame]:
    similar = find_similar_vocs(query, top_k=top_k)
    draft = build_template_answer(query, similar)
    return draft, similar
