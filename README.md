# Samsung Members VOC Collector & Auto Reply Draft

삼성멤버스 커뮤니티의 검색 결과/게시글 URL에서 VOC 원글과 담당자/Moderator 답글을 수집하고, SQLite에 저장한 뒤 유사 VOC 검색과 답변 초안을 생성하는 Streamlit 앱입니다.

## 주요 기능

- 검색어 기반 삼성멤버스 게시글 URL 수집
- 개별 게시글 URL 직접 수집
- 제목, 작성자, 날짜, 게시판, 본문 저장
- `Moderator`, `담당`, `운영자`, `관리자` 표시 기반 담당자 답글 추출
- SQLite 저장 및 CSV 내보내기
- TF-IDF 기반 유사 VOC 검색 모델 생성
- 기존 담당자 답글 기반 예상 답변 초안 작성
- Hugging Face `HF_TOKEN`으로 `google/gemma-4-26B-A4B-it` 모델을 `deepinfra` provider에서 호출해 답변 초안 생성

## 폴더 구조

```text
samsung_members_voc_app/
├─ app.py
├─ requirements.txt
├─ runtime.txt
├─ README.md
├─ .env.example
├─ .streamlit/
│  └─ config.toml
├─ data/
│  └─ voc.db              # 실행 후 자동 생성
├─ models/                # 학습 후 자동 생성
└─ src/
   ├─ db.py
   ├─ scraper.py
   ├─ voc_model.py
   └─ hf_llm.py
```

## VS Code 로컬 실행

```powershell
cd samsung_members_voc_app
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속.

## Streamlit Cloud 배포

1. GitHub에 이 폴더 전체 업로드
2. Streamlit Cloud → New app
3. Repository 선택
4. Main file path: `app.py`
5. Python 버전: `runtime.txt`의 `python-3.11`
6. Gemma 4 / Hugging Face를 사용할 경우 Secrets에 아래 값 입력

```toml
SAMSUNG_USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/147.0 Safari/537.36"
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
HF_MODEL="google/gemma-4-26B-A4B-it"
HF_PROVIDER="deepinfra"
HF_MAX_TOKENS="1600"
HF_TEMPERATURE="0.25"
HF_TOP_P="0.95"
```

사이드바의 `HF 모델:Provider` 입력란에는 아래처럼 한 줄로 넣어도 됩니다.

```text
google/gemma-4-26B-A4B-it:deepinfra
```

내부 구현은 `huggingface_hub.InferenceClient(provider="deepinfra", token=HF_TOKEN)` 방식이며, 모델은 chat completions 호출 시 `google/gemma-4-26B-A4B-it`로 전달합니다.

## 사용 순서

1. **수집** 탭에서 검색어 또는 개별 URL로 게시글 수집
2. **리스트 관리** 탭에서 저장된 원글/담당자 답글 확인
3. **학습/색인** 탭에서 TF-IDF 검색 모델 생성
4. **자동 답변** 탭에서 새 VOC 입력 후 TF-IDF 유사 사례 검색 + Gemma 4 답변 초안 생성
5. CSV 내보내기로 수집 데이터를 백업

## 운영 주의사항

- 공개 웹페이지 요청 시 사이트 이용 약관과 robots 정책을 준수하세요.
- 요청 지연 시간을 충분히 두고 과도한 수집을 피하세요.
- 고객 개인정보가 포함될 수 있으므로 외부 API 전송 전 개인정보 마스킹과 검토가 필요합니다.
- Streamlit Cloud 파일 시스템은 영구 DB 저장소로 적합하지 않을 수 있으므로 CSV export 또는 외부 DB 연동을 권장합니다.
- 삼성멤버스 HTML 구조가 바뀌면 `src/scraper.py`의 CSS selector 보정이 필요할 수 있습니다.


## Hugging Face / DeepInfra 호출 구조

이 프로젝트는 DeepInfra API 키를 직접 쓰지 않고 Hugging Face의 Inference Providers 라우팅을 사용합니다.
따라서 필요한 키는 `HF_TOKEN`입니다.

```python
from huggingface_hub import InferenceClient

client = InferenceClient(provider="deepinfra", token=HF_TOKEN)
response = client.chat.completions.create(
    model="google/gemma-4-26B-A4B-it",
    messages=[{"role": "user", "content": "..."}],
)
```

`google/gemma-4-26B-A4B-it:deepinfra`는 앱 입력 편의를 위한 표기이며, 실제 API 호출 시에는 모델과 provider를 분리해서 전달합니다.
