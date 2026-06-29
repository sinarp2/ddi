# CLAUDE.md — 프로젝트 지침서

이 파일은 Claude Code가 이 프로젝트를 열 때마다 자동으로 읽는 파일입니다.
여기에 적힌 내용은 매 대화마다 Claude가 기억합니다.

---

## 프로젝트 개요

**프로젝트명**: 한국어 능력 평가 벤치마크 (KoBALT)  
**발주처**: 나라지식정보 컨소시엄  
**목적**: AI의 한국어 능력을 평가하는 고품질 벤치마크 문항 11,000개 개발

---

## 핵심 참조 파일

| 파일 | 설명 |
|------|------|
| `참조/260527_지침서_v1.1_...pdf` | 문항 출제 공식 지침서 (47쪽). 문항 설계 시 반드시 참조. |
| `참조/260609_sinnce260605_AI모델의 한국어 능력 평가를 위한 문항 단위 증거 기반 평가 프레임워크_v0.3.pdf` | DDI 이론 기반 문서. 평가 프레임워크 설계 원리. |
| `참조/ddi_auto_engine.py` | DDI 산출 엔진 v1 (규칙 기반 전용, 참조용). |
| `참조/ddi_engine_v2.py` | DDI 산출 엔진 v2 원본 (현재 테스트 기준 파일). |
| `main.py` | **현재 운영 메인 파일** — v2 엔진 그대로, `uvicorn main:app --reload`로 실행. |
| `참조/sample_items.jsonl` | 문항 입력 샘플 (JSONL 형식). |
| `참조/ddi_results.jsonl` | DDI 엔진 출력 결과 샘플. |
| `참조/ddi_demo_output.json` | DDI 데모 실행 결과 (DEMO-L3-EMAIL-001). |
| `참조/ddi_viewer.html` | DDI 결과 시각화 뷰어. |
| `converter.py` | 원데이터 → DDI 엔진 입력 변환기. 앵커 자동 생성 포함. |
| `작업지시서_한국어능력평가벤치마크_문항출제.md` | 지침서를 요약한 실무용 작업지시파일. |

---

## DDI (설계난이도지수, Design Difficulty Index)

문항 난이도를 자동으로 수치화하는 핵심 개념. `ddi_auto_engine.py`가 이를 산출.

**10개 구성 요소 (벡터)**

| 코드 | 의미 |
|------|------|
| R | Reasoning — 추론 단계 수·복잡도 |
| C | Context — 맥락 의존도 |
| K | Korean-specificity — 한국어 특성 부하량 |
| P | Pragmatics — 화용·사회언어학 부하량 |
| E | External knowledge — 외부 지식 의존도 |
| O | Openness — 응답 개방성 |
| X | Constraint complexity — 제약 조건 복잡도 |
| Ddist | Distractor plausibility — 오답 매력도 (객관식만) |
| M | Multi-construct integration — 다중 구인 결합도 |
| Q | Question clarity — 문항 명료도 |

**DDI 점수 → 난이도 매핑**: L1 < 35 / L2: 35~70 / L3 > 70

**엔진 실행 명령 (PowerShell) — main.py 기준**
```powershell
# 패키지 설치 (처음 한 번만)
pip install fastapi uvicorn openai instructor pydantic

# FastAPI 서버 실행
uvicorn main:app --reload

# CLI 데모
python main.py --demo

# 문항 배치 처리
python main.py --input items.jsonl --output results.jsonl
python main.py --input items.jsonl --output results.csv --format csv

# 앵커 보정 포함
python main.py --input items.jsonl --anchors anchors.jsonl --output results.jsonl
```

서버 실행 후 `http://localhost:8000/docs` 에서 API 문서 확인 가능.

---

## 도메인 및 문항 구조

- **평가 도메인**: 14개 (맞춤법·띄어쓰기, 경어법, 문장성분 호응, 담화 연결·접속, 문체/레지스터, 화시, 화행·공손, 관용표현, 유의어, 고어, 방언, 신조어, 추론, 무형대용어 복원)
- **난이도**: L1(20%) / L2(50%) / L3(30%)
- **문항 유형**: 선택형(MCQ) / 단답형 / 서술형(개방형)
- **문항 ID 형식**: `[도메인코드]-L[난이도]-[4자리번호]` (예: `HON-L2-0001`)
- **총 문항 수**: 11,000개

---

## 작업 시 필수 규칙

- AI 생성 문항 사용 절대 금지 — 인간 전문가가 직접 설계
- 외부 자료 인용 시 70% 이상 실질적 개작 필수
- 정답 근거는 지문 내 명시적으로 포함되어야 함
- 혐오·차별 표현 배제

---

## 협업 방식

- 모든 소통은 **한국어**로
- 설명은 쉽고 구체적으로 (사용자가 Claude Code 입문자)
- 새 파일 생성 전 기존 파일 수정을 먼저 검토
