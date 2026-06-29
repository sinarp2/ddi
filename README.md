# DDI 엔진 — 한국어 능력 평가 벤치마크

## 실행 방법

```powershell
# 패키지 설치 (처음 한 번만)
pip install fastapi uvicorn openai instructor pydantic

# FastAPI 서버 실행
uvicorn main:app --reload
```

서버 실행 후 `http://localhost:8000/docs` 에서 API 문서(스웨거) 확인.

---

## 컨버터 실행

워크밴치 원데이터 JSONL → DDI 엔진 입력 형식으로 변환.

```powershell
# 단일 파일
python converter.py --input raw_items.jsonl

# 폴더 일괄 처리 (폴더 내 모든 .jsonl 변환)
python converter.py --dir ./raw --output-dir ./params
```

출력 파일명은 원본 파일명에 `_params` 접미사가 자동으로 붙습니다:
- `{name}_params.jsonl` — DDI 엔진 입력용
- `{name}_params_anchors.jsonl` — approved 문항 앵커 (보정용)
- `{name}_params_errors.jsonl` — 에러/경고 문항 원본

옵션:
| 옵션 | 설명 |
|------|------|
| `--output-dir` | 출력 폴더 지정 (기본: 입력과 같은 폴더) |
| `--no-anchors` | 앵커 파일 생성 안 함 |
| `--no-error-report` | 에러 리포트 생성 안 함 |
| `--strict` | 검증 실패 문항 변환 결과에서 제외 |

---

## 원데이터 필드 명세 (워크밴치 출력 포맷)

### 필수 필드

| 필드명 | 타입 | 설명 |
|--------|------|------|
| `item_id` | str | 문항 고유 ID. 예: `kor-d01-l1-000521` |
| `domain_code` | str | 도메인 코드. 예: `KOR-D01` ~ `KOR-D14` |
| `domain_name` | str | 도메인 명칭. 예: `맞춤법·띄어쓰기` |
| `answer_type` | str | 문항 유형 (아래 허용값 참조) |
| `difficulty` | str | 난이도. `L1` / `L2` / `L3` |
| `task_instruction` | str | 발문 (AI에게 제시할 지시문) |

### answer_type 허용값

| 워크밴치 선택값 | 저장값 | DDI 엔진 변환값 |
|----------------|--------|----------------|
| 객관식 | `single_choice` / `multi_choice` | `multiple_choice` |
| 단답형 | `short_answer` / `closed_constructed` | `short_answer` |
| 폐쇄형 서술형 | `open_constructed` | `constructed_response` |
| 개방형 서술형 | `constructed_response` / `essay` | `constructed_response` / `essay` |

### 선택 필드

| 필드명 | 타입 | 설명 |
|--------|------|------|
| `context` | object | 지문 객체. 실제 텍스트는 `context.passage`에 있음 |
| `context.passage` | str | 평가 지문 텍스트 |
| `user_prompt` | str | 발문 (task_instruction 대체 가능) |
| `choices` | list[object] | 객관식 선택지 목록 |
| `choices[].choice_id` | str | 선택지 기호. `①`~`⑤` |
| `choices[].text` | str | 선택지 텍스트 |
| `answer` | object \| str | 정답 정보 |
| `answer.choice_id` | str | 단일 정답 선택지 기호 |
| `answer.choice_ids` | list[str] | 복수 정답 선택지 기호 목록 |
| `distractor_rationale` | object | 오답별 해설. 키: `①`~`⑤`, 값: 해설 텍스트. Ddist 점수 신뢰도 향상에 활용 |
| `acceptable_answers` | list[str] | 단답형 허용 답안 목록 (띄어쓰기·변이형 포함) |
| `unacceptable_answers` | list[str] | 단답형 불허 답안 목록 |
| `rubric` | list[object] | 채점 루브릭 항목 목록 |
| `rubric[].criterion` | str | 채점 기준 설명 |
| `rubric[].score` | int | 배점 (합계로 정규화하여 weight로 변환) |
| `checkpoints` | list[str] | 채점 시 확인 필수 항목 |
| `explanation` | str | 정답 해설 |
| `difficulty_basis` | str | 난이도 판정 근거 |
| `quality_status` | str | 검토 상태. `approved` / `draft` 등 |
| `source_type` | str | 원천 자료 유형 |
| `source_id` | str | 원천 자료 ID |
| `creator_id` | str | 출제자 ID |
| `reviewer_id` | list[str] | 검토자 ID 목록 |
| `sub_skill` | str | 세부 기술 태그 |
| `scoring_method` | str | 채점 방식 |
| `data_split` | str | 데이터 분할 구분 (train/test 등) |
| `turns` | list[object] | 대화형 문항의 턴 목록 |
| `turns[].role` | str | 발화자 역할 |
| `turns[].content` | str | 발화 내용 |

---

## 배치 호출기 실행

컨버터 출력(`*_params.jsonl`)을 DDI 엔진 API에 일괄 전송.

```powershell
# 서버 먼저 실행 (별도 터미널)
uvicorn main:app --reload

# 단일 파일
python ddi_batch_caller.py --input items_params.jsonl

# 폴더 일괄 처리 (폴더 내 모든 *_params.jsonl 처리)
python ddi_batch_caller.py --dir ./params --output-dir ./results
```

결과 파일명은 `_params` 대신 `_results`가 붙습니다:
- `{name}_results.jsonl` — DDI 점수 결과

옵션:
| 옵션 | 설명 |
|------|------|
| `--output-dir` | 결과 출력 폴더 (기본: 입력과 같은 폴더) |
| `--concurrency` | 동시 API 호출 수 (기본: 1) |
| `--api-url` | API 서버 주소 (기본: `http://localhost:8000/api/v1/measure-ddi`) |

---

## 진단 도구

```powershell
# 폴더 내 answer_type 분포 확인
python scan_answer_types.py --dir ./data

# 폴더 내 전체 필드 구조 확인
python scan_schema.py --dir ./data
```
