# Asset Filter

NovelAI로 대량 생성한 캐릭터 감정 이미지를 자동으로 필터링하는 Windows 포터블 GUI 앱.

감정 키워드별로 수십 장씩 생성된 이미지 중 CLIP 또는 **WD Tagger v3**를 사용해 점수를 매기고, 상위 N장을 자동으로 출력 폴더로 복사합니다.

---

## 기능

- **WD Tagger v3** (기본) — Danbooru 기반 애니 이미지 분류 모델. NovelAI 감정 태그와 같은 어휘 체계를 사용해 CLIP보다 훨씬 정확하게 감정 표현 이미지를 선별합니다.
- **CLIP ViT-B-32 / ViT-L-14** — 텍스트-이미지 유사도 기반 대안
- PNG EXIF 메타데이터(`char_captions`)에서 캐릭터 프롬프트를 자동 추출해 CLIP 스코어링에 활용
- 진행 상황 실시간 표시 (프로그레스 바 + 현재 처리 중인 감정)
- 필터링 결과를 `report.json`으로 저장
- 완전한 포터블 구성 (임베디드 Python 지원)

---

## 입력 파일명 형식

```
{캐릭터}.{감정 키워드}.{번호}.png
```

예시:
```
gabriel.acting coy.14.png
gabriel.happy smile.3.png
gabriel.angry.30.png
```

- 감정 키워드에 공백 포함 가능 (`acting coy`, `happy smile`)
- 같은 폴더에 모든 이미지를 플랫하게 넣어두면 됩니다

---

## 설치 및 실행

### 일반 Python 환경 (개발용)

```bash
pip install -r requirements.txt
python src/main.py
```

### Windows 포터블 배포

1. [python.org](https://www.python.org/downloads/windows/)에서 **Windows embeddable package (64-bit)** 다운로드
2. `python/` 폴더에 압축 해제
3. `python3xx._pth` 파일에서 `import site` 주석 해제
4. `get-pip.py`를 받아서 `python\python.exe get-pip.py` 실행
5. `setup.bat` 실행 (의존성 설치)
6. 이후 `run.bat`으로 실행

```
asset-filter/
├── run.bat          ← 더블클릭으로 실행
├── setup.bat        ← 최초 1회 실행
├── requirements.txt
├── src/
├── python/          ← 임베디드 Python (직접 준비)
└── cache/           ← 모델 캐시 (자동 생성)
```

---

## 사용 방법

1. **Input Folder** — 이미지 폴더를 선택하면 감정 수와 이미지 수를 자동으로 표시합니다.
2. **Top N per emotion** — 감정별 상위 N장을 선택합니다 (기본값: 10).
3. **Model** — WD Tagger v3 (권장), CLIP ViT-B-32, CLIP ViT-L-14 중 선택.
4. **Run Filter** — 백그라운드에서 스코어링 후 출력 폴더에 복사합니다.
5. 완료 후 **Open Output**으로 결과 폴더를 엽니다.

출력 폴더는 입력 폴더 옆에 `{폴더명}_filtered`로 생성됩니다. 모든 필터링된 이미지가 단일 폴더에 저장됩니다.

---

## 의존성

| 패키지 | 용도 |
|--------|------|
| `onnxruntime` | WD Tagger v3 ONNX 추론 |
| `huggingface_hub` | WD Tagger v3 모델 다운로드 |
| `open-clip-torch` | CLIP 모델 |
| `torch` (CPU) | CLIP 추론 |
| `Pillow` | 이미지 로드 및 전처리 |
| `numpy` | 배열 연산 |

---

## 모델 비교

| | WD Tagger v3 | CLIP ViT-B-32 | CLIP ViT-L-14 |
|---|---|---|---|
| 학습 데이터 | Danbooru (애니) | 자연 이미지 | 자연 이미지 |
| 감정 분류 방식 | 태그 확률 직접 출력 | 텍스트 유사도 | 텍스트 유사도 |
| 애니 이미지 적합성 | 매우 높음 | 낮음 | 보통 |
| 속도 (CPU) | 빠름 (단일 패스) | 보통 | 느림 |
| 모델 크기 | ~350MB | ~350MB | ~900MB |

WD Tagger v3는 이미지 전체를 **한 번만** 추론하여 모든 감정 점수를 동시에 계산하므로, 감정 키워드 수가 많아도 속도가 일정합니다.
