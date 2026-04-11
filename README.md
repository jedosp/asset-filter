# Asset Filter

NovelAI로 대량 생성한 캐릭터 감정 이미지를 자동으로 필터링하는 Windows GUI 앱.

**Camie Tagger v2** + **Aesthetic Predictor V2.5** + **MediaPipe Face Detection** + **DINOv2 Reference Consistency**를 조합해 감정 키워드 매칭, 시각적 품질, 얼굴 프레이밍, 캐릭터 디자인 일관성을 함께 평가하고 상위 N장을 자동으로 출력 폴더로 복사합니다.

---

## 기능

- **Camie Tagger v2** — 70k+ Danbooru 태그로 정밀한 감정·결함 분류 (512px, Micro F1 67.3%)
- **Aesthetic Predictor V2.5** — SigLIP 기반 시각적 품질 평가 (1-10 스케일, 일러스트/애니메 특화)
- **Face Framing** — MediaPipe 얼굴 인식으로 표정 가시성 평가 (Hard Filter / Weighted 모드)
- **Reference Consistency** — 참조 이미지 1-5장과의 DINOv2 시각 유사도로 디자인 일관성 평가
- **Tag Deviation Auto-Exclude** — 레퍼런스에 없는 시각적 특징(안경, 귀걸이 등)을 자동 감지·제외
- **다차원 스코어링** — 감정 + 미학 + 얼굴 + 일관성 가중치를 조합한 통합 랭킹
- **EXIF 태그 결합 스코어링** — NovelAI char_captions 메타데이터를 활용해 정확도 향상
- **해부학 결함 자동 감지** — 22개 결함 태그 중 최대값(max)으로 불량 이미지 자동 감점
- **DirectML GPU 가속** — Windows GPU 자동 감지, GPU 없으면 CPU 자동 전환
- **이미지 1회 로딩** — 모든 스코어러가 동일 이미지를 공유, I/O 최소화
- 전체 이미지를 단일 패스로 추론 (감정 수가 많아도 속도 일정)
- 모델 다운로드 / 이미지 분석 단계별 프로그레스 바
- 필터링 결과를 `report.json`으로 저장 (`neg_score` 포함)

---

## 다운로드 및 실행

### 릴리즈 (일반 사용자)

[Releases](https://github.com/jedosp/asset-filter/releases) 페이지에서 다운로드:

| 파일 | 설명 |
|------|------|
| `AssetFilter-v1.1.0-cuda-split.*` | 임베디드 Python + CUDA torch 포함 (분할 압축, GPU 없으면 CPU 자동 전환) |

1. `.z01` + `.zip` 두 파일을 모두 다운로드한 뒤 같은 폴더에 놓고 `.zip`을 해제
2. `run.bat` 또는 `AssetFilter.exe` 실행
3. 첫 실행 시 모델 자동 다운로드 (Camie Tagger ~800MB, Aesthetic Predictor ~3.5GB)

> **AssetFilter.exe** — torch 미포함 독립 실행 파일. Aesthetic Score / Reference Consistency 옵션은 비활성화되며, Camie Tagger는 DirectML 가능 환경에서는 GPU를 사용하고 아니면 CPU로 동작합니다.
> **run.bat** — 임베디드 Python 환경으로 실행. NVIDIA GPU가 있으면 CUDA 가속, 없으면 CPU로 자동 전환.

자동 다운로드되는 모델들은 모두 실행 기준 폴더 아래의 `models` 폴더에 저장됩니다.

- Camie Tagger: `models/huggingface/...`
- Aesthetic Predictor: `models/huggingface/...`, `models/torch/...`
- DINOv2 Reference Consistency: `models/huggingface/...`

### 소스에서 실행 (개발용)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python src/main.py
```

### 빌드 (독립 실행 파일)

```bash
pip install pyinstaller
build.bat
```

PyInstaller 결과물은 `dist/AssetFilter.exe`에 생성되고, 배포용 실행 파일은 `release/AssetFilter.exe`로 별도 복사됩니다.

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

## 사용 방법

1. **Input Folder** — 이미지 폴더를 선택하면 감정 수와 이미지 수를 자동으로 표시합니다.
2. **Top N per emotion** — 감정별 상위 N장을 선택합니다 (기본값: 10).
3. **Scoring Options** — 아래 옵션을 조합해서 사용합니다:
   - **Enable Aesthetic Score** — 시각적 품질 평가 (기본: 켜짐)
   - **Enable Face Framing** — 얼굴 프레이밍 평가 (기본: 꺼짐)
   - **Enable Reference Consistency** — DINOv2 기반 참조 이미지 일치도 평가 (기본: 꺼짐)
   - **Weights** — 감정/미학/얼굴 가중치를 자유롭게 조절 (합계 1.0 자동 유지)
4. **Reference Consistency** — 원하는 캐릭터 디자인이 잘 나온 이미지 1-5장을 선택합니다. 같은 입력 폴더의 이미지나 이전 실행 결과를 참조로 사용할 수 있습니다.
5. **Run Filter** — 백그라운드에서 스코어링 후 출력 폴더에 복사합니다.
6. 완료 후 **Open Output**으로 결과 폴더를 엽니다.

### DINOv2 모델 준비

Reference Consistency 기능은 공개 DINOv2 모델을 첫 실행 시 자동으로 다운로드합니다.

별도 승인이나 수동 모델 배치는 필요하지 않으며, 캐시는 실행 기준 폴더 아래의 다음 위치에 저장됩니다.

`models/huggingface`

실제로는 Hugging Face 캐시 구조 아래에 `facebook/dinov2-base` 관련 파일이 저장됩니다.

`AssetFilter.exe`에는 `torch`가 없으므로 Reference Consistency가 비활성화됩니다. `run.bat` 또는 개발 환경에서 실행하면 DINOv2를 사용할 수 있습니다.

---

## 스코어링 옵션 상세 가이드

### 점수 계산 공식

최종 점수는 활성화된 스코어러의 가중 합산으로 계산됩니다:

```
combined = emotion × W_emotion + aesthetic_norm × W_aesthetic + face × W_face + consistency × W_consistency
```

- 모든 가중치의 합은 항상 **1.0** (GUI가 자동 조정)
- Negative tag penalty: `combined *= (1 - neg_score)` (해부학 결함 감점)
- Consistency Weighted 모드 penalty: consistency가 gate threshold 미만이면 추가 감점

---

### 1. Emotion Score (기본, 항상 활성)

Camie Tagger v2가 예측한 감정 태그 확률값. 파일명의 감정 키워드와 일치하는 태그를 찾아 점수화합니다.

| 항목 | 값 |
|------|-----|
| 범위 | 0.0 ~ 1.0 |
| 단독 사용 시 weight | 1.0 |

EXIF에 캐릭터 태그가 있으면 감정 점수와 50:50으로 혼합되어 정확도가 향상됩니다.

---

### 2. Aesthetic Score (선택)

SigLIP 기반 Aesthetic Predictor V2.5의 시각적 품질 점수.

| 항목 | 값 |
|------|-----|
| Raw 범위 | 1.0 ~ 10.0 |
| 정규화 | `(raw - 1.0) / 9.0` → 0.0 ~ 1.0 |
| Min Quality (기본) | 3.0 |

**Min Quality (Quality Floor)**

이 값 미만의 이미지는 **모든 감정 그룹에서 전역 제거**됩니다. 가중치 합산 이전에 적용되는 hard cut입니다.

| 설정값 | 용도 |
|--------|------|
| **1.0** | 필터링 없음 (모든 이미지 통과) |
| **3.0** (기본) | 명백한 저품질만 제거. 대부분의 AI 생성 이미지는 통과 |
| **4.0 ~ 5.0** | 중간 품질 이상만 허용. 후보 수가 충분할 때 사용 |
| **6.0+** | 매우 엄격. 후보가 많지 않으면 감정별 이미지가 부족해질 수 있음 |

> 💡 NovelAI 이미지는 대체로 4.0~7.0 범위에 분포합니다. 3.0이면 거의 탈락이 없고 5.0이면 약 30~40%가 제거됩니다.

---

### 3. Face Framing (선택)

MediaPipe 얼굴 인식으로 표정이 잘 보이는지 평가합니다.

| 항목 | 값 |
|------|-----|
| 범위 | 0.0 ~ 1.0 |
| 기본 threshold | 0.30 |

**모드 선택**

| 모드 | 동작 | 적합한 상황 |
|------|------|-------------|
| **Hard Filter** | threshold 미만 이미지를 완전 제거 (combined에 미포함) | 얼굴이 반드시 보여야 하는 캐릭터 시트 |
| **Weighted** | 점수를 combined에 가중 합산 | 얼굴이 없어도 좋은 구도면 허용하고 싶을 때 |

**Threshold 설정 (Hard Filter 모드)**

| 값 | 의미 |
|----|------|
| **0.10 ~ 0.20** | 매우 느슨함. 얼굴이 아주 작거나 옆모습이라도 통과 |
| **0.30** (기본) | 적당함. 정면~3/4 각도에서 얼굴이 인식 가능하면 통과 |
| **0.50+** | 엄격. 정면 근접 얼굴만 통과. 전신/뒷모습 대부분 탈락 |

> 💡 감정 표현 평가가 목적이면 0.25~0.35 정도가 적합합니다. 전신 포즈가 많은 데이터셋이면 0.15~0.25로 낮추세요.

---

### 4. Reference Consistency (선택)

DINOv2로 참조 이미지 1~5장과의 시각적 유사도를 평가합니다. 캐릭터 디자인(의상, 체형, 색감)이 참조와 얼마나 일치하는지를 측정합니다.

| 항목 | 값 |
|------|-----|
| Raw 범위 | cosine similarity (-1.0 ~ 1.0, 보통 0.3~0.8) |
| 정규화 | 10th~90th percentile 기반, 0.0 ~ 1.0으로 스케일링 |
| 기본 threshold (Hard Filter) | 0.60 |

**모드 선택**

| 모드 | 동작 | 적합한 상황 |
|------|------|-------------|
| **Hard Filter** | 정규화 점수가 threshold 미만이면 완전 제거 (combined에 미포함) | 참조와 확실히 다른 이미지만 걸러내고 싶을 때 |
| **Weighted** | 점수를 combined에 가중 합산 + gate penalty 적용 | 유사도를 부드럽게 반영하고 싶을 때 |

**Threshold 설정 (Hard Filter 모드)**

| 값 | 의미 |
|----|------|
| **0.30 ~ 0.40** | 느슨함. 확실히 다른 캐릭터만 제거 |
| **0.50 ~ 0.60** (기본) | 적당함. 의상/디자인이 비슷한 이미지만 통과 |
| **0.70+** | 엄격. 참조와 매우 유사한 이미지만 통과. 포즈 변화도 감점됨 |

> ⚠️ Consistency 점수는 percentile 기반 정규화를 사용하므로, 동일 데이터셋 내에서의 상대 점수입니다. 절대값이 아닌 "이 데이터셋 내에서 참조에 가까운 상위 몇 %인가"로 해석하세요.

**Weighted 모드의 Gate Penalty**

Weighted 모드에서 consistency 점수가 gate threshold 미만이면 추가 penalty가 적용됩니다:

```
penalty = (consistency_score / gate_threshold) ^ penalty_power
combined *= penalty
```

- Gate Threshold (기본 0.60): 이 값 미만이면 penalty 발동
- Penalty Power (기본 3.0): 지수가 높을수록 급격한 감점

| consistency | penalty (threshold=0.60, power=3.0) | 해석 |
|-------------|--------------------------------------|------|
| 0.60+ | 1.0 (감점 없음) | 참조와 충분히 유사 |
| 0.45 | 0.42 | combined의 58% 감소 |
| 0.30 | 0.125 | combined의 87.5% 감소 (사실상 탈락) |
| 0.15 | 0.016 | 거의 0점 |

> 💡 power=3.0은 매우 공격적입니다. gate threshold 이하의 이미지는 사실상 선택되지 않습니다. 부드러운 감점을 원하면 threshold를 낮추거나 power를 줄이세요 (현재 GUI에서 power는 조절 불가, 코드에서 3.0 고정).

---

### 5. Tag Deviation Auto-Exclude (자동)

Reference Consistency를 활성화하면 자동으로 작동합니다.
참조 이미지에는 없는 시각적 특징(안경, 귀걸이, 머리 색 변경 등)이 후보 이미지에서 감지되면 자동 제외합니다.

| 조건 | 값 |
|------|-----|
| 후보 태그 확률 ≥ | 0.70 |
| 참조 대비 편차 ≥ | 0.30 |

두 조건 모두 만족해야 제외됩니다. 별도 설정은 없으며 참조 이미지만 잘 선택하면 됩니다.

---

### 6. Exclude Tags (수동)

사용자 지정 태그 목록 (쉼표 구분). 해당 태그가 감지되면 이미지를 제거합니다.

```
glasses, red_eyes, multiple_girls
```

- Camie Tagger가 해당 태그를 threshold(0.5) 이상으로 감지해야 제거
- EXIF 메타데이터에 해당 태그가 포함된 경우에는 **의도된 것**으로 간주하여 제거하지 않음

---

### 7. 가중치 (Weights) 설정

2개 이상의 스코어러가 활성화되면 가중치 조절 UI가 나타납니다. 하나를 변경하면 나머지가 자동으로 비례 조정되어 합계 1.0을 유지합니다.

**프리셋 기본값**

| 활성 조합 | Emotion | Aesthetic | Face | Consistency |
|-----------|---------|-----------|------|-------------|
| Emotion only | 1.00 | — | — | — |
| + Aesthetic | 0.65 | 0.35 | — | — |
| + Face (Weighted) | 0.80 | — | 0.20 | — |
| + Consistency (Weighted) | 0.60 | — | — | 0.40 |
| + Aesthetic + Face | 0.55 | 0.30 | 0.15 | — |
| + Aesthetic + Consistency | 0.50 | 0.25 | — | 0.25 |
| + Face + Consistency | 0.60 | — | 0.15 | 0.25 |
| **전체 (4개 모두)** | **0.45** | **0.25** | **0.10** | **0.20** |

> Hard Filter 모드의 스코어러(Face/Consistency)는 combined 가중치에 포함되지 않습니다. 해당 스코어러는 오직 합격/불합격 필터로만 동작합니다.

---

### 8. 상황별 추천 설정

#### 🎯 캐릭터 감정 시트 (기본 용도)

특정 캐릭터의 감정 표현 다양성을 최대화하면서 품질을 보장하고 싶을 때.

| 옵션 | 값 |
|------|-----|
| Aesthetic | ✅ 활성, Min Quality = **3.0** |
| Face | ✅ Hard Filter, Threshold = **0.25** |
| Consistency | ❌ 비활성 (단일 캐릭터라 불필요) |
| Weights | Emotion **0.65**, Aesthetic **0.35** |

#### 🎨 디자인 일관성이 중요한 캐릭터 시트

여러 감정으로 생성했지만 어떤 이미지는 의상/머리 색이 변하는 경우.

| 옵션 | 값 |
|------|-----|
| Aesthetic | ✅ 활성, Min Quality = **3.0** |
| Face | ✅ Hard Filter, Threshold = **0.30** |
| Consistency | ✅ Hard Filter, Threshold = **0.50** |
| Weights | Emotion **0.65**, Aesthetic **0.35** |

> Consistency를 Hard Filter로 사용하면 디자인이 다른 이미지를 먼저 제거하고, Emotion+Aesthetic으로만 순위를 매깁니다. 간단하고 직관적입니다.

#### 🖌️ 부드러운 유사도 반영 (디자인 변형도 허용)

디자인이 약간 다른 이미지도 점수를 깎되 완전히 제거하지는 않고 싶을 때.

| 옵션 | 값 |
|------|-----|
| Aesthetic | ✅ 활성, Min Quality = **3.0** |
| Face | ✅ Weighted |
| Consistency | ✅ Weighted, Threshold = **0.40** |
| Weights | Emotion **0.45**, Aesthetic **0.25**, Face **0.10**, Consistency **0.20** |

> Gate Threshold를 0.40으로 낮추면 penalty가 완화됩니다. 참조와 다소 다른 이미지도 감정 점수가 높으면 살아남을 수 있습니다.

#### 📷 품질 우선 (감정 정확도보다 비주얼 중심)

감정 태그 매칭보다 전반적인 이미지 퀄리티를 우선시하고 싶을 때.

| 옵션 | 값 |
|------|-----|
| Aesthetic | ✅ 활성, Min Quality = **5.0** |
| Face | ❌ 비활성 |
| Consistency | ❌ 비활성 |
| Weights | Emotion **0.40**, Aesthetic **0.60** |

> Min Quality를 높이면 저품질을 먼저 제거하고, Aesthetic weight를 높이면 남은 이미지 중 비주얼이 좋은 것을 우선합니다.

#### 🔬 최소 필터링 (거의 모든 이미지 유지)

생성 결과를 최대한 보존하면서 최악만 걸러내고 싶을 때.

| 옵션 | 값 |
|------|-----|
| Aesthetic | ✅ 활성, Min Quality = **1.0** |
| Face | ❌ 비활성 |
| Consistency | ❌ 비활성 |
| Weights | Emotion **0.85**, Aesthetic **0.15** |
| Top N | 높게 설정 (20~50) |

---

### 9. Fallback 동작

- **Emotion 태그 매칭 실패**: 모든 이미지의 감정 점수가 0이면, 보조 스코어러(Aesthetic/Face/Consistency)의 가중치를 자동 재분배하여 합계 1.0으로 만든 뒤 순위를 결정합니다.
- **Hard Filter가 모든 후보를 제거한 경우**: 해당 감정에서 가장 점수 높은 이미지 1장을 fallback으로 강제 보존합니다. 리포트에 `hard_filter_fallback_used` 플래그가 표시됩니다.
- **Face 점수 누락**: 기본값 0.3 (= hard filter threshold와 동일, 간신히 통과)
- **Consistency 점수 누락**: 기본값 0.0 (hard filter에서 탈락)

---

### 자동 다운로드 모델 저장 위치

`python src/main.py`로 실행하든 `AssetFilter.exe`로 실행하든, 자동 다운로드 모델은 모두 실행 기준 폴더 아래 `models` 폴더로 통일됩니다.

예시:

- Camie Tagger: `models/huggingface/hub/...`
- Aesthetic Predictor 관련 torch 캐시: `models/torch/...`
- Hugging Face 캐시: `models/huggingface/...`
- DINOv2 Reference Consistency: `models/huggingface/...`

출력 폴더는 입력 폴더 옆에 `{폴더명}_filtered`로 생성됩니다.

---

## 의존성

| 패키지 | 용도 |
|--------|------|
| `onnxruntime-directml` | Camie Tagger v2 ONNX 추론 (GPU/CPU) |
| `huggingface_hub` | 모델 자동 다운로드 |
| `Pillow` | 이미지 로드 및 전처리 |
| `numpy` | 배열 연산 |
| `torch` (CUDA) | Aesthetic Predictor V2.5 추론 (CUDA/CPU 자동 전환) |
| `aesthetic-predictor-v2-5` | SigLIP 기반 미학 점수 모델 |
| `transformers` | SigLIP 모델 로딩 |
| `mediapipe` | 얼굴 인식 |

---

## 라이선스

GPL-3.0-only — [LICENSE](LICENSE) 참조
