# Asset Filter

NovelAI로 대량 생성한 캐릭터 감정 이미지를 자동으로 필터링하는 Windows GUI 앱.

**Camie Tagger v2** + **Aesthetic Predictor V2.5** + **MediaPipe Face Detection**을 조합해 감정 키워드 매칭, 시각적 품질, 얼굴 프레이밍을 동시에 평가하고 상위 N장을 자동으로 출력 폴더로 복사합니다.

---

## 기능

- **Camie Tagger v2** — 70k+ Danbooru 태그로 정밀한 감정·결함 분류 (512px, Micro F1 67.3%)
- **Aesthetic Predictor V2.5** — SigLIP 기반 시각적 품질 평가 (1-10 스케일, 일러스트/애니메 특화)
- **Face Framing** — MediaPipe 얼굴 인식으로 표정 가시성 평가 (Hard Filter / Weighted 모드)
- **다차원 스코어링** — 감정 + 미학 + 얼굴 가중치를 조합한 통합 랭킹
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
| `AssetFilter-v1.0.0-cpu.zip` | 임베디드 Python + CPU torch 포함 (설치 불필요) |
| `AssetFilter-v1.0.0-cuda-split.*` | 임베디드 Python + CUDA torch 포함 (분할 압축) |

1. ZIP 압축 해제
2. `run.bat` 또는 `AssetFilter.exe` 실행
3. 첫 실행 시 모델 자동 다운로드 (Camie Tagger ~800MB, Aesthetic Predictor ~3.5GB)

> **AssetFilter.exe** — CPU 전용 독립 실행 파일. Python 환경 없이 단독 실행 가능.
> **run.bat** — 임베디드 Python 환경으로 실행. CUDA ZIP을 사용하면 GPU 가속됩니다.

### 소스에서 실행 (개발용)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python src/main.py
```

### 빌드 (독립 실행 파일)

```bash
pip install pyinstaller
build.bat
```

`dist/AssetFilter.exe`가 생성됩니다.

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
   - **Weights** — 감정/미학/얼굴 가중치를 자유롭게 조절 (합계 1.0 자동 유지)
4. **Run Filter** — 백그라운드에서 스코어링 후 출력 폴더에 복사합니다.
5. 완료 후 **Open Output**으로 결과 폴더를 엽니다.

출력 폴더는 입력 폴더 옆에 `{폴더명}_filtered`로 생성됩니다.

---

## 의존성

| 패키지 | 용도 |
|--------|------|
| `onnxruntime-directml` | Camie Tagger v2 ONNX 추론 (GPU/CPU) |
| `huggingface_hub` | 모델 자동 다운로드 |
| `Pillow` | 이미지 로드 및 전처리 |
| `numpy` | 배열 연산 |
| `torch` (CPU) | Aesthetic Predictor V2.5 추론 |
| `aesthetic-predictor-v2-5` | SigLIP 기반 미학 점수 모델 |
| `transformers` | SigLIP 모델 로딩 |
| `mediapipe` | 얼굴 인식 |

---

## 라이선스

GPL-3.0-only — [LICENSE](LICENSE) 참조
