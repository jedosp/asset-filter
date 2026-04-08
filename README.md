# Asset Filter

NovelAI로 대량 생성한 캐릭터 감정 이미지를 자동으로 필터링하는 Windows GUI 앱.

**Camie Tagger v2** (70,527개 Danbooru 태그, ViT 기반)를 사용해 감정 키워드별로 점수를 매기고, 상위 N장을 자동으로 출력 폴더로 복사합니다.

---

## 기능

- **Camie Tagger v2** — 70k+ Danbooru 태그로 정밀한 감정·결함 분류 (512px, Micro F1 67.3%)
- **EXIF 태그 결합 스코어링** — NovelAI char_captions 메타데이터를 활용해 정확도 향상
- **해부학 결함 자동 감지** — bad_anatomy, extra_fingers 등 22개 결함 태그로 불량 이미지 자동 감점
- **DirectML GPU 가속** — Windows GPU 자동 감지, GPU 없으면 CPU 자동 전환
- 전체 이미지를 단일 패스로 추론 (감정 수가 많아도 속도 일정)
- 모델 다운로드 / 이미지 분석 단계별 프로그레스 바
- 필터링 결과를 `report.json`으로 저장

---

## 다운로드 및 실행

### 릴리즈 (일반 사용자)

1. [Releases](https://github.com/jedosp/asset-filter/releases) 페이지에서 ZIP 다운로드
2. 압축 해제
3. `AssetFilter.exe` 더블클릭
4. 첫 실행 시 Camie Tagger v2 모델 자동 다운로드 (~800MB)

### 소스에서 실행 (개발용)

```bash
pip install -r requirements.txt
python src/main.py
```

### 빌드

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
3. **Run Filter** — 백그라운드에서 스코어링 후 출력 폴더에 복사합니다.
4. 완료 후 **Open Output**으로 결과 폴더를 엽니다.

출력 폴더는 입력 폴더 옆에 `{폴더명}_filtered`로 생성됩니다.

---

## 의존성

| 패키지 | 용도 |
|--------|------|
| `onnxruntime-directml` | Camie Tagger v2 ONNX 추론 (GPU/CPU) |
| `huggingface_hub` | 모델 자동 다운로드 |
| `Pillow` | 이미지 로드 및 전처리 |
| `numpy` | 배열 연산 |
