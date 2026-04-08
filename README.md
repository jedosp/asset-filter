# Asset Filter

NovelAI로 대량 생성한 캐릭터 감정 이미지를 자동으로 필터링하는 Windows GUI 앱.

**WD Tagger v3** (Danbooru 기반 애니 이미지 분류 모델)를 사용해 감정 키워드별로 점수를 매기고, 상위 N장을 자동으로 출력 폴더로 복사합니다.

---

## 기능

- **WD Tagger v3** — NovelAI 감정 태그와 동일한 Danbooru 태그 체계를 사용해 정확한 감정 표현 분류
- 전체 이미지를 단일 패스로 추론 (감정 수가 많아도 속도 일정)
- 진행 상황 실시간 표시 (프로그레스 바)
- 필터링 결과를 `report.json`으로 저장

---

## 다운로드 및 실행

### 릴리즈 (일반 사용자)

1. [Releases](https://github.com/jedosp/asset-filter/releases) 페이지에서 ZIP 다운로드
2. 압축 해제
3. `AssetFilter.exe` 더블클릭
4. 첫 실행 시 WD Tagger 모델 자동 다운로드 (~350MB)

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
| `onnxruntime` | WD Tagger v3 ONNX 추론 |
| `huggingface_hub` | 모델 자동 다운로드 |
| `Pillow` | 이미지 로드 및 전처리 |
| `numpy` | 배열 연산 |
