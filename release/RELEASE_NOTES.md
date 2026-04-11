# Asset Filter v1.1.2 Release Notes

## 버그 수정

### Weight 조정 시 보정 대상이 일관되지 않던 문제 수정

- **문제**: Scoring Weights에서 버튼으로 수치를 변경할 때, 합계 1.0을 맞추기 위해 변경되는 다른 weight의 위치가 매번 달랐음
- **원인**: 비례 분배(largest-remainder) 방식이 각 weight의 상대 비율에 따라 보정 대상을 매번 다르게 선택
- **수정**: 고정 우선순위 보정 방식으로 변경
  - Aesthetic/Face/Consistency 변경 시 → 항상 **Emotion**이 먼저 흡수
  - Emotion 변경 시 → 항상 **Aesthetic**이 먼저 흡수 (0이면 다음 순서로 전파)
  - 우선순위: Emotion > Aesthetic > Face > Consistency

---

### v1.1.1: Recovery Pass에서 tag_deviation 제외 이미지 복구 허용

- **문제**: `tag_deviation_filter`로 자동 제외된 이미지가 recovery pass에서도 복구 후보에서 제외되어, 부족한 이모션(top_n 미달)을 채울 수 없었음
- **원인**: recovery pass가 유저 지정 `exclude_tags`와 자동 생성 `tag_dev_excluded`를 동일하게 차단
- **수정**: recovery pass에서 `tag_dev_excluded` 제한을 해제. 유저 지정 `exclude_tags`만 계속 차단
  - tag_deviation은 참조 이미지 기반 자동 필터이므로, 부족분 복구 시 후보로 허용하는 것이 적절

---

## 포함 파일

| 파일 | 설명 |
|------|------|
| `AssetFilter-v1.1.2.z01` | 분할 압축 파트 1 (1.9 GB) |
| `AssetFilter-v1.1.2.zip` | 분할 압축 파트 2. 임베디드 Python + CUDA torch 포함 |
| `release-description.html` | GitHub Releases 페이지 설명 HTML |

- 두 파일을 모두 다운로드한 뒤 같은 폴더에 놓고 `.zip` 파일을 해제하여 `AssetFilter.exe` (또는 `run.bat`)로 실행하세요.
- NVIDIA GPU(드라이버 560+)가 있으면 CUDA, 없으면 CPU로 자동 전환됩니다.
- 첫 실행 시 모델이 자동 다운로드됩니다: Camie Tagger ~800 MB, Aesthetic Predictor ~3.5 GB, DINOv2 ~350 MB.
