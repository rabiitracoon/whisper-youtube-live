# Lecture Scribe

NVIDIA GPU로 YouTube 강의를 로컬 전사하고, ChatGPT Plus의 Codex OAuth 세션으로 학습용 Markdown 노트를 만드는 Windows 앱입니다. 앱 자체와 전사 도구는 무료·오픈 소스이며 OpenAI API 키를 요구하지 않습니다.

## 제공 기능

- YouTube 링크 하나로 영상 가져오기 → 쉬는 시간 정리 → 강의 글 확인 → 내 노트 완성
- 선택 시 최대 1080p 영상을 받아 쉬는 시간을 제거한 `edited_lecture.mp4` 보관
- 영상의 시작·종료 시간을 지정해 필요한 강의 부분만 글로 변환
- 파형을 클릭하는 **자르기** 도구로 쉬는 시간의 시작·끝을 나누고 구간 단위로 제외
- **쉬는 시간 자동 찾기** 버튼으로 수업 시작 전·중간 쉬는 시간·종료 후의 비음성/음악 구간을 표시
- `HH:MM:SS.mmm` 직접 입력으로 현재 위치를 정확히 옮기고 그 위치에서 자르기
- 현재 위치 클릭 이동, 선택 구간 듣기, 일시정지와 음량 조절로 배경음악 여부 확인
- 파형 위 마우스 휠로 커서 중심 확대·축소, `Shift + 휠`로 좌우 탐색
- 단계별 `checkpoint.json` 저장으로 중단 후 다운로드·편집·전사·노트 단계부터 재개
- 기본 자동 모드와 네 단계를 각각 실행하는 고급 단계 모드
- 언어 신뢰도·전사량·반복·음성 밀도 자동 점검과 전체 전사문 미리보기
- 사용자가 전사 결과를 승인한 뒤에만 GPT 실행
- 정확도 우선 기본값: `faster-whisper`의 `large-v3`, CUDA FP16, beam size 5
- 결과물: `transcript.md`, `transcript.txt`, `transcript.srt`, `lecture_notes.md`
- GPT-5.6 Sol + high reasoning 기본값 (`gpt-5.6-sol`)
- ChatGPT Plus 계정의 공식 Codex 브라우저 OAuth 사용
- 앱 안에서 노트 프롬프트 편집·저장·기본값 복원
- 버튼 한 번으로 yt-dlp 최신 버전 확인 및 업데이트
- Chrome/Edge/Firefox 쿠키를 선택적으로 사용해 로그인이 필요한 영상 처리
- 다운로드, 전사, 노트 생성 진행률과 로그 표시
- 쉬는 시간 편집·강의 글 승인처럼 사용자 확인이 필요할 때 작업표시줄 점멸 및 Windows 알림

## 빠른 시작

### 1. 설치

[`install.bat`](./install.bat)을 더블 클릭합니다. 설치기는 다음 작업을 자동으로 수행합니다.

1. 호환 Python 3.11 이상이 없으면 Python 3.12를 `winget`으로 사용자 범위에 설치
2. 프로젝트 전용 `.venv` 생성
3. RTX GPU용 CUDA 12.8 PyTorch 런타임 설치
4. PySide6, faster-whisper, yt-dlp와 영상 편집용 FFmpeg 런타임 설치
5. 공식 Codex CLI를 프로젝트 안에 설치
6. GPU 동작 검사 및 바탕 화면 바로가기 생성

> 설치에는 수 GB의 여유 공간과 인터넷 연결이 필요합니다. Whisper `large-v3` 모델(약 3GB)은 첫 전사 때 한 번 더 내려받습니다.

### 2. 실행 및 OAuth 로그인

[`run.bat`](./run.bat) 또는 바탕 화면의 **Lecture Scribe** 바로가기를 실행합니다.

1. 왼쪽 사이드바에서 **연결과 업데이트**를 엽니다.
2. **ChatGPT 연결하기**를 누릅니다.
3. 열린 브라우저에서 Plus 계정으로 로그인합니다.
4. 상태가 `ChatGPT 연결됨`으로 바뀌면 준비가 끝납니다.

앱은 OAuth 토큰을 직접 읽거나 저장하지 않습니다. 공식 Codex CLI의 자격 증명 저장소와 자동 갱신을 그대로 사용합니다. API 키로 로그인하면 Plus 포함 사용량이 아니라 API 종량제 과금이 적용되므로 이 앱에서는 ChatGPT 로그인을 사용하세요.

### 3. 강의 노트 만들기

1. 왼쪽 사이드바의 **새 강의 노트**에서 단일 YouTube 영상 링크를 붙여넣습니다.
2. 저장할 곳을 확인하고 **강의 노트 만들기**를 누릅니다. 영상 파일도 필요하면 **쉬는 시간을 뺀 영상도 보관하기**를 켜세요. 정확한 시작·종료 시간을 이미 아는 경우에만 **영상의 일부만 가져오기**를 켜세요.
3. 영상을 가져오면 전체 파형 편집 창이 열립니다.
4. **쉬는 시간 자동 찾기**를 누르면 시작 전·쉬는 시간·종료 후의 음악 구간을 찾아 표시합니다. 표시된 구간을 들어보고, 잘못 찾았다면 **자동 찾기 되돌리기** 또는 **선택 구간 다시 넣기**를 사용합니다.
5. 직접 편집하려면 **자르기(C)** 도구로 쉬는 시간 시작과 끝을 한 번씩 클릭합니다. 파형 위에서 휠을 굴리면 커서 위치를 중심으로 확대·축소되고, `Shift + 휠`은 좌우로 이동합니다. 정확한 위치는 `HH:MM:SS.mmm` 입력 후 **찾기 → 여기서 자르기**를 사용합니다.
6. **구간 선택(V)** 도구로 잘린 가운데 구간을 클릭하고 실제로 들어본 뒤 **선택 구간 빼기**를 누릅니다.
7. **이대로 오디오 정리하기**를 누르면 포함된 강의 부분만 이어 붙인 `edited_lecture.flac`를 만든 뒤 그 파일만 글로 변환합니다. 영상 보관 옵션을 켰다면 같은 구간을 적용한 `edited_lecture.mp4`도 만듭니다.
8. 강의 글이 완성되면 자동 확인 결과와 전체 내용을 살펴봅니다.
9. 이상이 없을 때 **확인했어요 · 노트 만들기**를 누릅니다. 이 확인 전에는 ChatGPT가 실행되지 않습니다.
10. 완료 후 노트·강의 글·저장 폴더 버튼으로 파일을 엽니다.

### 고급 단계 모드와 이어하기

**단계별로 직접 진행하기**를 켜면 같은 링크로 가장 최근에 끝낸 작업을 찾아 다음 단계를 각각 실행할 수 있습니다.

1. **영상 가져오기** — 원본 오디오와 선택한 원본 영상을 저장하거나 기존 파일 재사용
2. **쉬는 시간 정리** — 파형 편집 후 FLAC 및 선택한 MP4 편집본 생성
3. **강의 글 만들기** — 정리한 오디오만 글로 옮기고 확인 결과 저장
4. **노트 정리하기** — 확인된 강의 글로 노트만 생성

기본 모드는 이미 완료된 체크포인트를 자동으로 건너뛰고 남은 단계부터 끝까지 진행합니다. 편집본 생성 중 중단되면 `.partial.flac`과 `.partial.mp4`를 정리하며, 원본과 완료된 단계 결과는 유지합니다.

첫 실행은 모델 다운로드 때문에 오래 걸립니다. 그 다음부터는 로컬 캐시를 재사용합니다.

## 출력 구조

```text
outputs/
└── 20260720_101530_VIDEOID_영상 제목/
    ├── metadata.json
    ├── checkpoint.json
    ├── prompt_used.md
    ├── edited_lecture.flac
    ├── edited_lecture.mp4       # 영상 보관 옵션 사용 시
    ├── transcript_review.json
    ├── transcript.txt
    ├── transcript.md
    ├── transcript.srt
    └── lecture_notes.md
```

오디오 보관 옵션을 켜면 원본 오디오 스트림도 같은 폴더에 남습니다. 기본값에서는 최종 노트 생성 후 원본만 삭제하며, 재전사에 사용하는 `edited_lecture.flac`는 유지합니다.

영상 보관 옵션을 켜면 최대 1080p 영상을 받아 Razor에서 제외한 쉬는 시간을 동일하게 제거한 H.264/AAC MP4를 보관합니다. 편집에 사용한 원본 영상은 전체 작업 완료 후 정리됩니다. 영상은 NVIDIA NVENC로만 인코딩하며 GPU 인코더를 사용할 수 없으면 느린 CPU 인코딩으로 대체하지 않고 오류를 표시합니다.

Windows 미디어 미리듣기나 다른 프로그램이 원본 파일을 잠근 경우에는 삭제를 잠시 재시도합니다. 잠금이 계속되더라도 완료된 전사와 노트를 오류 처리하지 않고 원본 파일만 결과 폴더에 보존합니다.

## 노트 작성 방식 수정

왼쪽 사이드바의 **노트 작성 방식**에서 바로 수정할 수 있습니다. 사용 중인 파일은 [`prompts/lecture_notes_ko.md`](./prompts/lecture_notes_ko.md)이고, 복원용 원본은 [`prompts/default_lecture_notes_ko.md`](./prompts/default_lecture_notes_ko.md)입니다.

사용 가능한 자리표시자:

| 자리표시자 | 값 |
|---|---|
| `{{title}}` | 영상 제목 |
| `{{channel}}` | 채널 또는 강사 |
| `{{url}}` | 원본 URL |
| `{{duration}}` | 영상 길이 |
| `{{language}}` | Whisper 감지 언어와 신뢰도 |
| `{{transcript}}` | 타임스탬프가 없는 일반 전사문(필수) |

기본 프롬프트는 공개 강의 정리 사례를 참고하되 영상 요약 보고서가 아니라 실제 대학생의 수업 필기에 가깝게 조정했습니다.

- 내용에 맞춰 직접 만든 자연스러운 소제목
- 간결한 문장, 계층형 글머리표, `→`와 대비 표기
- 정의뿐 아니라 인과관계·조건·비교·단계·강의 사례 보존
- 수식·코드·수치·강사의 주의사항 유지
- 영상 정보·타임스탬프·도입 요약·퀴즈·표 형식 제거
- 전사문 밖의 내용을 강의 사실처럼 만들지 않는 근거 제약

참고한 공개 사례: [YouTube Summarisation prompt](https://gist.github.com/strickvl/ac2a6a6e6f642ed6be375bd1943bd65f), [GPTStore AlphaNotes/YT transcriber 프롬프트 모음](https://github.com/1003715231/gptstore-prompts), [Skill-Anything의 12-section study guide](https://github.com/SYuan03/Skill-Anything).

## yt-dlp 업데이트

왼쪽 사이드바의 **연결과 업데이트**에서 **한 번에 확인하고 업데이트**를 누르면 현재 버전과 최신 버전을 비교하고, 새 버전이 있을 때 같은 가상환경에 업데이트합니다.

- `새 기능 빠르게 받기 · 권장`: YouTube 변경 대응이 가장 빠른 공식 권장 채널
- `검증된 버전만 받기`: 변경이 적은 안정 채널

업데이트 중에는 로그가 표시됩니다. 실행 중 이미 yt-dlp를 사용했다면 앱을 한 번 재시작하는 것이 안전합니다. 자세한 채널 정책은 [yt-dlp 공식 업데이트 안내](https://github.com/yt-dlp/yt-dlp#update)를 참고하세요.

## 정확도와 GPU 설정

현재 확인된 PC 사양인 RTX 5060 Ti 16GB에 맞춰 `large-v3 + float16`을 기본으로 설정했습니다. 속도보다 정확도를 우선해 beam search를 사용합니다.

- `large-v3`: 가장 높은 정확도 권장
- `large-v2`: 특정 언어에서 비교용
- `medium`: 메모리·다운로드 공간 절약용
- Beam 5: 기본 정확도 우선값. 높이면 더 느려질 수 있으며 항상 더 정확한 것은 아닙니다.
- 긴 무음 제거: 무음 환각을 줄입니다. 발화가 잘리는 특수 음원에서는 끌 수 있습니다.
- 구간 전사: 원본 영상 시간 기준으로 지정한 부분만 GPU가 처리합니다. 구간 지정은 faster-whisper 특성상 긴 무음 제거(VAD)보다 우선합니다.
- 자동 인식: 인터넷 연결이나 유료 API 없이 bundled Silero VAD를 CPU에서 실행합니다. 15분 단위 스트리밍 분석으로 장시간 영상의 메모리 사용을 제한합니다. 일반적인 말 사이 공백은 보호하기 위해 중간 구간은 45초 기준을 유지하고, 짧은 인트로·아웃트로 음악을 놓치지 않도록 시작과 종료 경계에만 5초 기준을 적용합니다.
- Razor 편집: 1–24배 확대를 지원합니다. 일반 휠은 커서 중심 확대·축소, `Shift + 휠`은 좌우 탐색입니다. `C`는 Razor, `V`는 클립 선택, `P`는 재생헤드 도구입니다. 주황선은 컷, 붉은 클립은 전사 제외를 뜻합니다.
- 정확한 이동: `HH:MM:SS.mmm` 형식으로 입력해 밀리초 단위 재생헤드 이동과 자르기가 가능합니다.
- 키보드 조작: 파형에 초점을 둔 상태에서 좌우 방향키로 5초, Shift+방향키로 30초 이동하고 Space로 재생·일시정지할 수 있습니다.

Whisper는 원본 장시간 파일과 여러 타임스탬프를 직접 처리하지 않습니다. 쉬는 시간이 제거된 연속 16 kHz mono 무손실 FLAC만 읽으며, VAD를 다시 적용하고 이전 구간 텍스트 조건화를 끄기 때문에 배경음악 구간의 반복 문장 환각과 불필요한 처리 시간이 줄어듭니다.

전체 오디오 스트림 다운로드는 대체로 빠르고, 계산 시간이 큰 Whisper 전사만 선택 구간으로 제한됩니다. 전사 후 자동 점검은 참고 신호이며, GPT 실행 전 미리보기에서 전문용어와 강의 시작·끝 부분을 직접 확인하는 것이 가장 안전합니다.

`faster-whisper`는 시스템 FFmpeg 설치 없이 PyAV로 오디오를 읽습니다. GPU 요구 사항은 [faster-whisper 공식 README](https://github.com/SYSTRAN/faster-whisper#gpu)를 참고하세요.

## ChatGPT Plus와 비용

- 로컬 Whisper 전사, yt-dlp, 앱 코드는 무료입니다.
- 노트 생성은 이미 보유한 ChatGPT Plus의 Codex 포함 사용량을 이용하므로 별도 API 과금이 없습니다.
- Plus 요금제 자체의 구독료와 Codex 사용량 제한은 그대로 적용됩니다.
- 공식 문서에 따라 ChatGPT 로그인과 API 키 로그인의 과금 체계는 다릅니다. [Codex 인증 안내](https://learn.chatgpt.com/docs/auth), [Codex CLI README](https://github.com/openai/codex#using-codex-with-your-chatgpt-plan)를 참고하세요.
- 기본 모델인 GPT-5.6 Sol은 현재 공식 모델 카탈로그의 복잡한 작업용 최상위 모델입니다. [GPT-5.6 Sol 모델 문서](https://developers.openai.com/api/docs/models/gpt-5.6-sol)

## 문제 해결

### `Codex CLI가 없습니다`

`install.bat`을 다시 실행하세요. 이 프로젝트의 `tools/codex-cli` 아래에 최신 CLI를 설치합니다.

### OAuth 브라우저가 열리지 않거나 로그인이 끝나지 않음

도구 로그를 확인한 뒤 앱을 다시 시작해 로그인하세요. 회사 네트워크가 localhost OAuth 콜백을 차단하면 터미널에서 로컬 Codex CLI의 device auth를 사용할 수 있습니다.

```powershell
.\tools\codex-cli\node_modules\.bin\codex.cmd login --device-auth
```

### YouTube가 로그인을 요구하거나 봇 확인을 표시함

전사 설정의 **로그인이 필요한 영상**에서 현재 로그인된 브라우저 쿠키를 선택하세요. 브라우저 프로필의 쿠키를 읽으므로 본인에게 시청 권한이 있는 영상에만 사용하세요.

### CUDA 또는 cuDNN DLL 오류

`install.bat`을 다시 실행해 CUDA 12.8 PyTorch 런타임을 복구하세요. 설치 검사는 다음 명령으로 다시 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_install.py
```

### 전사 정확도가 낮음

- 언어를 자동이 아닌 실제 강의 언어로 고정
- Whisper 모델이 `large-v3`인지 확인
- 소리가 매우 작거나 배경음이 큰 영상은 원본 음질 확인
- 발화 일부가 사라지면 `긴 무음 구간 제거`를 끄고 재시도
- 쉬는 시간의 배경음악에서 문장이 반복되면 Razor로 앞뒤를 자르고 가운데 클립을 전사 제외한 뒤 편집본을 다시 저장

### 파형 편집에서 오디오가 재생되지 않음

편집 창의 상태가 `미리듣기 준비됨`으로 바뀐 뒤 재생하세요. 계속 실패하면 Windows의 기본 출력 장치와 앱별 음량을 확인하세요. 재생 오류가 있더라도 파형 구간 편집은 계속 사용할 수 있습니다.

## 개발 및 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q app.py lecture_scribe tests scripts
```

핵심 모듈:

- [`lecture_scribe/youtube.py`](./lecture_scribe/youtube.py): URL 검증, 메타데이터, 오디오 다운로드
- [`lecture_scribe/transcription.py`](./lecture_scribe/transcription.py): CUDA 준비, Whisper 전사, md/txt/srt 저장
- [`lecture_scribe/time_range.py`](./lecture_scribe/time_range.py): 전사 시작·종료 시간 해석과 검증
- [`lecture_scribe/audio_edit.py`](./lecture_scribe/audio_edit.py): 파형 추출, Razor 클립 계산, 16 kHz FLAC 편집본 저장
- [`lecture_scribe/ui/audio_editor.py`](./lecture_scribe/ui/audio_editor.py): Razor 클릭 편집, 정확한 재생헤드 이동과 미리듣기
- [`lecture_scribe/quality.py`](./lecture_scribe/quality.py): 전사 자동 품질 점검과 검토 데이터 생성
- [`lecture_scribe/codex_client.py`](./lecture_scribe/codex_client.py): OAuth 상태와 `codex exec` 노트 생성
- [`lecture_scribe/updater.py`](./lecture_scribe/updater.py): yt-dlp 버전 확인·업데이트
- [`lecture_scribe/pipeline.py`](./lecture_scribe/pipeline.py): 전체 파이프라인
- [`lecture_scribe/ui/main_window.py`](./lecture_scribe/ui/main_window.py): PySide6 데스크톱 UI

## 이용 시 주의

본인이 시청·다운로드·변환할 권한이 있는 영상에만 사용하세요. YouTube 서비스 약관, 저작권, 강의 제공자의 이용 조건을 준수할 책임은 사용자에게 있습니다. DRM 우회 기능은 포함하지 않습니다.

## 라이선스

[MIT License](./LICENSE)
