# Lecture Scribe for Apple Silicon

YouTube 강의를 내려받아 Apple Silicon의 **MLX Metal GPU**로 로컬 전사하고, 선택적으로 ChatGPT Plus의 Codex OAuth 세션으로 학습용 Markdown 노트를 만드는 macOS 앱입니다. NVIDIA GPU, CUDA, OpenAI API 키는 필요하지 않습니다.

이 저장소는 [rabiitracoon/whisper-youtube-live](https://github.com/rabiitracoon/whisper-youtube-live)의 Windows/NVIDIA 버전을 Mac MLX 환경에 맞게 변환한 버전입니다.

## 지원 환경

- Apple Silicon Mac(M1/M2/M3/M4 계열, `arm64`)
- macOS 13 이상 권장
- Python 3.11 또는 3.12
- 인터넷 연결: YouTube 다운로드와 최초 Whisper 모델 다운로드 시 필요
- Node.js: AI 노트 기능을 사용할 때만 필요

Intel Mac에서는 MLX를 사용할 수 없습니다.

## 설치

Finder에서 [`install.command`](./install.command)를 더블클릭합니다. macOS가 처음 실행을 막으면 파일을 우클릭하고 **열기**를 선택하세요.

터미널에서는 다음처럼 실행할 수 있습니다.

```bash
./install.command
```

설치기는 프로젝트 전용 `.venv`를 만들고 다음 항목을 설치합니다.

- `mlx-whisper`: Apple Silicon Metal 기반 Whisper 전사
- `PySide6`: 데스크톱 UI
- `yt-dlp`: YouTube 오디오 다운로드
- `imageio-ffmpeg`: 쉬는 시간을 제거한 영상의 Apple 하드웨어 인코딩
- `faster-whisper`의 Silero VAD: 쉬는 시간 자동 감지에만 CPU로 사용
- `@openai/codex`: AI 노트 기능(Node.js가 있을 때)

Python이 없다면 Homebrew 설치 후 아래 명령을 먼저 실행하세요.

```bash
brew install python@3.12
```

AI 노트 기능까지 사용할 경우 Node.js도 설치합니다.

```bash
brew install node
```

## 실행

Finder에서 [`run.command`](./run.command)를 더블클릭하거나 터미널에서 실행합니다.

```bash
./run.command
```

가상환경이 없으면 실행 스크립트가 설치를 먼저 시작합니다.

## 사용 순서

1. **새 강의 노트**에서 단일 YouTube 영상 URL을 입력합니다.
2. 필요하면 언어, Whisper 모델, 영상 시작·종료 구간을 지정합니다. 전문용어 보정이 필요하면 **전문용어 자동 입력**을 켭니다.
3. 전문용어 자동 입력을 켰다면 시작할 때 강의명을 입력합니다. 연결된 GPT가 관련 용어를 만들고 Whisper 전사에 자동 적용합니다.
4. 영상을 가져온 뒤 파형 편집기에서 쉬는 시간을 자동으로 찾거나 직접 제외합니다.
5. `edited_lecture.flac`가 만들어지면 MLX Whisper가 로컬에서 전사합니다. 영상 보관 옵션을 켰다면 같은 구간으로 `edited_lecture.mp4`도 만듭니다.
6. 전사 내용을 확인합니다.
7. ChatGPT 연결을 한 경우 **확인했어요 · 노트 만들기**로 Markdown 노트를 생성합니다.

전문용어 자동 입력을 끄고 전사만 사용할 때는 Node.js, Codex 로그인, ChatGPT 구독이 필요하지 않습니다. 전문용어 자동 입력에는 도구 탭에서 연결한 ChatGPT 로그인이 필요합니다.

## MLX 모델

UI의 기존 모델 이름은 아래 MLX 모델로 자동 연결됩니다.

| UI 표시 | 실제 MLX 모델 |
|---|---|
| `large-v3` | `mlx-community/whisper-large-v3-mlx` |
| `large-v2` | `mlx-community/whisper-large-v2-mlx` |
| `medium` | `mlx-community/whisper-medium-mlx` |

기본값은 정확도 우선 `large-v3`입니다. 모델은 처음 전사할 때 Hugging Face에서 내려받고 이후 로컬 캐시를 재사용합니다. 8GB 메모리 Mac에서 메모리 부족이 발생하면 `medium`을 사용하거나 다른 앱을 종료하세요.

이미 받은 `large-v3` MLX 모델이 있다면 모델 폴더의 내용(`config.json`, `weights.npz` 또는 `weights.safetensors` 등)을 `models/whisper-large-v3-mlx/`에 복사하세요. 앱은 완전한 로컬 모델을 발견하면 Hugging Face 다운로드보다 우선 사용합니다.

## 주요 기능

- YouTube 오디오 다운로드와 로그인 브라우저 쿠키 지원
- 선택 시 최대 1080p 원본에서 쉬는 시간을 제거한 H.264/AAC MP4 저장
- Apple VideoToolbox 하드웨어 영상 인코딩(CPU 인코더로 자동 대체하지 않음)
- 시작·종료 시간 지정
- 파형 기반 쉬는 시간 제거와 자동 비음성 구간 감지
- 중단 후 `checkpoint.json` 단계별 재개
- MLX Whisper 전사 결과 자동 점검
- 강의명 기반 GPT 전문용어 생성 및 Whisper `initial_prompt` 자동 적용
- `transcript.txt`, `transcript.md`, `transcript.srt` 저장
- 선택적인 Codex OAuth 기반 `lecture_notes.md` 생성
- OpenAI API 키 및 API 종량제 호출 없음
- 쉬는 시간 편집과 전사 확인이 필요할 때 macOS 알림

## 출력 구조

```text
outputs/
└── 20260720_101530_VIDEOID_영상 제목/
    ├── metadata.json
    ├── checkpoint.json
    ├── edited_lecture.flac
    ├── edited_lecture.mp4       # 영상 보관 옵션 사용 시
    ├── transcription_terms.txt  # 전문용어 자동 입력 사용 시
    ├── transcript_review.json
    ├── transcript.txt
    ├── transcript.md
    ├── transcript.srt
    └── lecture_notes.md
```

영상 보관 옵션을 사용하면 원본 영상은 전체 작업 완료 후 정리되고 `edited_lecture.mp4`만 보존됩니다. 영상 인코딩에는 Apple VideoToolbox 하드웨어 가속을 사용합니다.

## ChatGPT 연결

AI 노트를 만들려면 앱의 **연결과 업데이트** 화면에서 **ChatGPT 연결하기**를 누르고 브라우저 로그인을 완료합니다. 앱은 OAuth 토큰을 직접 읽거나 저장하지 않고 공식 Codex CLI의 로그인 상태를 사용합니다.

전사 과정은 전부 Mac 안에서 처리됩니다. AI 노트 생성 단계에서만 확인한 전사문이 Codex에 전달됩니다.

## 점검과 테스트

```bash
.venv/bin/python scripts/verify_install.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q app.py lecture_scribe tests scripts
```

샌드박스나 원격 세션에서는 `No Metal device available`이 발생할 수 있습니다. Finder나 일반 Terminal.app에서 실행하세요.

## 문제 해결

### `MLX cannot access Metal`

Apple Silicon Mac인지 확인하고 일반 macOS 로그인 세션의 Terminal.app 또는 Finder에서 실행하세요. SSH, 가상 머신, 일부 샌드박스 환경에서는 Metal 장치에 접근할 수 없습니다.

### 첫 전사가 오래 걸림

`large-v3` 모델을 최초 한 번 다운로드하고 메모리에 올리는 시간입니다. 다음 전사부터 캐시를 재사용합니다.

### YouTube가 로그인을 요구함

앱 설정에서 현재 로그인된 Chrome/Edge/Firefox 쿠키를 선택하세요. 본인에게 시청 및 변환 권한이 있는 영상에만 사용하세요.

### Codex CLI가 없다고 표시됨

`brew install node` 실행 후 `./install.command`를 다시 실행하세요. 전사 기능에는 영향이 없습니다.

## 이용 시 주의

본인이 시청·다운로드·변환할 권한이 있는 영상에만 사용하세요. YouTube 서비스 약관, 저작권, 강의 제공자의 이용 조건을 준수할 책임은 사용자에게 있습니다. DRM 우회 기능은 포함하지 않습니다.

## 라이선스

[MIT License](./LICENSE)
