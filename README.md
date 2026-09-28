# Lecture Scribe

YouTube 강의를 내려받아 GPU로 로컬 전사하고, 선택적으로 ChatGPT(Codex CLI) 또는 Claude(Claude Code CLI)의 OAuth 세션으로 학습용 Markdown 노트를 만드는 데스크톱 앱입니다. 하나의 코드베이스가 실행 환경을 감지해 Windows에서는 NVIDIA CUDA, Apple Silicon Mac에서는 MLX Metal을 사용합니다.

## 지원 환경

| 운영체제 | 음성 전사 | 편집 영상 인코딩 | 실행 파일 |
|---|---|---|---|
| Windows 10/11 + NVIDIA GPU | `faster-whisper` + CUDA FP16 | NVIDIA NVENC | `install.bat`, `run.bat` |
| Apple Silicon macOS | `mlx-whisper` + MLX Metal | Apple VideoToolbox | `install.command`, `run.command` |

Intel Mac, Linux, NVIDIA GPU가 없는 Windows는 현재 GPU 전사를 지원하지 않습니다.

## 주요 기능

- YouTube 링크 또는 내 컴퓨터의 영상·음성 파일로 가져오기 → 쉬는 시간 정리 → 전사 확인 → 학습 노트 생성
- 시작·종료 시간 지정과 파형 기반 Razor 편집
- Silero VAD를 이용한 긴 비음성 구간 자동 감지
- 쉬는 시간을 동일하게 제거한 H.264/AAC `edited_lecture.mp4` 선택 저장
- Windows NVIDIA NVENC 및 macOS Apple VideoToolbox 하드웨어 전용 인코딩
- `checkpoint.json`을 이용한 다운로드·편집·전사·노트 단계 재개
- 강의명 기반 AI 전문용어 생성 및 Whisper `initial_prompt` 자동 적용
- 강의용·주식방송용 등 원하는 개수의 프롬프트 슬롯 저장 및 작업별 선택
- `transcript.txt`, `transcript.md`, `transcript.srt`, `lecture_notes.md` 저장
- ChatGPT 또는 Claude 구독 계정 OAuth 사용, API 키와 API 종량제 호출 없음
- 연결된 계정에서 쓸 수 있는 최신 모델과 모델별 추론 강도를 목록에서 선택

## 설치 및 실행

### Windows + NVIDIA

1. [`install.bat`](./install.bat)을 실행합니다.
2. 설치가 끝나면 [`run.bat`](./run.bat)을 실행합니다.

설치기는 Python 가상환경, CUDA 12.8 PyTorch와 앱 의존성을 준비합니다. Node.js가 설치되어 있으면 Codex CLI와 Claude Code CLI도 함께 준비합니다. NVIDIA 드라이버가 최신 상태인지 확인하세요.

### Apple Silicon Mac

1. Finder에서 [`install.command`](./install.command)를 더블클릭합니다.
2. macOS가 실행을 막으면 파일을 우클릭하고 **열기**를 선택합니다.
3. 설치가 끝나면 [`run.command`](./run.command)를 실행합니다.

터미널에서는 다음과 같이 실행할 수 있습니다.

```bash
./install.command
./run.command
```

Python이 없다면 먼저 설치합니다.

```bash
brew install python@3.12
```

AI 노트나 전문용어 자동 입력을 사용할 경우 Node.js도 필요합니다.

```bash
brew install node
```

## 사용 순서

1. **새 강의 노트**에서 단일 YouTube URL을 입력하거나 **파일 선택**으로 영상·음성 파일을 고릅니다.
2. 필요하면 강의 구간, 언어, Whisper 모델을 설정합니다.
3. 전문용어 보정이 필요하면 **전문용어 자동 입력**을 켭니다.
4. 시작할 때 강의명을 입력하면 연결된 AI가 관련 용어를 생성합니다.
5. 파형 편집기에서 쉬는 시간을 자동으로 찾거나 직접 제외합니다.
6. 운영체제에 맞는 GPU 백엔드가 편집본을 전사합니다.
7. 전사 결과를 확인한 뒤 Markdown 노트를 만듭니다.

전문용어 자동 입력을 끄고 전사만 사용할 때는 AI 로그인과 구독이 필요하지 않습니다. 전문용어 자동 입력과 노트 생성에는 앱의 **연결과 업데이트** 화면에서 연결한 ChatGPT 또는 Claude 로그인이 필요합니다.

선택한 AI 서비스에는 강의명과 영상 제목이 전달됩니다. 음성과 Whisper 전사는 컴퓨터에서 로컬로 처리됩니다. 로컬 파일을 선택한 경우에도 원본 파일은 수정하거나 삭제하지 않고 작업 폴더에 복사해 사용합니다. 노트 생성 단계에서는 사용자가 확인한 전사문이 선택한 AI 서비스에 전달됩니다.

## AI 서비스와 모델

**노트 작성 방식** 화면에서 AI 서비스(ChatGPT 또는 Claude)를 고른 뒤 노트 작성 모델, 생각 깊이, 전문용어 생성 모델을 목록에서 선택합니다.

- ChatGPT: Codex CLI의 `codex debug models`로 현재 계정의 모델 목록과 모델별 지원 추론 강도를 불러옵니다.
- Claude: Claude Code CLI가 알려주는 현재 계정의 모델 목록(Opus, Fable, Sonnet, Haiku 등)과 지원 effort를 불러옵니다.
- CLI가 없거나 목록을 불러오지 못하면 내장된 기본 목록을 보여줍니다. **모델 목록 새로고침**으로 다시 불러올 수 있습니다.
- 서비스별로 마지막에 고른 모델이 따로 저장되므로 서비스를 바꿔도 이전 선택이 유지됩니다.

## Whisper 모델

UI에서는 두 운영체제 모두 `large-v3`, `large-v2`, `medium` 이름을 사용합니다.

- Windows: `faster-whisper`가 모델을 내려받아 CUDA로 실행합니다.
- macOS: 동일한 이름을 `mlx-community/whisper-*-mlx` 모델로 자동 변환합니다.

이미 받은 macOS `large-v3` MLX 모델이 있다면 모델 폴더 내용을 `models/whisper-large-v3-mlx/`에 복사하세요. `config.json`과 `weights.npz` 또는 `weights.safetensors`가 있으면 원격 다운로드보다 우선 사용합니다. 실제 모델 파일은 Git에 포함되지 않습니다.

## 전문용어 자동 입력

생성된 단어는 각 작업 폴더의 `transcription_terms.txt`에 저장되고 Whisper 초기 문맥에 적용됩니다. 같은 강의명으로 다시 실행하면 저장된 목록을 재사용해 AI를 중복 호출하지 않습니다.
노트 작성 설정에서 전문용어 생성 전용 모델을 별도로 지정할 수 있으며, 기본값은 ChatGPT `gpt-5.6-luna`, Claude `sonnet`입니다.

## 출력 구조

```text
outputs/
└── 작업 폴더/
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

영상 보관 옵션을 사용하면 원본 영상은 전체 작업 완료 후 정리되고 편집본만 보존됩니다. 하드웨어 영상 인코더를 사용할 수 없을 때는 느린 CPU 인코딩으로 자동 대체하지 않고 오류를 표시합니다.

## 개발 및 테스트

```bash
python -m unittest discover -s tests -v
python scripts/verify_install.py
```

`verify_install.py`는 현재 운영체제를 감지해 CUDA/NVENC 또는 MLX/VideoToolbox를 검사합니다.

## 문제 해결

### NVIDIA CUDA 또는 NVENC를 찾지 못함

NVIDIA 드라이버를 업데이트하고 Windows를 재부팅한 뒤 `install.bat`을 다시 실행하세요.

### MLX 또는 VideoToolbox를 찾지 못함

Apple Silicon Mac인지 확인하고 Finder 또는 일반 Terminal.app에서 실행하세요. SSH, 가상 머신, 일부 샌드박스에서는 Metal 장치에 접근할 수 없습니다.

### Codex CLI 또는 Claude Code CLI가 없음

현재 운영체제의 설치 파일을 다시 실행하세요. 이미 `claude` 명령을 설치해 두었다면 PATH에서 자동으로 찾습니다. 전사만 사용할 경우에는 전문용어 자동 입력을 끄면 AI 연결 없이 사용할 수 있습니다.

### YouTube 다운로드 오류

앱의 **연결과 업데이트**에서 yt-dlp를 업데이트하고, 필요한 경우 Chrome·Edge·Firefox 쿠키를 선택하세요.
