# Lecture Scribe Design System: Calm Light

A calm, light desktop app (Linear / Granola / MacWhisper inspired). One indigo accent, hairline borders, no shadows, Korean-first plain language. Source of truth: `lecture_scribe/ui/theme.py` (`COLORS`, `APP_STYLE`). Never hard-code hex values elsewhere; read `COLORS`.

## Layout

- Window: min 960x640, default 1240x820 clamped to the screen, geometry saved via QSettings.
- Sidebar 232 px, `sidebar` background, 1 px right hairline. Brand = small accent dot + "Lecture Scribe". Four nav rows (icon + label); the selected row is a white pill with a border and weight 600 (not color alone). Ctrl+1..4 switch pages. Bottom: compact status rows (컴퓨터, AI 계정) using StatusPill, elided with tooltip.
- Content: every page is a frameless transparent QScrollArea; padding 32; content column max 960 px, centered. Header = PageTitle (24/700) + one PageSubtitle line.
- Spacing scale 4/8/12/16/24/32. Radii 6/8/12 only (controls 8, cards 12, pills 11).

## Tokens (`COLORS`)

Surfaces `bg #F7F7F8`, `surface #FFF`, `surface_muted`, `sidebar`, `sidebar_active`, `sidebar_hover`. Borders `border`, `border_strong`. Text `text`, `text_secondary`, `text_tertiary`. Accent `accent #4F46E5` (+ hover/pressed/on/subtle/border). Status `success`, `warning`, `danger` each with `_subtle` (and `_border`). `focus`. Dark `editor_*` and `wave_*` tokens belong to the audio editor only (it styles itself).

## Typography

Base font set in `configure_fonts` (14 px pixel size; Pretendard if installed, else Malgun Gothic / Apple SD Gothic Neo). The stylesheet never sets font-size on the generic `QWidget` selector, only on named selectors: PageTitle 24/700, SectionTitle 16/600, FieldLabel 13/500, BodyText 13, HelperText 12 (tertiary), StatusPill 12/600. Log views use `MONO_STACK` at 12 px.

## Components (objectName)

- `QFrame#Card` (surface, 1 px border, radius 12); `tone="muted"` for nested panels.
- Buttons, height 36, radius 8, padding 0 14: `PrimaryButton` (accent fill; one per page), `SecondaryButton` (also the bare QPushButton look), `GhostButton` (transparent; used for disclosure rows such as "고급 설정" with a chevron icon), `DangerButton` (danger text, tinted hover). `[compact="true"]` = min-height 0, padding 0 6.
- `QLabel#StatusPill[tone=neutral|success|warning|danger|accent]` tinted bg, matching text and border.
- Inputs: 36 px high (`SourceInput` 44), 1 px border; focus changes only border color (no layout shift). Combo chevron and spinbox arrows are SVG assets (absolute paths).
- `Stepper`: four numbered circles (pending, active, done check, error "!") joined by connector lines; accent for active/done, danger for error.
- Progress bar 6 px, no text; percent label beside it.

## Pages

1. 새 강의 노트: one source card (URL/file input 44 px + 붙여넣기/파일 선택, 강의명 (선택), save folder, primary 강의 노트 만들기). Below it a single status region that morphs idle (hidden) / running (stepper, bar, plain detail line, 잠시 멈추기, 자세히 보기 log) / review (transcript check) / done (title, message, elided path, 노트 열기 primary) / stopped. Then a "처리 방식" card with the main toggles and a "고급 설정" disclosure (recognition settings, step-by-step mode).
2. 실시간 자막: separate module (`live_caption_page.py`), wrapped in the shared scroll shell.
3. 노트 작성 방식: AI model card (provider, model, 생각 깊이: 빠르게/보통/깊게/더 깊게/최대) and 노트 양식 card (form combo, 새로 만들기/이름 바꾸기/삭제, editor min 240, 되돌리기, 저장).
4. 연결과 업데이트: stacked cards (내 컴퓨터, AI 계정 연결, 업데이트, 자세한 기록), each with title + StatusPill + one-line description + actions. CLI paths only in tooltips. During login the 연결하기 button becomes 취소 and a code row with an instruction line appears.

## Behavior rules

- Worker signals connect only to bound MainWindow slots (workers emit `completed(worker)`); never lambdas that touch widgets on the worker thread.
- Buttons use Tab-only focus so a mouse click leaves no focus ring.
- Visible copy is plain Korean; technical names (Whisper, CUDA, beam size) appear only in tooltips.
- Errors: friendly dialog with raw detail written to the log; startup/status checks report inline instead of a dialog.
