# Lecture Scribe Design System — Studio Shell

## Product architecture

Lecture Scribe is a native Windows study-production tool, not a settings form. Its interface is built as a persistent desktop app shell:

- 238 px dark navigation rail for product identity, destinations, and environment status.
- Light adaptive workspace for page-specific work.
- Bento project board for lecture intake and run options.
- Dedicated dark media-editing workspace for waveform work.
- Visible workflow: 영상 가져오기 → 쉬는 시간 정리 → 강의 글 확인 → 내 노트 완성.

## Design dials

- Style: cinematic productivity shell + bento workspace.
- Variance: 7/10 — visually distinct, asymmetric hierarchy.
- Motion: 3/10 — restrained native state feedback only.
- Density: 7/10 — productive without feeling cramped.
- Platform: PySide6 on Windows, Korean-first.

## Semantic colour system

| Token | Value | Purpose |
|---|---:|---|
| Chrome | `#151722` | Persistent sidebar |
| Chrome raised | `#1C1F2A` | Sidebar status card |
| Canvas | `#F5F6F9` | Main workspace |
| Surface | `#FFFFFF` | Primary bento cards |
| Surface muted | `#F0F0F7` | Secondary/control cards |
| Hero start | `#1C1E2A` | Project hero |
| Hero end | `#352D68` | Project hero accent depth |
| Primary | `#7161E5` | Primary action and active editor tool |
| Primary soft | `#EBE8FF` | Suggested action |
| Text | `#1A1D2B` | Main content |
| Text muted | `#747A8A` | Supporting copy |
| Dark text | `#D9DCE6` | Editor content |
| Border | `#E0E2E9` | Light surface separation |
| Dark border | `#303445` | Editor separation |
| Success | `#7EDAA4` / `#17653A` | Ready states with labels |
| Warning | `#F0C879` / `#845117` | Attention states with labels |
| Danger | `#B42318` | Stop/error only |

## Typography

- Malgun Gothic first for complete Korean rendering; Segoe UI fallback.
- Cascadia Mono/Consolas for precise time values.
- Page title 23 px; project title 23 px; section title 17 px; body 14 px; helper 11–12 px.
- Use weight 700–750 for page/section identity, 600–650 for controls, regular for content.

## Navigation

- Three persistent, labelled destinations with matching 20 px outline SVG icons:
  - 새 강의 노트
  - 노트 작성 방식
  - 연결과 업데이트
- Navigation remains visible while page content scrolls.
- Active destination uses both a raised surface and stronger text; never colour alone.
- Environment readiness lives at the bottom of the rail and is always available.

## Surface hierarchy

- Project hero: 20 px radius, dark tonal gradient, one clear story and four-step journey.
- Main bento cards: 18 px radius, 1 px border, no generic drop shadows.
- Action dock: dark 16 px surface separating the primary action from configuration.
- Nested advanced panels: 12 px radius and a quieter tonal surface.
- Inputs/buttons: 10–11 px radius; 40 px minimum height, 46 px primary height.

## Interaction

- One primary action per page.
- Long work exposes progress, a plain-language stage, safe stop, and detailed log on request.
- Advanced recognition settings and logs use progressive disclosure.
- Completed pipeline stages remain resumable.
- Focus is always visible; tab order follows the visual hierarchy.
- SVG navigation icons use a consistent 1.5 px outline style.

## Plain-language contract

Default screens never expose internal terms. Mappings:

| Internal | Visible |
|---|---|
| transcription | 강의 글 만들기 |
| transcript | 강의 글 |
| prompt | 노트 작성 방식 |
| OAuth | ChatGPT 연결 |
| VAD | 말이 없는 긴 구간 건너뛰기 |
| checkpoint | 완료한 단계 / 저장된 단계 |
| Razor | 자르기 |
| playhead | 현재 위치 |
| clip | 구간 |
| yt-dlp | 영상 가져오기 도구 |

Technical names are allowed only in troubleshooting details.

## Accessibility and QA

- Normal text contrast ≥4.5:1; dark secondary text ≥3:1.
- Status always combines colour with readable text.
- Interactive controls have pressed, focus, disabled, and selected states without layout shift.
- Gestures and keyboard shortcuts always have visible button alternatives.
- Long content uses one page scroll area; waveform horizontal scroll is intentional and labelled.
- No emoji structural icons; vector assets only.
- QA renders: main, advanced, review, prompt, tools, editor, and automatic detection.

