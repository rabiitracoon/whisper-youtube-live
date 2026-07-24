from __future__ import annotations


def parse_timecode(value: str) -> float | None:
    """Parse seconds, MM:SS, or HH:MM:SS. An empty value means no boundary."""
    text = value.strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) > 3 or any(not part.strip() for part in parts):
        raise ValueError("시간은 초, MM:SS 또는 HH:MM:SS 형식으로 입력해주세요.")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("시간에는 숫자와 콜론(:)만 사용할 수 있습니다.") from exc
    if any(number < 0 for number in numbers):
        raise ValueError("시간은 0보다 작을 수 없습니다.")
    if len(numbers) >= 2 and numbers[-1] >= 60:
        raise ValueError("초는 60보다 작아야 합니다.")
    if len(numbers) == 3 and numbers[-2] >= 60:
        raise ValueError("분은 60보다 작아야 합니다.")
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def validate_clip_range(start: float | None, end: float | None) -> tuple[float, float | None]:
    normalized_start = float(start or 0.0)
    normalized_end = float(end) if end is not None else None
    if normalized_start < 0 or (normalized_end is not None and normalized_end < 0):
        raise ValueError("전사 구간은 0보다 작을 수 없습니다.")
    if normalized_end is not None and normalized_end <= normalized_start:
        raise ValueError("전사 종료 시간은 시작 시간보다 뒤여야 합니다.")
    return normalized_start, normalized_end


def format_timecode(seconds: float | None, *, blank_for_none: bool = True) -> str:
    if seconds is None and blank_for_none:
        return ""
    total = max(0, int(seconds or 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
