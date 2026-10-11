"""Turn phones, prices, dates, and times into Arabic words before speech."""

from __future__ import annotations

import re

_EASTERN_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

_DIGIT_WORDS = (
    "صفر",
    "واحد",
    "اثنين",
    "ثلاثة",
    "أربعة",
    "خمسة",
    "ستة",
    "سبعة",
    "ثمانية",
    "تسعة",
)

_ONES = (
    "",
    "واحد",
    "اثنان",
    "ثلاثة",
    "أربعة",
    "خمسة",
    "ستة",
    "سبعة",
    "ثمانية",
    "تسعة",
)
_TEENS = (
    "عشرة",
    "أحد عشر",
    "اثنا عشر",
    "ثلاثة عشر",
    "أربعة عشر",
    "خمسة عشر",
    "ستة عشر",
    "سبعة عشر",
    "ثمانية عشر",
    "تسعة عشر",
)
_TEENS_OBLIQUE_12 = "اثني عشر"
_TENS = (
    "",
    "",
    "عشرون",
    "ثلاثون",
    "أربعون",
    "خمسون",
    "ستون",
    "سبعون",
    "ثمانون",
    "تسعون",
)
_TENS_OBLIQUE = (
    "",
    "",
    "عشرين",
    "ثلاثين",
    "أربعين",
    "خمسين",
    "ستين",
    "سبعين",
    "ثمانين",
    "تسعين",
)

_MONTHS = (
    "ربيع الأول",
    "ربيع الآخر",
    "جمادى الأولى",
    "جمادى الآخرة",
    "ذو القعدة",
    "ذو الحجة",
    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "ابريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "اغسطس",
    "سبتمبر",
    "أكتوبر",
    "اكتوبر",
    "نوفمبر",
    "ديسمبر",
    "محرم",
    "صفر",
    "رجب",
    "شعبان",
    "رمضان",
    "شوال",
)

_PHONE = re.compile(r"\+[0-9]{7,15}|(?<![\d.])0[0-9]{8,14}(?![\d.])")
_PRICE = re.compile(
    r"(?<![\d.])(?P<whole>[0-9]{1,6})\.(?P<frac>[0-9]{1,2})(?!\d)\s*ريال"
)
_TIME = re.compile(r"(?<![\d.])(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})(?!\d)")
_MONTH_PATTERN = "|".join(
    re.escape(name) for name in sorted(_MONTHS, key=len, reverse=True)
)
_DATE = re.compile(
    rf"(?<!\d)(?P<day>[0-9]{{1,2}})\s+(?P<month>{_MONTH_PATTERN})"
    r"(?:\s+(?P<year>[0-9]{4}))?"
)


def normalize_for_speech(text: str) -> str:
    """Replace phones, prices, dates, and times in this text with spoken words."""
    spoken = text.translate(_EASTERN_DIGITS)
    spoken = _PHONE.sub(_replace_phone, spoken)
    spoken = _replace_prices(spoken)
    spoken = _TIME.sub(_replace_time, spoken)
    return _DATE.sub(_replace_date, spoken)


def _replace_phone(match: re.Match[str]) -> str:
    """Spell the phone number this pattern found."""
    return _spell_phone(match.group(0))


def _replace_prices(source: str) -> str:
    """Spell every riyal price in the text."""

    def replace(match: re.Match[str]) -> str:
        """Spell one price, using the words that fit the text around it."""
        return _spell_price(source, match)

    return _PRICE.sub(replace, source)


def _replace_time(match: re.Match[str]) -> str:
    """Spell the clock time this pattern found."""
    return _spell_time(match)


def _replace_date(match: re.Match[str]) -> str:
    """Spell the day, month, and year this pattern found."""
    return _spell_date(match)


def _spell_phone(raw: str) -> str:
    """Say a leading plus as زائد, then one Arabic word per digit."""
    words: list[str] = []
    if raw.startswith("+"):
        words.append("زائد")
        raw = raw[1:]
    words.extend(_DIGIT_WORDS[int(char)] for char in raw)
    return " ".join(words)


def _spell_price(source: str, match: re.Match[str]) -> str:
    """Say a riyal amount in words, including half and halalas."""
    whole = int(match.group("whole"))
    amount = _cardinal(whole, oblique=_after_bi(source, match.start()))
    if amount is None:
        return match.group(0)
    frac = match.group("frac")
    if frac in {"5", "50"}:
        return f"{amount} ريال ونص"
    if frac in {"0", "00"}:
        return f"{amount} ريال"
    halalas = int(frac) * 10 if len(frac) == 1 else int(frac)
    halala_words = _cardinal(halalas, oblique=True)
    if halala_words is None:
        return match.group(0)
    return f"{amount} ريال و{halala_words} هللة"


def _spell_time(match: re.Match[str]) -> str:
    """Say an hour, and say ونص for half past."""
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        return match.group(0)
    hour_words = _cardinal(hour, oblique=False)
    if hour_words is None:
        return match.group(0)
    if minute == 0:
        return hour_words
    if minute == 30:
        return f"{hour_words} ونص"
    minute_words = _cardinal(minute, oblique=True)
    if minute_words is None:
        return match.group(0)
    return f"{hour_words} و{minute_words}"


def _spell_date(match: re.Match[str]) -> str:
    """Say the day and month, and the year when one is present."""
    day = int(match.group("day"))
    if not 1 <= day <= 31:
        return match.group(0)
    day_words = _cardinal(day, oblique=False)
    if day_words is None:
        return match.group(0)
    spoken = f"{day_words} {match.group('month')}"
    year = match.group("year")
    if year is None:
        return spoken
    year_words = _year(int(year))
    if year_words is None:
        return f"{spoken} {year}"
    return f"{spoken} {year_words}"


def _after_bi(source: str, start: int) -> bool:
    """Return true when بـ comes just before this number."""
    prefix = source[max(0, start - 3) : start]
    return prefix.endswith("بـ ") or prefix.endswith("ب ")


def _year(value: int) -> str | None:
    """Say a year from 2000 to 2099, or return nothing if it is outside that range."""
    if not 2000 <= value <= 2099:
        return None
    rest = value - 2000
    if rest == 0:
        return "ألفين"
    rest_words = _cardinal(rest, oblique=True)
    if rest_words is None:
        return None
    return f"ألفين و{rest_words}"


def _cardinal(value: int, *, oblique: bool) -> str | None:
    """Say a number from 0 to 99. Oblique is the form used after بـ or و."""
    if not 0 <= value <= 99:
        return None
    if value == 0:
        return "صفر"
    if value < 10:
        if value == 2 and oblique:
            return "اثنين"
        return _ONES[value]
    if value < 20:
        if value == 12 and oblique:
            return _TEENS_OBLIQUE_12
        return _TEENS[value - 10]
    tens, ones = divmod(value, 10)
    ten_words = _TENS_OBLIQUE[tens] if oblique else _TENS[tens]
    if ones == 0:
        return ten_words
    unit = "اثنين" if ones == 2 and oblique else _ONES[ones]
    return f"{unit} و{ten_words}"
