# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Text normalization module for TTS processing.
Handles various text formats including URLs, emails, numbers, money, and special characters.
Converts them into a format suitable for text-to-speech processing.

Ported and adapted from Kokoro-FastAPI project.
"""

import math
import re
from typing import Optional

try:
    import inflect
    INFLECT_ENGINE = inflect.engine()
except ImportError:
    INFLECT_ENGINE = None

from ..structures.schemas import NormalizationOptions


# Constants for URL normalization
VALID_TLDS = [
    "com", "org", "net", "edu", "gov", "mil", "int", "biz", "info", "name",
    "pro", "coop", "museum", "travel", "jobs", "mobi", "tel", "asia", "cat",
    "xxx", "aero", "arpa", "bg", "br", "ca", "cn", "de", "es", "eu", "fr",
    "in", "it", "jp", "mx", "nl", "ru", "uk", "us", "io", "co", "ai", "app",
]

# Valid measurement units
VALID_UNITS = {
    # Length
    "m": "meter", "cm": "centimeter", "mm": "millimeter", "km": "kilometer",
    "in": "inch", "ft": "foot", "yd": "yard", "mi": "mile",
    # Mass
    "g": "gram", "kg": "kilogram", "mg": "milligram", "lb": "pound", "oz": "ounce",
    # Time
    "s": "second", "ms": "millisecond", "min": "minutes", "h": "hour",
    # Volume
    "l": "liter", "ml": "milliliter", "cl": "centiliter", "dl": "deciliter",
    # Speed
    "kph": "kilometer per hour", "mph": "mile per hour", "m/s": "meter per second",
    "km/h": "kilometer per hour",
    # Temperature
    "°c": "degree celsius", "°f": "degree fahrenheit", "k": "kelvin",
    # Frequency
    "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
    # Power/Energy
    "w": "watt", "kw": "kilowatt", "mw": "megawatt",
    "j": "joule", "kj": "kilojoule",
    # Data
    "b": "bit", "kb": "kilobit", "mb": "megabit", "gb": "gigabit", "tb": "terabit",
    "kbps": "kilobit per second", "mbps": "megabit per second",
    "gbps": "gigabit per second",
    "px": "pixel",
}

# Symbol replacements for TTS
SYMBOL_REPLACEMENTS = {
    '~': ' ',
    '@': ' at ',
    '#': ' number ',
    '$': ' dollar ',
    '%': ' percent ',
    '^': ' ',
    '&': ' and ',
    '*': ' ',
    '_': ' ',
    '|': ' ',
    '\\': ' ',
    '/': ' slash ',
    '=': ' equals ',
    '+': ' plus ',
}

# Currency units
MONEY_UNITS = {
    "$": ("dollar", "cent"),
    "£": ("pound", "pence"),
    "€": ("euro", "cent"),
    "¥": ("yen", "sen"),
}

# Pre-compiled regex patterns
EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b",
    re.IGNORECASE
)

URL_PATTERN = re.compile(
    r"(https?://|www\.|)+(localhost|[a-zA-Z0-9.-]+(\\.(?:"
    + "|".join(VALID_TLDS)
    + "))+|[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3})(:[0-9]+)?([/?][^\\s]*)?",
    re.IGNORECASE,
)

UNIT_PATTERN = re.compile(
    r"((?<!\w)([+-]?)(\d{1,3}(,\d{3})*|\d+)(\.\d+)?)\s*("
    + "|".join(sorted(list(VALID_UNITS.keys()), reverse=True))
    + r"""){1}(?=[^\w\d]{1}|\b)""",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"([0-9]{1,2} ?: ?[0-9]{2}( ?: ?[0-9]{2})?)( ?(pm|am)\b)?",
    re.IGNORECASE
)

MONEY_PATTERN = re.compile(
    r"(-?)(["
    + "".join(MONEY_UNITS.keys())
    + r"])(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b|t)*)\b",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"(-?)(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b)*)\b",
    re.IGNORECASE,
)


def _number_to_words(n) -> str:
    """Convert a number to words using inflect if available."""
    if INFLECT_ENGINE:
        return INFLECT_ENGINE.number_to_words(n)
    return str(n)


def _plural(word: str, count=2) -> str:
    """Return plural form of word if needed."""
    if INFLECT_ENGINE:
        return INFLECT_ENGINE.plural(word, count)
    return word if count == 1 else word + "s"


def _no(word: str, count) -> str:
    """Return number and word combination."""
    if INFLECT_ENGINE:
        return INFLECT_ENGINE.no(word, count)
    return f"{count} {word}s" if count != 1 else f"{count} {word}"


def conditional_int(number: float, threshold: float = 0.00001) -> int | float:
    """Convert float to int if close to a whole number."""
    if abs(round(number) - number) < threshold:
        return int(round(number))
    return number


def translate_multiplier(multiplier: str) -> str:
    """Translate multiplier abbreviations to words."""
    multiplier_translation = {
        "k": "thousand",
        "m": "million",
        "b": "billion",
        "t": "trillion",
    }
    if multiplier.lower() in multiplier_translation:
        return multiplier_translation[multiplier.lower()]
    return multiplier.strip()


def split_four_digit(number: float) -> str:
    """Split 4-digit numbers like years into two parts."""
    part1 = str(conditional_int(number))[:2]
    part2 = str(conditional_int(number))[2:]
    return f"{_number_to_words(part1)} {_number_to_words(part2)}"


def handle_units(u: re.Match[str]) -> str:
    """Convert units to their full form."""
    unit_string = u.group(6).strip()
    unit = unit_string

    if unit_string.lower() in VALID_UNITS:
        unit_parts = VALID_UNITS[unit_string.lower()].split(" ")
        
        # Handle the B vs b case (bit vs byte)
        if unit_parts[0].endswith("bit"):
            b_case = unit_string[min(1, len(unit_string) - 1)]
            if b_case == "B":
                unit_parts[0] = unit_parts[0][:-3] + "byte"
        
        number = u.group(1).strip()
        unit_parts[0] = _no(unit_parts[0], number)
        return " ".join(unit_parts)
    
    return f"{u.group(1)} {unit}"


def handle_numbers(n: re.Match[str]) -> str:
    """Convert numbers to spoken form."""
    number = n.group(2)
    
    try:
        number = float(number)
    except ValueError:
        return n.group()
    
    if n.group(1) == "-":
        number *= -1
    
    multiplier = translate_multiplier(n.group(3))
    
    number = conditional_int(number)
    if multiplier != "":
        multiplier = f" {multiplier}"
    else:
        # Special handling for 4-digit years
        if (number % 1 == 0 and len(str(int(abs(number)))) == 4 and 
            abs(number) > 1500 and abs(number) % 1000 > 9):
            return split_four_digit(abs(number))
    
    return f"{_number_to_words(number)}{multiplier}"


def handle_money(m: re.Match[str]) -> str:
    """Convert money expressions to spoken form."""
    bill, coin = MONEY_UNITS[m.group(2)]
    
    number = m.group(3)
    
    try:
        number = float(number)
    except ValueError:
        return m.group()
    
    if m.group(1) == "-":
        number *= -1
    
    multiplier = translate_multiplier(m.group(4))
    
    if multiplier != "":
        multiplier = f" {multiplier}"
    
    if number % 1 == 0 or multiplier != "":
        text_number = f"{_number_to_words(conditional_int(number))}{multiplier} {_plural(bill, count=int(abs(number)))}"
    else:
        sub_number = int(str(number).split(".")[-1].ljust(2, "0"))
        text_number = (
            f"{_number_to_words(int(math.floor(number)))} {_plural(bill, count=int(abs(number)))} "
            f"and {_number_to_words(sub_number)} {_plural(coin, count=sub_number)}"
        )
    
    return text_number


def handle_decimal(num: re.Match[str]) -> str:
    """Convert decimal numbers to spoken form."""
    a, b = num.group().split(".")
    return " point ".join([a, " ".join(b)])


def handle_email(m: re.Match[str]) -> str:
    """Convert email addresses into speakable format."""
    email = m.group(0)
    parts = email.split("@")
    if len(parts) == 2:
        user, domain = parts
        domain = domain.replace(".", " dot ")
        return f"{user} at {domain}"
    return email


def handle_url(u: re.Match[str]) -> str:
    """Make URLs speakable by converting special characters to spoken words."""
    if not u:
        return ""
    
    url = u.group(0).strip()
    
    # Handle protocol first
    url = re.sub(
        r"^https?://",
        lambda a: "https " if "https" in a.group() else "http ",
        url,
        flags=re.IGNORECASE,
    )
    url = re.sub(r"^www\.", "www ", url, flags=re.IGNORECASE)
    
    # Handle port numbers before other replacements
    url = re.sub(r":(\d+)(?=/|$)", lambda m: f" colon {m.group(1)}", url)
    
    # Split into domain and path
    parts = url.split("/", 1)
    domain = parts[0]
    path = parts[1] if len(parts) > 1 else ""
    
    # Handle dots in domain
    domain = domain.replace(".", " dot ")
    
    # Reconstruct URL
    if path:
        url = f"{domain} slash {path}"
    else:
        url = domain
    
    # Replace remaining symbols with words
    url = url.replace("-", " dash ")
    url = url.replace("_", " underscore ")
    url = url.replace("?", " question-mark ")
    url = url.replace("=", " equals ")
    url = url.replace("&", " ampersand ")
    url = url.replace("%", " percent ")
    url = url.replace(":", " colon ")
    url = url.replace("/", " slash ")
    
    # Clean up extra spaces
    return re.sub(r"\s+", " ", url).strip()


def handle_phone_number(p: re.Match[str]) -> str:
    """Convert phone numbers to spoken form."""
    groups = list(p.groups())
    
    parts = []
    
    # Country code
    if groups[0] is not None:
        code = groups[0].replace("+", "")
        parts.append(_number_to_words(code))
    
    # Area code
    area = groups[2].replace("(", "").replace(")", "")
    if INFLECT_ENGINE:
        parts.append(INFLECT_ENGINE.number_to_words(area, group=1, comma=""))
    else:
        parts.append(" ".join(area))
    
    # Telephone prefix
    if INFLECT_ENGINE:
        parts.append(INFLECT_ENGINE.number_to_words(groups[3], group=1, comma=""))
    else:
        parts.append(" ".join(groups[3]))
    
    # Line number
    if INFLECT_ENGINE:
        parts.append(INFLECT_ENGINE.number_to_words(groups[4], group=1, comma=""))
    else:
        parts.append(" ".join(groups[4]))
    
    return ", ".join(parts)


def handle_time(t: re.Match[str]) -> str:
    """Convert time expressions to spoken form."""
    groups = t.groups()
    time_parts = groups[0].split(":")
    
    numbers = []
    numbers.append(_number_to_words(time_parts[0].strip()))
    
    minute_number = _number_to_words(time_parts[1].strip())
    minute_val = int(time_parts[1])
    if minute_val < 10:
        if minute_val != 0:
            numbers.append(f"oh {minute_number}")
    else:
        numbers.append(minute_number)
    
    half = ""
    if len(time_parts) > 2:
        seconds_val = int(time_parts[2].strip())
        seconds_number = _number_to_words(seconds_val)
        second_word = _plural("second", seconds_val)
        numbers.append(f"and {seconds_number} {second_word}")
    else:
        if groups[2] is not None:
            half = " " + groups[2].strip()
        else:
            if minute_val == 0:
                numbers.append("o'clock")
    
    return " ".join(numbers) + half


def normalize_text(text: str, options: Optional[NormalizationOptions] = None) -> str:
    """
    Normalize text for TTS processing.
    
    Handles URLs, emails, numbers, money, units, time, phone numbers,
    and various special characters to make them pronounceable.
    
    Args:
        text: The input text to normalize
        options: Normalization options controlling which transformations to apply
        
    Returns:
        Normalized text suitable for TTS processing
    """
    if options is None:
        options = NormalizationOptions()
    
    if not options.normalize:
        return text
    
    # Handle email addresses first if enabled
    if options.email_normalization:
        text = EMAIL_PATTERN.sub(handle_email, text)
    
    # Handle URLs if enabled
    if options.url_normalization:
        text = URL_PATTERN.sub(handle_url, text)
    
    # Pre-process numbers with units if enabled
    if options.unit_normalization:
        text = UNIT_PATTERN.sub(handle_units, text)
    
    # Replace optional pluralization
    if options.optional_pluralization_normalization:
        text = re.sub(r"\(s\)", "s", text)
    
    # Replace phone numbers
    if options.phone_normalization:
        text = re.sub(
            r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})",
            handle_phone_number,
            text,
        )
    
    # Replace quotes and brackets
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace("«", chr(8220)).replace("»", chr(8221))
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    
    # Handle CJK punctuation
    for a, b in zip("、。！，：；？–", ",.!,:;?-"):
        text = text.replace(a, b + " ")
    
    # Handle time patterns
    text = TIME_PATTERN.sub(handle_time, text)
    
    # Clean up whitespace
    text = re.sub(r"[^\S \n]", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"(?<=\n) +(?=\n)", "", text)
    
    # Handle special characters that might cause audio artifacts
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    
    # Handle titles and abbreviations
    text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
    text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
    text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
    text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
    text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
    
    # Handle common words
    text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)
    
    # Handle numbers and money
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = MONEY_PATTERN.sub(handle_money, text)
    text = NUMBER_PATTERN.sub(handle_numbers, text)
    text = re.sub(r"\d*\.\d+", handle_decimal, text)
    
    # Handle remaining symbols
    if options.replace_remaining_symbols:
        for symbol, replacement in SYMBOL_REPLACEMENTS.items():
            text = text.replace(symbol, replacement)
    
    # Handle various formatting
    text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
    text = re.sub(r"(?<=\d)S", " S", text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", "s", text)
    text = re.sub(
        r"(?:[A-Za-z]\.){2,} [a-z]", lambda m: m.group().replace(".", "-"), text
    )
    text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)
    
    # Final whitespace cleanup
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    
    return text
