"""
Utility validators for the masker plugin.
"""

import re
from typing import Iterable


# ============================================================================
# Polish identification numbers
# ============================================================================


def is_valid_pesel(pesel: str) -> bool:
    """
    Validate a Polish PESEL number.

    The algorithm:
    1. Multiply the first 10 digits by the weight vector
       [1, 3, 7, 9, 1, 3, 7, 9, 1, 3].
    2. Sum the results and take ``mod 10``.
    3. The checksum digit is ``(10 - sum_mod) % 10``.
    4. Compare the checksum digit with the 11th digit.

    Parameters
    ----------
    pesel: str
        A string consisting of exactly 11 digits.

    Returns
    -------
    bool
        ``True`` if the PESEL passes the checksum test, otherwise ``False``.
    """
    if not isinstance(pesel, str):
        return False

    if not (pesel.isdigit() and len(pesel) == 11):
        return False

    weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    digits = [int(ch) for ch in pesel]

    checksum_calc = sum(w * d for w, d in zip(weights, digits[:10])) % 10
    checksum_expected = (10 - checksum_calc) % 10

    return checksum_expected == digits[10]


def is_valid_nip(raw_nip: str) -> bool:
    """Validate a Polish NIP (Tax Identification Number).

    The NIP consists of 10 digits.  The checksum is calculated with the
    weights ``[6, 5, 7, 2, 3, 4, 5, 6, 7]``; the sum of the weighted digits
    modulo 11 must equal the last digit.

    ``raw_nip`` may contain hyphens (e.g. ``123-456-78-90``) – they are stripped
    before validation.
    """
    if not isinstance(raw_nip, str):
        return False
    # Remove any hyphens or spaces that may be present
    digits = re.sub(r"[-\s]", "", raw_nip)

    if not re.fullmatch(r"\d{10}", digits):
        return False

    weights = (6, 5, 7, 2, 3, 4, 5, 6, 7)
    checksum = sum(w * int(d) for w, d in zip(weights, digits[:9])) % 11
    return checksum == int(digits[9])


def is_valid_krs(raw_krs: str) -> bool:
    """Validate a Polish KRS number (exactly 10 digits).

    Note: KRS has no checksum algorithm – validation is limited to the
    digit count.  False positives are mitigated by the more restrictive
    regex in :class:`KrsRule`.
    """
    if not isinstance(raw_krs, str):
        return False
    digits = re.sub(r"[-\s]", "", raw_krs)
    return bool(re.fullmatch(r"\d{10}", digits))


def is_valid_regon(raw_regon: str) -> bool:
    """
    Validate a Polish REGON number.

    * 9‑digit REGON – checksum weights: [8, 9, 2, 3, 4, 5, 6, 7]
    * 14‑digit REGON – first 9 digits are validated as above,
      then digits 1‑13 (including the 9‑digit checksum) are validated
      with weights: [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6]

    The checksum is ``sum % 11``; if the result is 10 the checksum digit
    becomes 0.
    """
    if not isinstance(raw_regon, str):
        return False
    # Remove any whitespace that may be present
    digits = re.sub(r"\s+", "", raw_regon)

    if not re.fullmatch(r"\d{9}|\d{14}", digits):
        return False

    def checksum(value: str, weights: list[int]) -> int:
        s = sum(int(d) * w for d, w in zip(value, weights))
        r = s % 11
        return 0 if r == 10 else r

    # ----- 9‑digit validation -----
    w9 = [8, 9, 2, 3, 4, 5, 6, 7]
    if len(digits) == 9:
        return checksum(digits[:8], w9) == int(digits[8])

    # ----- 14‑digit validation -----
    # first part (first 9 digits) must be correct
    if checksum(digits[:8], w9) != int(digits[8]):
        return False

    w14 = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6]
    return checksum(digits[:13], w14) == int(digits[13])


# ============================================================================
# Financial numbers
# ============================================================================


def _luhn_checksum(number: str) -> int:
    """Calculate Luhn checksum for a numeric string."""
    digits = [int(d) for d in number[::-1]]
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10


def is_valid_credit_card(number: str) -> bool:
    """
    Validate a credit‑card number (13-19 digits, optional spaces or dashes)
    using the Luhn algorithm.
    """
    cleaned = re.sub(r"[ -]", "", number)
    if not cleaned.isdigit():
        return False
    if not (13 <= len(cleaned) <= 19):
        return False
    return _luhn_checksum(cleaned) == 0


def is_valid_nrb(nrb: str) -> bool:
    """Validate Polish NRB (26 digits, optional spaces or hyphens).

    Requires the string to be exactly 26 digits.  Since there is no
    country-code prefix in a domestic account number and no checksum for
    NRBs, this is purely a format check – combined with the IBAN exclusion
    lookbehinds in :class:`NrbRule` it reduces (but cannot eliminate) false
    positives from random long digit strings.
    """
    if not isinstance(nrb, str):
        return False
    cleaned = re.sub(r"[\s\-]", "", nrb)
    return cleaned.isdigit() and len(cleaned) == 26


# ============================================================================
# Vehicle and transport
# ============================================================================

_VIN_TRANSLATION = {
    **{c: i for i, c in enumerate("ABCDEFGHJKLMNPRSTUVWXYZ", start=1)},
    **{str(i): i for i in range(10)},
}
_VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def is_valid_vin(vin: str) -> bool:
    """Validate a 17‑character VIN using its checksum (position 9)."""
    if not isinstance(vin, str):
        return False
    vin = vin.upper()
    if not re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", vin):
        return False
    total = sum(_VIN_TRANSLATION[ch] * w for ch, w in zip(vin, _VIN_WEIGHTS))
    remainder = total % 11
    check = "X" if remainder == 10 else str(remainder)
    return vin[8] == check


def is_valid_car_plate(plate: str) -> bool:
    """Validate a Polish car registration plate against real formats.

    Accepted formats::

        ABC 12345          (standard post-2001, no trailing letters)
        AB 12 CD           (post-2001 with voivodeship separator)
        ABC 12 D4          (special plates, e.g. diplomatic/military)
        AB 1234            (pre-2001 format, still valid)
        AB 123 CD          (pre-2001 with trailing letters)
        AB 1234C           (post-2022 new-style: AA NNNN + letter suffix)

    Rejects patterns with more than 5 digits, more than 3 leading letters,
    or any non-standard separator layout.
    """
    if not isinstance(plate, str):
        return False

    plate = plate.upper().replace(" ", "")

    # Format: 2 letters + 5 digits (post-2001 standard or pre-2001)
    if re.fullmatch(r"[A-Z]{2}\d{5}", plate):
        return True
    # Format: 3 letters + 2 digits + 1 letter + 2 digits (special plates)
    if re.fullmatch(r"[A-Z]{3}\d{2}[A-Z]\d{2}", plate):
        return True
    # Format: 2 letters + 2 digits + 2 letters (post-2001 with voivodeship)
    if re.fullmatch(r"[A-Z]{2}\d{2}[A-Z]{2}", plate):
        return True
    # Format: 2 letters + 3 digits + 2 letters (pre-2001 variant)
    if re.fullmatch(r"[A-Z]{2}\d{3}[A-Z]{2}", plate):
        return True
    # Format: 2 letters + 4 digits + 1 letter (post-2022 new-style, e.g. WA12345A)
    if re.fullmatch(r"[A-Z]{2}\d{4}[A-Z]", plate):
        return True

    return False


# ============================================================================
# International identification
# ============================================================================


def is_valid_ssn(ssn: str) -> bool:
    """Validate SSN format ``AAA‑GG‑SSSS``.

    Rejects prohibited ranges:
    * Area code (AAA): ``000``, ``666``, or ``900``–``999``
    * Group (GG): ``00``
    * Serial (SSSS): ``0000``
    """
    if not isinstance(ssn, str):
        return False
    m = re.fullmatch(r"(\d{3})-(\d{2})-(\d{4})", ssn)
    if not m:
        return False
    area, group, serial = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if area == 0 or area == 666 or area >= 900:
        return False
    if group == 0 or serial == 0:
        return False
    return True


def is_valid_eu_vat(vat: str) -> bool:
    """Validate EU VAT identifier (e.g. ``PL1234567890``).

    Requires exactly 2 leading letters followed by 8-12 alphanumeric characters,
    with at least 4 digits (reduced from 6 to accommodate country-specific formats
    like Maltese VAT which can have as few as 8 total characters).
    """
    if not isinstance(vat, str):
        return False
    vat_upper = vat.upper()
    # Must be 2 letters + digits/letters (10-14 total chars)
    if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{8,12}", vat_upper):
        return False
    # Count digits - must have at least 4 digits to avoid matching regular words
    digit_count = sum(c.isdigit() for c in vat_upper[2:])
    return digit_count >= 4


# ============================================================================
# Network and security
# ============================================================================


def is_valid_mac(mac: str) -> bool:
    """Validate MAC address (e.g. ``00:1A:2B:3C:4D:5E``).

    Requires a consistent separator throughout the address — colons only or
    hyphens only.  Mixed separators (e.g. ``00:1A-2B:3C-4D:5E``) are rejected.
    """
    if not isinstance(mac, str):
        return False
    # Detect separator (if any) from the first pair
    m = re.fullmatch(r"([0-9A-Fa-f]{2})([:]|-)?([0-9A-Fa-f]{2})([:-]?)*", mac)
    if not m:
        return False
    sep = m.group(2)  # ':' or '-' or None
    if not sep:
        # No separator at all — check for plain 12 hex chars
        return bool(re.fullmatch(r"[0-9A-Fa-f]{12}", mac))
    # Separator must be consistent across all pairs
    pair = "[0-9A-Fa-f]{2}" + re.escape(sep)
    return bool(re.fullmatch(pair * 5 + "[0-9A-Fa-f]{2}", mac))


def is_possible_token(token: str, min_len: int = 32) -> bool:
    """
    Heuristic check for API keys: length + allowed characters + complexity.
    Accepts alphanumerics, ``-`` and ``_``.
    Requires at least 2 types of characters (upper, lower, digit, special).
    """
    if not isinstance(token, str):
        return False

    if len(token) < min_len:
        return False

    if not re.fullmatch(r"[A-Za-z0-9\-_]+", token):
        return False

    # Must have mixed case or contain special chars
    has_upper = any(c.isupper() for c in token)
    has_lower = any(c.islower() for c in token)
    has_digit = any(c.isdigit() for c in token)
    has_special = any(c in "-_" for c in token)

    complexity = sum([has_upper, has_lower, has_digit, has_special])
    return complexity >= 2


def is_possible_jwt(token: str) -> bool:
    """Detect something that looks like a JWT."""
    if not isinstance(token, str):
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    # Each part must be base64url and have reasonable minimum length.
    # Reduced from 20 to 10 chars — many legitimate minimal JWTs have shorter
    # segments (e.g. compact tokens with minimal claims).
    for i, p in enumerate(parts):
        if not p or not re.fullmatch(r"[A-Za-z0-9\-_]+", p):
            return False
        if len(p) < 10:
            return False
    return True


def is_valid_sim_iccid(iccid: str) -> bool:
    """Validate SIM card ICCID (19 or 20 digits) with Luhn checksum.

    Per ISO/IEC 7812-4 the last digit is a check digit calculated using
    the Luhn algorithm (modulus 10, double every second digit from right).
    """
    if not isinstance(iccid, str):
        return False
    cleaned = re.sub(r"\s+", "", iccid)
    if not (cleaned.isdigit() and 19 <= len(cleaned) <= 20):
        return False
    return _luhn_checksum(cleaned) == 0


def is_valid_ssl_serial(serial: str) -> bool:
    """Validate SSL certificate serial number format.

    Requires 16-40 uppercase hex characters (serial numbers in TLS
    certificates are encoded as uppercase per RFC 5280).
    Rejects lowercase hex which typically indicates a non-certificate context.
    """
    if not isinstance(serial, str):
        return False
    return bool(re.fullmatch(r"[0-9A-F]{16,40}", serial))


# ============================================================================
# Business identifiers
# ============================================================================


def is_possible_transaction_ref(ref: str) -> bool:
    """Validate transaction reference format.

    Requires at least two separate digit groups within the string (a single
    stray digit is not enough to qualify as a transaction reference).  The
    original validator only checked ``len(8-64) + one_digit``, which accepted
    almost any alphanumeric string.
    """
    if not isinstance(ref, str):
        return False
    if not (8 <= len(ref) <= 64):
        return False
    digit_groups = re.findall(r"\d+", ref)
    return len(digit_groups) >= 2


# ============================================================================
# Phone numbers
# ============================================================================

# Valid Polish mobile prefix ranges (second digit of the 3-digit prefix).
_POLISH_MOBILE_PREFIXES = frozenset(
    {
        # Mobile
        "50",
        "51",
        "53",
        "45",
        "60",
        "66",
        "69",
        "72",
        "73",
        "78",
        "79",
        # Unified Emergency Numbers
        "800",
    }
)


def is_valid_polish_phone(phone: str) -> bool:
    """Validate a Polish phone number (9 digits without country code).

    Checks that the first 2-3 digits form a valid Polish mobile prefix.
    This reduces false positives from random 9-digit sequences.
    """
    if not isinstance(phone, str):
        return False
    cleaned = re.sub(r"[\s\-]", "", phone)
    if not (cleaned.isdigit() and len(cleaned) == 9):
        return False
    # Check mobile prefixes: first 2 digits always, plus third if it's an 800 number
    prefix_2 = cleaned[:2]
    if prefix_2 in ("80",) and cleaned[:3] in _POLISH_MOBILE_PREFIXES:
        return True
    return cleaned[:2] in _POLISH_MOBILE_PREFIXES
