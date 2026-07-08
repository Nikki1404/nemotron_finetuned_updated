import re

try:
    from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
except Exception:
    InverseNormalizer = None

DIGITS = {"zero":"0","oh":"0","o":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
ONES = {"zero":0,"oh":0,"o":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}
TEENS = {"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19}
TENS = {"twenty":20,"thirty":30,"forty":40,"fourty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
MULTIPLIERS = {"double":2,"triple":3,"tripe":3,"quadruple":4}
NUMBER_WORDS = set(DIGITS) | set(TEENS) | set(TENS) | {"hundred","thousand","and"} | set(MULTIPLIERS)
_ITN = None

def get_itn():
    global _ITN
    if _ITN is None and InverseNormalizer is not None:
        try:
            _ITN = InverseNormalizer(lang="en")
        except Exception:
            _ITN = False
    return _ITN if _ITN is not False else None

def clean_token(tok: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", tok).lower()

def token_to_number(tok: str):
    tok = clean_token(tok)
    if tok.isdigit(): return int(tok)
    if tok in ONES: return ONES[tok]
    if tok in TEENS: return TEENS[tok]
    if tok in TENS: return TENS[tok]
    return None

def parse_number_phrase(tokens):
    toks = [clean_token(t) for t in tokens if clean_token(t)]
    if not toks: return None
    result = ""; i = 0; used_multiplier = False
    while i < len(toks):
        t = toks[i]
        if t in MULTIPLIERS and i + 1 < len(toks) and toks[i+1] in DIGITS:
            result += DIGITS[toks[i+1]] * MULTIPLIERS[t]; used_multiplier = True; i += 2; continue
        if t in DIGITS: result += DIGITS[t]; i += 1; continue
        if t.isdigit() and len(t) == 1: result += t; i += 1; continue
        break
    if used_multiplier and result:
        while i < len(toks):
            t = toks[i]
            if t in DIGITS: result += DIGITS[t]
            elif t.isdigit() and len(t) == 1: result += t
            else: return None
            i += 1
        return result
    if all(t in DIGITS or (t.isdigit() and len(t) == 1) for t in toks):
        return "".join(DIGITS.get(t, t) for t in toks)
    if "thousand" in toks:
        idx = toks.index("thousand"); before = toks[:idx]; after = [t for t in toks[idx+1:] if t != "and"]
        before_num = sum(token_to_number(t) or 0 for t in before)
        if before_num == 0: return None
        if not after: return str(before_num * 1000)
        after_parts = []
        for t in after:
            n = token_to_number(t)
            if n is None: return None
            after_parts.append(n)
        if len(after_parts) == 1: return str(before_num * 1000 + after_parts[0])
        return str(before_num) + "".join(f"{p:02d}" if p < 100 else str(p) for p in after_parts)
    if "hundred" in toks:
        idx = toks.index("hundred"); before = toks[:idx]; after = [t for t in toks[idx+1:] if t != "and"]
        base = sum(token_to_number(t) or 0 for t in before) * 100
        if base == 0: return None
        if not after: return str(base)
        after_parts = []
        for t in after:
            n = token_to_number(t)
            if n is None: return None
            after_parts.append(n)
        if len(after_parts) == 1: return str(base + after_parts[0])
        if len(after_parts) == 2 and after_parts[0] < 10 and after_parts[1] < 10:
            return str(base + after_parts[0]) + str(after_parts[1])
        normal = base + after_parts[0]
        rest = "".join(str(x) if x < 10 else f"{x:02d}" for x in after_parts[1:])
        return str(normal) + rest
    if len(toks) >= 2:
        parts = []
        for t in toks:
            n = token_to_number(t)
            if n is None: return None
            parts.append(n)
        return str(parts[0]) + "".join(f"{p:02d}" if p < 100 else str(p) for p in parts[1:])
    return None

def apply_custom_asr_rules(text: str) -> str:
    words = text.split(); out = []; i = 0
    while i < len(words):
        collected = []; j = i
        while j < len(words):
            c = clean_token(words[j])
            if c in NUMBER_WORDS or c.isdigit(): collected.append(words[j]); j += 1
            else: break
        if collected:
            converted = parse_number_phrase(collected)
            if converted: out.append(converted); i = j; continue
        out.append(words[i]); i += 1
    return " ".join(out)

def apply_itn(text: str) -> str:
    itn = get_itn()
    if itn is None: return text
    try: return itn.inverse_normalize(text, verbose=False)
    except Exception: return text

def normalize_ticket_ids(text: str) -> str:
    return re.sub(r"\bT\s*K\s*T\s*([A-Z0-9\s]{4,30})", lambda m: "TKT" + re.sub(r"\s+", "", m.group(1).upper()), text, flags=re.I)

def normalize_asr_numbers(text: str, use_itn: bool = True) -> str:
    if not text: return text
    text = apply_custom_asr_rules(text)
    if use_itn: text = apply_itn(text)
    text = apply_custom_asr_rules(text)
    text = normalize_ticket_ids(text)
    return text
