import re
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")
def _tokenize(text):
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]

print(f"Tokenize '建築法': {_tokenize('建築法')}")
