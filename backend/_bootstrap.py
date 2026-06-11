"""_bootstrap.py — install minimal SDK stubs BEFORE importing backend modules.

Headless gate support: llm.py constructs a genai client at import and logger.py inits
firebase at import. No key / network available in this env, so stub google.genai, dotenv,
and firebase_admin with inert doubles. Import this module first (it mutates sys.modules).
Mirrors the established gate harness (google.genai, firebase_admin, dotenv).
"""
import sys, types as _t


def _mod(name):
    m = _t.ModuleType(name)
    sys.modules[name] = m
    return m


# ── dotenv ──
_dotenv = _mod("dotenv")
_dotenv.load_dotenv = lambda *a, **k: None

# ── google.genai (+ types) ──
_google = _mod("google")
_genai = _mod("google.genai")
_gtypes = _mod("google.genai.types")
_google.genai = _genai
_genai.types = _gtypes


class _Resp:
    text = "{}"


class _Models:
    def generate_content(self, *a, **k):
        return _Resp()


class _Client:
    def __init__(self, *a, **k):
        self.models = _Models()


_genai.Client = _Client
for _n in ("GenerateContentConfig", "Content", "Part", "Schema", "Tool"):
    setattr(_gtypes, _n, type(_n, (), {"__init__": lambda self, *a, **k: None}))

# ── firebase_admin (+ credentials, firestore) ──
_fb = _mod("firebase_admin")
_cred = _mod("firebase_admin.credentials")
_fs = _mod("firebase_admin.firestore")
_fb.credentials = _cred
_fb.firestore = _fs
_fb._apps = {"stub": True}            # so _init_firebase() short-circuits
_fb.initialize_app = lambda *a, **k: None
_cred.Certificate = lambda *a, **k: None


class _FSClient:
    def collection(self, *a, **k):
        return self

    def document(self, *a, **k):
        return self

    def set(self, *a, **k):
        return None

    def update(self, *a, **k):
        return None


_fs.client = lambda *a, **k: _FSClient()
