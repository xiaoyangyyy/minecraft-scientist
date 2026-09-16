"""Optional OS-backed credential lookup without persisting secrets in the repo."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes


DEEPSEEK_CREDENTIAL_TARGET = "minecraft-scientist/deepseek"


class _CredentialW(ctypes.Structure):
    _fields_ = [
        ("flags", wintypes.DWORD),
        ("type", wintypes.DWORD),
        ("target_name", wintypes.LPWSTR),
        ("comment", wintypes.LPWSTR),
        ("last_written", wintypes.FILETIME),
        ("credential_blob_size", wintypes.DWORD),
        ("credential_blob", ctypes.POINTER(ctypes.c_ubyte)),
        ("persist", wintypes.DWORD),
        ("attribute_count", wintypes.DWORD),
        ("attributes", ctypes.c_void_p),
        ("target_alias", wintypes.LPWSTR),
        ("user_name", wintypes.LPWSTR),
    ]


def read_windows_generic_credential(target: str) -> str | None:
    """Read one generic credential for the current Windows user."""

    if sys.platform != "win32":
        return None
    advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    cred_read = advapi32.CredReadW
    cred_read.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(_CredentialW)),
    ]
    cred_read.restype = wintypes.BOOL
    cred_free = advapi32.CredFree
    cred_free.argtypes = [ctypes.c_void_p]
    cred_free.restype = None

    pointer = ctypes.POINTER(_CredentialW)()
    if not cred_read(target, 1, 0, ctypes.byref(pointer)):
        return None
    try:
        credential = pointer.contents
        raw = ctypes.string_at(
            credential.credential_blob,
            credential.credential_blob_size,
        )
        return raw.decode("utf-8")
    finally:
        cred_free(pointer)


def read_deepseek_api_key() -> str | None:
    """Return the project DeepSeek key from Windows Credential Manager."""

    return read_windows_generic_credential(DEEPSEEK_CREDENTIAL_TARGET)
