"""Minimal imghdr shim for environments missing the stdlib module.

This implements a tiny subset of the stdlib imghdr.what() behavior
sufficient for Streamlit's use (detect common image formats by header).
"""
from __future__ import annotations

def _bytes_prefix(b, prefix):
    return isinstance(b, (bytes, bytearray)) and b.startswith(prefix)

def what(file, h=None):
    """Detect image type.

    Arguments:
    - file: filename or None (if h provided)
    - h: initial bytes (optional)

    Returns: one of 'jpeg','png','gif','bmp','tiff','webp' or None
    """
    if h is None:
        try:
            with open(file, 'rb') as f:
                h = f.read(32)
        except Exception:
            return None
    if not h:
        return None
    b = h if isinstance(h, (bytes, bytearray)) else bytes(h)

    if _bytes_prefix(b, b'\xff\xd8\xff'):
        return 'jpeg'
    if _bytes_prefix(b, b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if _bytes_prefix(b, b'GIF87a') or _bytes_prefix(b, b'GIF89a'):
        return 'gif'
    if _bytes_prefix(b, b'BM'):
        return 'bmp'
    if _bytes_prefix(b, b'II*\x00') or _bytes_prefix(b, b'MM\x00*'):
        return 'tiff'
    # WEBP is RIFF....WEBP
    if _bytes_prefix(b, b'RIFF') and len(b) >= 12 and b[8:12] == b'WEBP':
        return 'webp'
    return None

__all__ = ['what']
