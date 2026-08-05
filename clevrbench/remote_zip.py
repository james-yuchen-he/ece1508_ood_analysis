"""A seekable file object over HTTP range requests.

CLEVR ships as one 19 GB zip. Evaluating a 1,000-question subset needs the val
questions file and a few hundred PNGs out of it -- a couple of hundred
megabytes. Since the host advertises `Accept-Ranges: bytes`, the archive can be
read in place: hand this object to `zipfile.ZipFile` and stdlib does the rest
(central directory parsing, ZIP64, decompression), fetching only the byte
ranges it touches.

Reads are served through an aligned block cache, because zipfile issues many
small reads and one HTTP request per 4 KB read would be unusable.

Falls back cleanly: `RangeUnsupported` is raised if the server won't do partial
content, and the caller downloads the archive the ordinary way instead.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request

USER_AGENT = "clevrbench/0.1 (+https://cs.stanford.edu/people/jcjohns/clevr/)"
DEFAULT_BLOCK = 1 << 20  # 1 MiB


class RangeUnsupported(RuntimeError):
    """The server will not serve partial content for this URL."""


def _request(url, headers, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    return urllib.request.urlopen(req, timeout=timeout)


def head(url, timeout=30):
    """(content_length, accepts_ranges) for a URL, following redirects."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        length = resp.headers.get("Content-Length")
        accepts = (resp.headers.get("Accept-Ranges") or "").lower() == "bytes"
    return (int(length) if length else None, accepts)


class HTTPRangeFile(io.RawIOBase):
    """Random-access reads over HTTP, cached in aligned blocks."""

    def __init__(self, url, block_size=DEFAULT_BLOCK, timeout=60, max_retries=5):
        self.url = url
        self.block_size = block_size
        self.timeout = timeout
        self.max_retries = max_retries
        self._pos = 0
        self._cache_start = 0
        self._cache = b""
        self.bytes_fetched = 0
        self.requests_made = 0

        size, accepts = head(url, timeout=timeout)
        if size is None:
            raise RangeUnsupported(f"{url} did not report a Content-Length")
        if not accepts:
            raise RangeUnsupported(f"{url} does not advertise byte ranges")
        self.size = size

    # -- io plumbing ------------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        self._pos = max(0, new)
        return self._pos

    # -- transport --------------------------------------------------------
    def _fetch(self, start, end):
        """GET bytes [start, end] inclusive, with retry and backoff."""
        end = min(end, self.size - 1)
        if start > end:
            return b""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                with _request(
                    self.url, {"Range": f"bytes={start}-{end}"}, self.timeout
                ) as resp:
                    if resp.status != 206:
                        raise RangeUnsupported(
                            f"expected 206 Partial Content, got {resp.status}"
                        )
                    data = resp.read()
                self.requests_made += 1
                self.bytes_fetched += len(data)
                return data
            except RangeUnsupported:
                raise
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 30))
        raise OSError(f"range request failed after {self.max_retries} tries: {last_error}")

    def _cached(self, pos):
        return self._cache_start <= pos < self._cache_start + len(self._cache)

    def prefetch(self, start, length):
        """Pull an exact byte range into the cache in a single request.

        Callers that know a member's extent -- the zip central directory gives
        it -- use this so a 183 KB image costs one 183 KB request instead of a
        whole cache block.
        """
        start = max(0, min(start, self.size))
        end = min(start + length, self.size) - 1
        if end < start:
            return
        self._cache = self._fetch(start, end)
        self._cache_start = start

    def readinto(self, buffer):
        if self._pos >= self.size:
            return 0
        want = min(len(buffer), self.size - self._pos)
        written = 0
        while written < want:
            pos = self._pos + written
            if not self._cached(pos):
                start = (pos // self.block_size) * self.block_size
                self._cache = self._fetch(start, start + self.block_size - 1)
                self._cache_start = start
                if not self._cache:
                    break
            offset = pos - self._cache_start
            chunk = self._cache[offset : offset + (want - written)]
            buffer[written : written + len(chunk)] = chunk
            written += len(chunk)
        self._pos += written
        return written

    def read(self, size=-1):
        if size is None or size < 0:
            size = max(0, self.size - self._pos)
        buffer = bytearray(size)
        got = self.readinto(buffer)
        return bytes(buffer[:got])


def download(url, dest, chunk=1 << 22, timeout=60, progress=None):
    """Stream a URL to `dest`, resuming a partial file if one is there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total, accepts = head(url, timeout=timeout)
    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        return dest
    headers, mode = {}, "wb"
    if have and accepts:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
    else:
        have = 0

    with _request(url, headers, timeout) as resp, open(dest, mode) as fh:
        done = have
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if progress:
                progress(done, total)
    return dest
