"""Getting CLEVR onto the machine, from whatever state the machine is in.

The harness should run on a clean checkout without a setup ritual, but CLEVR is
a 19 GB archive and a subset evaluation needs a fraction of a percent of it.
So members are resolved through a chain, cheapest first:

    1. already extracted under data/CLEVR_v1.0/   -> free
    2. data/CLEVR_v1.0.zip on disk               -> local read
    3. the published archive, over HTTP ranges   -> fetch just those members
    4. full download, then (2)                   -> only if ranges are refused

Anything fetched is written into `data/CLEVR_v1.0/` in the dataset's own layout,
so it is indistinguishable from a partial manual extraction and case 1 picks it
up next time. A subset build after a cold start pulls the val questions file
(~152 MB) plus one PNG per sampled image, not the archive.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import zipfile
from pathlib import Path

from .remote_zip import HTTPRangeFile, RangeUnsupported, download

CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
ARCHIVE_DIR = "CLEVR_v1.0"
SPLITS = ("train", "val", "test")


def questions_member(split):
    return f"{ARCHIVE_DIR}/questions/CLEVR_{split}_questions.json"


def image_member(split, filename):
    return f"{ARCHIVE_DIR}/images/{split}/{filename}"


def _human(n):
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


class ClevrData:
    """Resolves CLEVR archive members to local files, fetching what is missing."""

    def __init__(
        self,
        data_dir="data",
        url=CLEVR_URL,
        allow_download=True,
        force_full_download=False,
        quiet=False,
    ):
        self.data_dir = Path(data_dir)
        self.url = url
        self.allow_download = allow_download
        self.force_full_download = force_full_download
        self.quiet = quiet
        self.source_used = None

    # -- layout -----------------------------------------------------------
    @property
    def root(self):
        """The dataset root, i.e. what other CLEVR tooling calls CLEVR_v1.0/."""
        return self.data_dir / ARCHIVE_DIR

    @property
    def zip_path(self):
        return self.data_dir / f"{ARCHIVE_DIR}.zip"

    def path(self, member):
        """Local path for an archive member (whether or not it exists yet)."""
        return self.data_dir / member

    def have(self, member):
        p = self.path(member)
        return p.exists() and p.stat().st_size > 0

    def _log(self, msg):
        if not self.quiet:
            print(msg, flush=True)

    # -- acquisition ------------------------------------------------------
    @contextlib.contextmanager
    def _open_archive(self):
        """A ZipFile over the archive, local if possible, remote if not.

        Yields (zipfile, range_reader); the range reader is None for local
        archives and is what lets the caller prefetch exact member extents.
        """
        if self.force_full_download and not self.zip_path.exists():
            self._download_archive()

        if self.zip_path.exists():
            self.source_used = f"local archive {self.zip_path}"
            self._log(f"reading {self.zip_path}")
            with zipfile.ZipFile(self.zip_path) as zf:
                yield zf, None
            return

        if not self.allow_download:
            raise SystemExit(
                "CLEVR is not present and downloads are disabled (--no-download).\n"
                f"Expected extracted data at {self.root} or an archive at {self.zip_path}."
            )

        try:
            self._log(f"opening {self.url} with HTTP range requests")
            remote = HTTPRangeFile(self.url)
        except RangeUnsupported as exc:
            self._log(f"range requests unavailable ({exc}); downloading the full archive")
            self._download_archive()
            self.source_used = f"local archive {self.zip_path}"
            with zipfile.ZipFile(self.zip_path) as zf:
                yield zf, None
            return

        self.source_used = f"remote archive {self.url} (range requests)"
        try:
            with zipfile.ZipFile(remote) as zf:
                yield zf, remote
            self._log(
                f"fetched {_human(remote.bytes_fetched)} "
                f"in {remote.requests_made} range requests"
            )
        finally:
            remote.close()

    def _download_archive(self):
        """Last-resort path: pull the whole 19 GB archive, resumable."""
        if not self.allow_download:
            raise SystemExit("CLEVR archive missing and downloads are disabled.")
        self._log(f"downloading {self.url} -> {self.zip_path} (~18 GB, resumable)")
        free = shutil.disk_usage(self.data_dir.parent).free
        if free < 20 * 1024 ** 3:
            self._log(f"warning: only {_human(free)} free; the archive needs ~18 GB")

        state = {"last": 0}

        def progress(done, total):
            if self.quiet:
                return
            if done - state["last"] > 100 * 1024 ** 2:  # every ~100 MB
                state["last"] = done
                pct = f" ({100 * done / total:.1f}%)" if total else ""
                print(f"  {_human(done)}{pct}", flush=True)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        download(self.url, self.zip_path, progress=progress)

    def _extract(self, zf, member, dest):
        """Extract one member to `dest` atomically."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with zf.open(member) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)

    def ensure(self, members, desc="files"):
        """Make every member available locally; returns their paths."""
        members = list(members)
        missing = [m for m in members if not self.have(m)]
        if missing:
            self._log(f"fetching {len(missing)} {desc} from CLEVR_v1.0")
            with self._open_archive() as (zf, remote):
                available = set(zf.namelist())
                unknown = [m for m in missing if m not in available]
                if unknown:
                    raise SystemExit(f"not in the CLEVR archive: {unknown[0]}")
                # Reading in archive order keeps access sequential, which the
                # local page cache and the remote block cache both prefer.
                missing.sort(key=lambda m: zf.getinfo(m).header_offset)
                for i, member in enumerate(missing, 1):
                    info = zf.getinfo(member)
                    if remote is not None:
                        # Local header + name + extra field, then the data. The
                        # slack covers the header without a second request.
                        remote.prefetch(
                            info.header_offset,
                            info.compress_size + len(member) + 1024,
                        )
                    self._extract(zf, member, self.path(member))
                    if not self.quiet and (i % 100 == 0 or i == len(missing)):
                        print(f"  {i}/{len(missing)} {desc}", flush=True)
        return [self.path(m) for m in members]

    # -- dataset accessors ------------------------------------------------
    def ensure_questions(self, split):
        if split not in SPLITS:
            raise SystemExit(f"unknown split {split!r}; expected one of {SPLITS}")
        return self.ensure([questions_member(split)], f"{split} question file")[0]

    def load_questions(self, split):
        """All questions for a split, with ground-truth answers required."""
        path = self.ensure_questions(split)
        with open(path) as fh:
            questions = json.load(fh)["questions"]
        if not questions or "answer" not in questions[0]:
            raise SystemExit(
                f"split {split!r} has no ground-truth answers -- CLEVR withholds the "
                "test answers. Use --split val (or train)."
            )
        return questions

    def ensure_images(self, split, filenames):
        """Make the named images available; returns their local directory."""
        self.ensure([image_member(split, f) for f in filenames], f"{split} images")
        return self.root / "images" / split

    def status(self):
        """What is present locally, for `clevrbench status`."""
        info = {
            "data_dir": str(self.data_dir),
            "root": str(self.root),
            "archive_present": self.zip_path.exists(),
            "splits": {},
        }
        for split in SPLITS:
            image_dir = self.root / "images" / split
            info["splits"][split] = {
                "questions": self.have(questions_member(split)),
                "images_present": (
                    sum(1 for _ in image_dir.glob("*.png")) if image_dir.is_dir() else 0
                ),
            }
        return info
