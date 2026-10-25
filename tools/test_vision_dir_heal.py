#!/usr/bin/env python3
"""Regression test: a long-running server survives its vision directory being deleted.

Background: `Vision.work_dir()` makes a `strata-vision-xxxx` directory under the system temp directory once, in
`__init__`.  A cleaner such as `systemd-tmpfiles` (`q /tmp 1777 root root 10d`) removes directories in /tmp by age, so
a server that ran for days outlived its own directory and every request with an image died with

    FileNotFoundError: [Errno 2] No such file or directory: '/tmp/strata-vision-kaswj9bb/<hash>.img'

(or, for an image still in the cache, on the `.sve` that prepare() stats).  Nothing re-created the directory, so only
a restart cured it.

This test needs no GPU, no model and no vision encoder: it uses a fake encoder speaking the same pipe protocol
(`READY`, `ENC <in> <out>` -> `OK <tokens>`), and it drives the real Vision class.

    python tools/test_vision_dir_heal.py
"""
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import serve.server as server  # noqa: E402


FAKE_ENCODER = r'''
import sys, pathlib
print("READY", flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    if line == "QUIT":
        break
    if not line.startswith("ENC "):
        print("ERR what", flush=True)
        continue
    _, src, out = line.split()
    data = pathlib.Path(src).read_bytes()
    pathlib.Path(out).write_bytes(b"E" * 1024)
    print("OK %d" % max(1, len(data) // 4), flush=True)
'''

PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082")


def make_vision(tmp: Path) -> "server.Vision":
    """A Vision whose encoder is the fake above, working in `tmp`."""
    script = tmp / "fake_encoder.py"
    script.write_text(FAKE_ENCODER)
    vision = server.Vision.__new__(server.Vision)          # bypass __init__: no model files, no GPU
    vision.dir = tmp / "strata-vision-fake"
    vision.dir.mkdir()
    vision.lock = threading.Lock()
    vision.cache = {}
    vision.log = None
    import subprocess
    proc = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1, cwd=vision.dir)
    vision.proc = proc
    vision.stopped = False
    vision.spawn = ([sys.executable, str(script)], None, None)
    line = proc.stdout.readline().strip()
    assert line == "READY", f"the fake encoder did not start: {line!r}"
    return vision


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        return False
    print(f"  ok    {name}")
    return True


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 1) the plain path still works
        v = make_vision(tmp / "a") if (tmp / "a").mkdir() or True else None
        out, ntok = v.encode(PNG_1x1)
        ok &= check("everyday encode writes a .sve",
                    lambda: (_ for _ in ()).throw(AssertionError(f"missing {out}")) if not out.is_file() else None)
        ok &= check("everyday encode reports image tokens", lambda: None if ntok > 0 else (_ for _ in ()).throw(
            AssertionError(f"tokens={ntok}")))

        # 2) the directory is deleted, and an image NOT in the cache arrives: the write used to raise FileNotFoundError
        shutil.rmtree(v.dir)
        fresh = PNG_1x1 + b"different image"                # a different hash, so it misses the cache
        def healed_uncached():
            got, n = v.encode(fresh)
            assert got.is_file(), f"no embeddings file at {got}"
            assert v.dir.is_dir(), "the encoder's directory was not re-created"
        ok &= check("a removed directory is re-created for a new image", healed_uncached)

        # 3) the directory is deleted while an image IS cached: the cached .sve is gone with it, so the cache entry
        #    must be dropped and the image encoded again (prepare() used to die on stat())
        def healed_cached():
            shutil.rmtree(v.dir)
            got, n = v.encode(fresh)
            assert got.is_file(), f"the cached .sve is gone and was not replaced: {got}"
            assert got.parent == v.dir, "the embeddings went somewhere else"
        ok &= check("a stale cache entry is dropped and re-encoded", healed_cached)

        # 4) the cache never keeps an entry whose file is missing, on any later hit
        def cache_is_honest():
            got, _ = v.encode(fresh)
            got.unlink()                                   # an eviction by something else
            again, _ = v.encode(fresh)
            assert again.is_file(), "a cache hit returned a path whose file does not exist"
        ok &= check("a cache hit never returns a missing file", cache_is_honest)

        v.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
