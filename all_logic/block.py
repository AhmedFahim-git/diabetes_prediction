from __future__ import annotations

import os

from prefect.filesystems import LocalFileSystem

fs = LocalFileSystem(basepath=os.getcwd())
fs.save("mylocal")
