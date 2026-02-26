from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

import jpype


def init_orekit(data_path: Optional[str] = None) -> None:
    """
    Start JVM with Orekit JARs on classpath, then load orekit-data.
    """

    # ---- Resolve project root & orekit-data path ----
    repo_root = Path.cwd()
    default_zip = repo_root / "data" / "orekit-data.zip"
    default_dir = repo_root / "data"

    if data_path is None:
        data_path = str(default_zip) if default_zip.exists() else str(default_dir)

    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(
            f"orekit-data not found: {p}\n"
            "Put orekit-data.zip into ./data/ or pass data_path explicitly."
        )

    # ---- Build classpath from orekit_jpype jar folder ----
    import orekit_jpype
    jar_dir = Path(orekit_jpype.__file__).resolve().parent / "jars"
    jars = sorted(jar_dir.glob("*.jar"))
    if not jars:
        raise RuntimeError(f"No jar files found in: {jar_dir}")

    # JPype accepts list of jar paths
    classpath_list = [str(j) for j in jars]

    # ---- Start JVM  ----
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=classpath_list, convertStrings=False)

    # ---- Register orekit-data provider ----
    from java.io import File
    from org.orekit.data import ZipJarCrawler, DirectoryCrawler
    from org.orekit.data import DataContext

    # Orekit 13.x: get DataProvidersManager from default DataContext
    mgr = DataContext.getDefault().getDataProvidersManager()

    # 기존 provider를 비우고 싶으면 clear/add 조합을 씁니다.
    mgr.clearProviders()

    if p.is_file() and p.suffix.lower() == ".zip":
        mgr.addProvider(ZipJarCrawler(File(str(p))))
    else:
        mgr.addProvider(DirectoryCrawler(File(str(p))))