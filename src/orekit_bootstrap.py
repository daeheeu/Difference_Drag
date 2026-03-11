from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional
import os
import jpype


def _existing_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp.exists() and str(rp) not in seen:
            out.append(rp)
            seen.add(str(rp))
    return out


def _split_env_paths(value: Optional[str]) -> List[Path]:
    if not value:
        return []
    parts = [x.strip() for x in value.split(os.pathsep) if x.strip()]
    return [Path(x) for x in parts]


def _default_data_roots(repo_root: Path) -> List[Path]:
    return [
        repo_root / "data" / "orekit-data.zip",
        repo_root / "data" / "orekit-data",
        repo_root / "data",
    ]


def _extra_data_roots(repo_root: Path, extra_data_paths: Optional[Iterable[str]]) -> List[Path]:
    roots: List[Path] = []

    # explicit args
    if extra_data_paths:
        roots.extend(Path(p) for p in extra_data_paths if p)

    # env overrides
    roots.extend(_split_env_paths(os.environ.get("OREKIT_EXTRA_DATA")))
    roots.extend(_split_env_paths(os.environ.get("OREKIT_EOP_PATH")))
    roots.extend(_split_env_paths(os.environ.get("OREKIT_SPACE_WEATHER_PATH")))

    # conventional local folders
    roots.extend(
        [
            repo_root / "data" / "eop",
            repo_root / "data" / "space-weather",
            repo_root / "data" / "space_weather",
        ]
    )

    return roots


def _add_provider(mgr, path: Path) -> None:
    from java.io import File
    from org.orekit.data import ZipJarCrawler, DirectoryCrawler

    if path.is_file() and path.suffix.lower() == ".zip":
        mgr.addProvider(ZipJarCrawler(File(str(path))))
    elif path.is_dir():
        mgr.addProvider(DirectoryCrawler(File(str(path))))
    else:
        raise FileNotFoundError(f"Unsupported Orekit data path: {path}")


def init_orekit(
    data_path: Optional[str] = None,
    extra_data_paths: Optional[Iterable[str]] = None,
    clear_providers: bool = True,
) -> None:
    """
    Start JVM with Orekit JARs on classpath, then register one or more Orekit data roots.

    Priority:
    1) explicit data_path
    2) default local orekit-data roots
    3) optional extra data roots (EOP / space weather / user paths)
    """

    repo_root = Path.cwd()

    # ---- Build classpath from orekit_jpype jar folder ----
    import orekit_jpype

    jar_dir = Path(orekit_jpype.__file__).resolve().parent / "jars"
    jars = sorted(jar_dir.glob("*.jar"))
    if not jars:
        raise RuntimeError(f"No jar files found in: {jar_dir}")

    classpath_list = [str(j) for j in jars]

    # ---- Start JVM ----
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=classpath_list, convertStrings=False)

    # ---- Build provider root list ----
    roots: List[Path] = []

    if data_path is not None:
        roots.append(Path(data_path))
    else:
        roots.extend(_default_data_roots(repo_root))

    roots.extend(_extra_data_roots(repo_root, extra_data_paths))
    roots = _existing_paths(roots)

    if not roots:
        raise FileNotFoundError(
            "No Orekit data roots found.\n"
            "Expected one of:\n"
            "  ./data/orekit-data.zip\n"
            "  ./data/orekit-data/\n"
            "  ./data/\n"
            "Or pass data_path / extra_data_paths explicitly."
        )

    # ---- Register providers ----
    from org.orekit.data import DataContext

    mgr = DataContext.getDefault().getDataProvidersManager()

    if clear_providers:
        mgr.clearProviders()

    for root in roots:
        _add_provider(mgr, root)

    # Optional runtime trace
    print("[init_orekit] Registered Orekit data roots:")
    for root in roots:
        print(f"  - {root}")