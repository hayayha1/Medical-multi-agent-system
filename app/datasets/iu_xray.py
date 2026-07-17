from functools import lru_cache
from pathlib import Path


SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg"}


class DatasetCaseNotFound(FileNotFoundError):
    pass


class IUXRayDataset:
    """Path-safe IU X-Ray image resolver tolerant of common extracted layouts."""

    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()

    def _ensure_root(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"IU X-Ray dataset root does not exist: {self.root}")

    def resolve_relative_paths(self, paths: list[str]) -> list[str]:
        self._ensure_root()
        resolved: list[str] = []
        for relative in paths:
            candidate = (self.root / relative).resolve()
            if self.root not in candidate.parents:
                raise ValueError(f"Image path escapes dataset root: {relative}")
            if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_IMAGES:
                raise DatasetCaseNotFound(f"Invalid IU X-Ray image: {relative}")
            resolved.append(str(candidate))
        return resolved

    @lru_cache(maxsize=1024)
    def find_case_images(self, case_id: str) -> tuple[str, ...]:
        self._ensure_root()
        needle = case_id.strip().lower()
        if not needle:
            raise ValueError("case_id cannot be empty")
        matches = [
            path for path in self.root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGES
            and needle in path.stem.lower()
        ]
        if not matches:
            raise DatasetCaseNotFound(f"No IU X-Ray images found for case: {case_id}")
        return tuple(str(path.resolve()) for path in sorted(matches)[:4])

    def resolve_request(
        self,
        study_uid: str,
        dataset_case_id: str | None,
        image_paths: list[str],
    ) -> list[str]:
        if image_paths:
            return self.resolve_relative_paths(image_paths)
        return list(self.find_case_images(dataset_case_id or study_uid))

