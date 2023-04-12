from pathlib import Path
from typing import Iterator, Optional

from vcap import BaseCapsule
from vcap.loading.capsule_loading import load_capsule
from vcap.loading.packaging import CAPSULE_EXTENSION, package_capsule


class CapsuleDir:

    def __init__(self, capsule_dir: Path):
        self.capsule_dir = capsule_dir

    def package_capsules(self, output_dir: Optional[Path] = None):
        unpackaged_capsules = (path for path in self.capsule_dir.iterdir()
                               if path.is_dir())

        output_dir = self.capsule_dir if output_dir is None else output_dir

        for unpackaged_capsule in unpackaged_capsules:
            capsule_name = unpackaged_capsule \
                .with_suffix(CAPSULE_EXTENSION) \
                .name

            packaged_capsule_path = output_dir / capsule_name

            package_capsule(unpackaged_capsule, packaged_capsule_path)

    def __iter__(self) -> Iterator[BaseCapsule]:
        """Iterates through directory, returning loaded capsules"""
        capsule_files = self.capsule_dir.glob(f"*{CAPSULE_EXTENSION}")

        for capsule_file in capsule_files:
            capsule = load_capsule(capsule_file)
            yield capsule

    def __len__(self) -> int:
        """Count of (packaged) capsules in directory"""
        capsule_files = self.capsule_dir.glob(f"*{CAPSULE_EXTENSION}")
        return len(list(capsule_files))
