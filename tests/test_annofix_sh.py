"""scripts/annofix.sh: works on any machine when given a folder; the built-in dataset folder is
only a default where it exists."""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "annofix.sh"
DEFAULT = Path("Documents/Honours/SamsungProject/DataSet/Annotations-20261030T173246Z-1-001/Annotations")


def run(tmp_path, *args):
    """Run the script with a fake uv (prints its arguments) and HOME in tmp_path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "uv"
    fake.write_text('#!/usr/bin/env bash\necho "uv $*"\n')
    fake.chmod(0o755)
    env = {**os.environ, "HOME": str(tmp_path / "home"), "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env)


def test_a_samples_folder_opens_with_fixes_next_to_it(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    r = run(tmp_path, str(samples))
    assert r.returncode == 0, r.stderr
    assert f"annofix.cvat {samples} --out {tmp_path / 'samples_fixed'}" in r.stdout


def test_extra_options_after_a_double_dash_reach_annofix(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    r = run(tmp_path, str(samples), str(tmp_path / "out"), "--", "--device", "cpu")
    assert r.returncode == 0, r.stderr
    assert r.stdout.rstrip().endswith(f"--out {tmp_path / 'out'} --device cpu")


def test_without_a_folder_on_another_machine_it_explains_instead_of_failing_on_a_path(tmp_path):
    r = run(tmp_path)
    assert r.returncode == 2
    assert "usage" in r.stderr and "SAMPLES" in r.stderr
    assert "no such folder" not in r.stderr


def test_without_a_folder_it_uses_the_dataset_where_that_exists(tmp_path):
    (tmp_path / "home" / DEFAULT).mkdir(parents=True)
    r = run(tmp_path)
    assert r.returncode == 0, r.stderr
    assert f"annofix.cvat {tmp_path / 'home' / DEFAULT} --out" in r.stdout
