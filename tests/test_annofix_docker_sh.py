"""scripts/annofix-docker.sh: runs compose with the folders, the user's ids and AnnoFix options."""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "annofix-docker.sh"
COMPOSE = Path(__file__).parent.parent / "compose.yaml"


def run(tmp_path, *args, cwd=None):
    """Run the script with a fake docker that prints its arguments and the compose variables."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "docker"
    fake.write_text('#!/usr/bin/env bash\n'
                    'echo "docker $*"\n'
                    'echo "SAMPLES=$SAMPLES OUT=$OUT UID=$HOST_UID GID=$HOST_GID ARGS=$ANNOFIX_ARGS"\n')
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    env.pop("ANNOFIX_ARGS", None)
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, cwd=cwd)


def test_samples_open_in_the_cpu_image_with_fixes_next_to_them(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    r = run(tmp_path, str(samples))
    assert r.returncode == 0, r.stderr
    assert f"docker compose -f {COMPOSE} up --build annofix" in r.stdout
    assert f"SAMPLES={samples} OUT={tmp_path / 'samples_fixed'} UID={os.getuid()} GID={os.getgid()}" in r.stdout
    assert (tmp_path / "samples_fixed").is_dir()     # made by you, so it is yours, not root's


def test_gpu_and_annofix_options(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    r = run(tmp_path, "--gpu", str(samples), str(tmp_path / "out"), "--", "--annotator", "sha")
    assert r.returncode == 0, r.stderr
    assert "--profile gpu up --build annofix-gpu" in r.stdout
    assert f"OUT={tmp_path / 'out'}" in r.stdout and "ARGS=--annotator sha" in r.stdout


def test_relative_folders_become_absolute(tmp_path):
    (tmp_path / "samples").mkdir()
    r = run(tmp_path, "samples", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert f"SAMPLES={tmp_path / 'samples'} " in r.stdout


def test_without_a_samples_folder_it_explains(tmp_path):
    r = run(tmp_path)
    assert r.returncode == 2 and "usage" in r.stderr
    r = run(tmp_path, str(tmp_path / "missing"))
    assert r.returncode == 1 and "no such folder" in r.stderr
