"""Contracts for source and installed command-line entry points."""

from importlib.resources import as_file, files
from pathlib import Path
from types import SimpleNamespace

from connectomics.data import download
from connectomics.runtime.dispatch import prepare_cli_args


def test_demo_tutorial_matches_packaged_resource():
    resource = files("connectomics.config").joinpath("demo/minimal.yaml")
    tutorial = Path(__file__).resolve().parents[2] / "tutorials" / "minimal.yaml"
    assert resource.read_bytes() == tutorial.read_bytes()


def test_demo_uses_packaged_config_from_unrelated_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(demo=True, config=None, fast_dev_run=0, mode="test")
    resource = files("connectomics.config").joinpath("demo/minimal.yaml")
    with as_file(resource) as config_path:
        prepare_cli_args(args, config_path)
        assert Path(args.config).read_bytes() == resource.read_bytes()
    assert args.mode == "train"
    assert args.fast_dev_run == 1


def test_cli_dispatches_packaged_cpu_demo(monkeypatch, tmp_path):
    from connectomics import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["pytc", "--demo"])
    dispatched = []
    monkeypatch.setattr(cli, "dispatch_runtime", lambda args, cfg: dispatched.append((args, cfg)))
    cli.main()
    args, cfg = dispatched.pop()
    assert args.fast_dev_run == 1
    assert cfg.system.accelerator == "cpu"
    assert cfg.system.num_gpus == 0
    assert cfg.data.train.image.startswith("random://")


def test_download_list_does_not_download(monkeypatch, capsys):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("--list must not download data")

    monkeypatch.setattr(download, "download_dataset", unexpected_download)
    assert download.main(["--list"]) == 0
    assert "lucchi" in capsys.readouterr().out


def test_download_cli_forwards_options_and_reports_failures(monkeypatch, tmp_path):
    calls = []

    def fake_download(name, base_dir, force):
        calls.append((name, base_dir, force))
        return name != "unknown"

    monkeypatch.setattr(download, "download_dataset", fake_download)
    assert download.main(["lucchi", "unknown", "--output", str(tmp_path), "--force"]) == 1
    assert calls == [("lucchi", tmp_path, True), ("unknown", tmp_path, True)]
