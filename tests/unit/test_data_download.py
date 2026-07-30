from __future__ import annotations

import shutil
import zipfile

from connectomics.data import download


def test_lucchi_archive_extracts_into_dataset_directory(tmp_path, monkeypatch):
    source_archive = tmp_path / "source.zip"
    expected_files = ("train_im.h5", "train_mito.h5", "test_im.h5", "test_mito.h5")
    with zipfile.ZipFile(source_archive, "w") as archive:
        for filename in expected_files:
            archive.writestr(filename, filename)

    def fake_urlretrieve(url, filename, reporthook=None):
        shutil.copyfile(source_archive, filename)
        return filename, None

    monkeypatch.setattr(download, "urlretrieve", fake_urlretrieve)

    assert download.download_dataset("lucchi++", tmp_path)

    extract_dir = tmp_path / "datasets" / "lucchi++"
    assert sorted(path.name for path in extract_dir.iterdir()) == sorted(expected_files)
    assert not (tmp_path / "datasets" / "lucchi++.zip").exists()
    assert download.DATASETS["lucchi++"]["url"].endswith("/mito_lucchi%2B%2B/lucchi%2B%2B.zip")
