import csv
from zipfile import ZipFile

from package_period_frontier_return_rates import package


def _create_csv(root, profile, filename):
    path = (
        root / profile / "final_shock_frontier_return_rates" / "users_50"
        / "max_hardware_invest_usd_50000" / "confidential_document_fraction_0p25"
        / "access_plan_pay_per_use" / filename
    )
    path.parent.mkdir(parents=True)
    path.write_text("period,month\n0,0\n", encoding="utf-8")
    return path


def test_package_uses_short_paths_and_manifest_by_default(tmp_path):
    first = _create_csv(tmp_path, "mixed", "first.csv")
    second = _create_csv(tmp_path, "software_engineering_heavy", "second.csv")
    output = tmp_path / "portable.zip"

    assert package(tmp_path, output) == 2

    with ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "manifest.csv" in names
        assert "mixed/000001.csv" in names
        assert "software_engineering_heavy/000002.csv" in names
        rows = list(csv.DictReader(archive.read("manifest.csv").decode().splitlines()))

    assert rows == [
        {"archive_path": "mixed/000001.csv", "source_path": first.relative_to(tmp_path).as_posix()},
        {
            "archive_path": "software_engineering_heavy/000002.csv",
            "source_path": second.relative_to(tmp_path).as_posix(),
        },
    ]


def test_package_legacy_layout_preserves_source_paths(tmp_path):
    source = _create_csv(tmp_path, "mixed", "source.csv")
    output = tmp_path / "legacy.zip"

    assert package(tmp_path, output, legacy_layout=True) == 1

    with ZipFile(output) as archive:
        assert source.relative_to(tmp_path).as_posix() in archive.namelist()
        assert "manifest.csv" not in archive.namelist()
