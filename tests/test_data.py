import pandas as pd
import pytest

from parkinsons_voice.data import (
    PPMIDataNotFoundError,
    build_participant_master,
    cohort_summary,
    find_file,
    load_csv_by_partial,
    require_data_dir,
)


def test_find_file_matches_partial_name(tmp_path):
    (tmp_path / "Demographics_2024-01-01.csv").write_text("PATNO\n1\n", encoding="utf-8")
    (tmp_path / "Participant_Status_2024-01-01.csv").write_text("PATNO\n1\n", encoding="utf-8")

    found = find_file("Demographics", tmp_path)
    assert found is not None
    assert "Demographics" in found


def test_find_file_returns_none_when_missing(tmp_path):
    assert find_file("NoSuchTable", tmp_path) is None


def test_load_csv_by_partial_reads_dataframe(tmp_path):
    (tmp_path / "Demographics_2024.csv").write_text("PATNO,SEX\n1,M\n2,F\n", encoding="utf-8")
    df = load_csv_by_partial("Demographics", tmp_path)
    assert df is not None
    assert list(df.columns) == ["PATNO", "SEX"]
    assert len(df) == 2


def test_load_csv_by_partial_none_when_missing(tmp_path):
    assert load_csv_by_partial("Demographics", tmp_path) is None


def test_require_data_dir_raises_clear_error(tmp_path):
    missing = tmp_path / "PPMI_raw"
    with pytest.raises(PPMIDataNotFoundError, match="ppmi-info.org"):
        require_data_dir(missing)


def test_require_data_dir_passes_when_present(tmp_path):
    present = tmp_path / "PPMI_raw"
    present.mkdir()
    assert require_data_dir(present) == present


def test_build_participant_master_filters_and_merges():
    ps = pd.DataFrame(
        {
            "PATNO": [1, 2, 3],
            "ENROLL_STATUS": ["Enrolled", "Screened", "Withdrew"],
        }
    )
    demo = pd.DataFrame({"PATNO": [1, 2, 3], "SEX": ["M", "F", "M"]})

    master = build_participant_master(ps, demo)
    assert master is not None
    # 'Screened' is not in the valid_status allowlist, so PATNO 2 is dropped
    assert sorted(master["PATNO"].tolist()) == [1, 3]
    assert "SEX" in master.columns


def test_build_participant_master_requires_participant_status():
    assert build_participant_master(None, None) is None


def test_cohort_summary_crosstab():
    ps = pd.DataFrame(
        {
            "COHORT_DEFINITION": ["PD", "PD", "Control"],
            "ENROLL_STATUS": ["Enrolled", "Enrolled", "Enrolled"],
        }
    )
    summary = cohort_summary(ps)
    assert summary is not None
    assert summary.loc["PD", "enrolled"] == 2
    assert summary.loc["Control", "enrolled"] == 1
