"""Unit tests for parsing notes.sus-style CSV exports (text_wrangling.parse_sus_notes_csv)."""
from datetime import timezone
from pathlib import Path

import pytest

from llacie.text_wrangling import parse_sus_notes_csv

REAL_CSV_PATH = Path(__file__).parents[2] / "examples" / "antibiotics" / "actual" / \
    "notes.sus-infxn.consistent-dates.200.csv"

# A minimal export reproducing the structure of the real notes.sus CSV,
# including the known export defect: the second note has an empty quoted
# note_text field, and its text was written as an unquoted line after the record.
SAMPLE_CSV = '\r\n'.join([
    '"patientEncounterID","startDTS","DateOfServiceDTS","NoteStatusDSC","NoteID","note_text"',
    '"9991070575","3085-09-28T18:36:00Z","3085-09-28T20:35:00Z","Signed","LN09019955","Medical ICU History and Physical\n\nChief Complaint: SOB"',
    '"9990869700","3083-10-28T01:17:00Z","3083-10-28T15:20:00Z","Signed","LN00056936",""',
    'Mr. Garza presents with two weeks of ongoing ocular infection, and agrees to surgery',
    '"9992927509","3090-04-05T06:56:00Z","3090-04-05T17:06:00Z","Signed","LN01437082","IM ADMISSION HISTORY & PHYSICAL NOTE"',
    '',
])


@pytest.fixture
def sample_csv(tmp_path):
    csv_path = tmp_path / "notes.sus.csv"
    csv_path.write_text(SAMPLE_CSV, encoding='utf-8')
    return csv_path


def test_splits_into_individual_notes(sample_csv):
    notes = parse_sus_notes_csv(sample_csv)
    assert len(notes) == 3


def test_metadata_is_converted_to_database_types(sample_csv):
    notes = parse_sus_notes_csv(sample_csv)
    first = notes[0]
    assert first['PatientEncounterID'] == 9991070575
    assert first['NoteID'] == 9019955  # LN prefix stripped
    assert first['startDTS'].tzinfo is timezone.utc
    assert first['startDTS'].strftime('%Y-%m-%dT%H:%M:%SZ') == '3085-09-28T18:36:00Z'
    assert first['DateOfServiceDTS'].strftime('%Y-%m-%dT%H:%M:%SZ') == '3085-09-28T20:35:00Z'
    assert first['note_text'].startswith('Medical ICU History and Physical')


def test_salvages_unquoted_note_text(sample_csv):
    """The unquoted line following the record with empty note_text is its text."""
    notes = parse_sus_notes_csv(sample_csv)
    salvaged = next(n for n in notes if n['NoteID'] == 56936)
    assert salvaged['note_text'] == \
        'Mr. Garza presents with two weeks of ongoing ocular infection, and agrees to surgery'


def test_row_after_salvaged_note_is_not_affected(sample_csv):
    notes = parse_sus_notes_csv(sample_csv)
    last = notes[2]
    assert last['NoteID'] == 1437082
    assert last['note_text'] == 'IM ADMISSION HISTORY & PHYSICAL NOTE'


def test_unparseable_row_without_preceding_empty_note_is_skipped(tmp_path):
    csv_path = tmp_path / "notes.sus.csv"
    csv_path.write_text('\r\n'.join([
        '"patientEncounterID","startDTS","DateOfServiceDTS","NoteStatusDSC","NoteID","note_text"',
        '"9991070575","3085-09-28T18:36:00Z","3085-09-28T20:35:00Z","Signed","LN09019955","Some note"',
        'a completely unquoted stray line',
    ]), encoding='utf-8')
    notes = parse_sus_notes_csv(csv_path)
    assert len(notes) == 1
    assert notes[0]['note_text'] == 'Some note'


def test_missing_required_columns_raises(tmp_path):
    csv_path = tmp_path / "notes.sus.csv"
    csv_path.write_text('"NoteID","note_text"\n"LN1","text"', encoding='utf-8')
    with pytest.raises(ValueError, match="missing expected columns"):
        parse_sus_notes_csv(csv_path)


def test_empty_file_returns_no_notes(tmp_path):
    csv_path = tmp_path / "notes.sus.csv"
    csv_path.write_text('"patientEncounterID","startDTS","DateOfServiceDTS",'
        '"NoteStatusDSC","NoteID","note_text"\r\n', encoding='utf-8')
    assert parse_sus_notes_csv(csv_path) == []


@pytest.mark.skipif(not REAL_CSV_PATH.exists(), reason="real notes.sus CSV not present")
def test_real_notes_sus_csv_yields_200_notes():
    notes = parse_sus_notes_csv(REAL_CSV_PATH)
    assert len(notes) == 200
    assert all(n['note_text'] for n in notes)
    assert len({n['NoteID'] for n in notes}) == 200
    assert len({n['PatientEncounterID'] for n in notes}) == 200
    # All notes are within 24h of admission, so they pass the SECS_IN_24H duration check
    assert all((n['DateOfServiceDTS'] - n['startDTS']).total_seconds() < 60 * 60 * 24
        for n in notes)
