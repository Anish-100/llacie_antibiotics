import csv
import re
from datetime import datetime, timezone

# A version of this was incorporated into ShortHPISectionRegexStrategy
# It uses the circumstance that EDW replaced newlines with two spaces to
# reformat note text into a more readable form, by hardwrapping it

def fix_note_whitespace(text):
    text = re.sub("  ", "\n", text)
    text = re.sub("\n[ ?]+", "\n", text)
    text = re.sub("\n\n+", "\n\n", text)
    return text.strip()

##################################################################
# Functions for cleaning up list-like text spit out by LLMs
##################################################################

def split_listlike_text(raw_text):
    numeric_list = re.match(r'[1]([).]) +.+?([;,]?) [2]\1( +.+?(\2 |\2? and )\d+\1)*', raw_text)
    bulleted_list = re.match(r'([·•*-] ?)[^\n]+(\n\n?)\1\S', raw_text)

    if raw_text.startswith('\\begin{itemize}'):
        # LLaMA sometimes composes in LaTeX-style lists
        without_prefix = re.sub(r'^\\begin\{itemize\}\s+\\item +', '', raw_text)
        middle = re.sub(r'\s+\\end\{itemize\}[\s\S]*', '', without_prefix)
        values = re.split(r'\s+\\item +', middle)
    elif numeric_list is not None:
        without_suffix = raw_text
        # depending on the style of list, we look for the end of the sentence or paragraph
        end_regex = r'([.]\s+|\n\n)' if numeric_list[1] == ')' else r'[.]?\n\n'
        end_regex_match = re.search(end_regex, raw_text[numeric_list.span()[1]:])
        if end_regex_match is not None:
            without_suffix = raw_text[:numeric_list.span()[1] + end_regex_match.span()[0]]
        middle = re.sub(r'^\d[' + numeric_list[1] + r']', '', without_suffix)
        if numeric_list[2] != '':
            values = re.split(r'(?:' + numeric_list[2] + r'|' + numeric_list[2] + r'? and)? \d+[' + 
                    numeric_list[1] + r']', middle)
        else:
            values = re.split(r'(?: and)? \d+[' + numeric_list[1] + r']', middle)
    elif bulleted_list is not None:
        bullet_regex = r'[' + bulleted_list[1][0] + r']' + bulleted_list[1][1:]
        without_prefix = re.sub(r'^' + bullet_regex, '', raw_text)
        middle = re.sub(r'\n\n[^' + bulleted_list[1][0] + r'][\s\S]*', '', without_prefix)
        values = re.split(bulleted_list[2] + bullet_regex, middle)
    else:
        # Try treating it like a list within a sentence (fragment)
        # Also, excise any superfluous first bullets, numbers, etc
        first_sentence = re.sub(r'^(1[.)]|[·•*-])\s+', '', raw_text)
        first_sentence = re.sub(r'([.]\s+|[.]?\n\n)[\s\S]*', '', first_sentence)
        # If it's not plausibly an inline list, with at least three items, just abort
        if re.search(r'([;,]) \S+.*(\1|\1? and) \S', first_sentence) is None: return None 
        values = re.split(r'[;,] (?:and )?|[;,]? and ', first_sentence)

    return values

def excise_parentheticals(values):
    values = [re.sub(r'\s*[(][^)]+[)]', '', val) for val in values]
    return values

def resplit_list_values(values):
    values = filter(lambda val: re.match(r'No\s+', val, re.IGNORECASE) is None, values)
    values = [re.split(r'[;,] (?:and )?|[;,]? and | */ *(?=[a-zA-Z]{2})|\n', val) for val in values]
    values = [val.strip(' -.') for sublist in values for val in sublist]
    return values

def is_acceptable_value(value):
    if re.match(r'No\s+', value, re.IGNORECASE) is not None: return False
    if re.search(r'[a-z]', value, re.IGNORECASE) is None: return False
    return True

def cleanup_presenting_sx(llm_output_raw):
    values = split_listlike_text(llm_output_raw)
    if values is None: return None
    values = excise_parentheticals(values)
    values = resplit_list_values(values)
    # TODO: excise timing qualifiers (e.g., X days of, for X days, x N days)
    values = filter(is_acceptable_value, values)
    return "\n".join(values)

##################################################################
# Parsing notes.sus CSV exports
##################################################################

# The columns of a notes.sus CSV export
SUS_NOTES_COLUMNS = ["patientEncounterID", "startDTS", "DateOfServiceDTS",
    "NoteStatusDSC", "NoteID", "note_text"]

def parse_sus_notes_csv(path):
    """Parse a notes.sus-style CSV export into a list of note records.

    Each record is a dict with `PatientEncounterID` (int), `NoteID` (int, with the
    letter prefix stripped: LN09019955 -> 9019955, as the database stores integer
    NoteIDs), `startDTS` and `DateOfServiceDTS` (UTC datetimes) and `note_text`.

    The export has a known defect: when a record's quoted note_text field is empty,
    the text itself follows as an unquoted line, which CSV readers see as a malformed
    extra row. Such a line is salvaged as the preceding note's text; unquoted lines
    that do not follow a note with empty text are skipped."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if len(rows) == 0:
        return []
    header = [name.strip() for name in rows[0]]
    missing = [col for col in SUS_NOTES_COLUMNS if col not in header]
    if missing:
        raise ValueError(f"SUS notes CSV is missing expected columns: {', '.join(missing)}")
    idx = {col: header.index(col) for col in SUS_NOTES_COLUMNS}

    notes = []
    for row in rows[1:]:
        if not any(field.strip() for field in row):
            continue
        record = _parse_sus_record(row, idx)
        if record is not None:
            notes.append(record)
        elif len(notes) > 0 and notes[-1]['note_text'] == '':
            # Salvage the export defect: this unquoted line holds the text of the
            # preceding note, whose quoted note_text field came out empty
            notes[-1]['note_text'] = ','.join(row).strip()
        # Any other unparseable line is skipped
    return notes

def _parse_sus_record(row, idx):
    """Parse one well-formed record row of a notes.sus CSV; None if the row is not one."""
    if len(row) < len(idx):
        return None
    try:
        encounter_id = int(row[idx['patientEncounterID']].strip())
        note_id = int(re.sub(r'^[A-Za-z]+', '', row[idx['NoteID']].strip()))
        start_dts = _parse_sus_timestamp(row[idx['startDTS']])
        dos_dts = _parse_sus_timestamp(row[idx['DateOfServiceDTS']])
    except ValueError:
        return None
    return {
        'PatientEncounterID': encounter_id,
        'startDTS': start_dts,
        'DateOfServiceDTS': dos_dts,
        'NoteID': note_id,
        'note_text': row[idx['note_text']],
    }

def _parse_sus_timestamp(value):
    """Parse a SUS export timestamp like 3085-09-28T18:36:00Z into a UTC datetime."""
    dt = datetime.strptime(value.strip(), '%Y-%m-%dT%H:%M:%SZ')
    return dt.replace(tzinfo=timezone.utc)