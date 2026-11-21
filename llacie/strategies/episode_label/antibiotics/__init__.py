from tqdm import tqdm
import pandas as pd
import json
import re
from ...abstract import AbstractStrategy
from ....tasks.episode_label import AntibioticsEpisodeLabelTask
from ....utils import chunker, echo_info, record_failure
from ....database.query_db import query
SECS_IN_24H = 60 * 60 * 24

class AbstractAntibioticsStrategy(AbstractStrategy):
    """\
    An abstract strategy for converting presenting_sx note features into episode labels using 
    vocab v2. This vocab is UPDATED since the K08 submission after manual examination of the
    misclassifications.
    """
    task = AntibioticsEpisodeLabelTask
    prereq_tasks = []

    BATCH_SIZE = 1000


    def run(self, all_episode_ids, batch_size = None):
        fail_count = 0
        vocab = self.task.vocab
        batch_size = batch_size if batch_size is not None else self.BATCH_SIZE

        echo_info(f"In batches of {batch_size}")
        needs_labels_pb = tqdm(chunker(all_episode_ids, batch_size))

        for ep_ids in needs_labels_pb:
            needs_labels_pb.set_description(
                f"Creating labels ({fail_count} failures/{len(all_episode_ids)} total)")
            df = self.db.get_earliest_notes_with_feature(ep_ids, self.feat_strat, SECS_IN_24H)

            # Some episodes may not have any notes with the required feature
            for ep_id in sorted(set(int(e) for e in ep_ids) - set(df['episode_id'].astype(int))):
                fail_count += 1
                record_failure(self.config, self.name,
                    f"No note with a {self.feat_strat.name} feature", episode_id=ep_id)
            for _, row in df.iterrows():
                try:
                    label_rows = [to_label_row(item) for item in json.loads(row['feature_value'])]
                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    # Not a JSON list of drug objects; leave the episode unfinished
                    fail_count += 1
                    record_failure(self.config, self.name, f"Could not read drug list: {e}",
                        note_id=int(row['id']), episode_id=int(row['episode_id']),
                        note_feature_id=int(row['note_feature_id']))
                    continue
                if len(label_rows) == 0:
                    # No antibiotics found: one empty row marks the episode as finished
                    label_rows = [empty_label_row()]
                self.db.replace_antibiotic_labels(self, row, label_rows)

        echo_info(f"Finished! ({fail_count} failures/{len(all_episode_ids)} total)")


def empty_label_row():
    return {"drug_name": None, "drug_name_raw": None, "start_date": None, "end_date": None,
        "duration_days": None, "is_current": None}


def to_label_row(item):
    """Converts one drug dict from the feature's JSON into an antibiotic_labels row."""
    raw = (item.get('Drug Name') or '').strip()
    try:
        drug_name = query(raw.lower().replace(' ', '_'))[0][0]
    except IndexError:
        drug_name = None    # Not in drugs.db; drug_name_raw keeps what the model wrote
    start_date = parse_date(item.get('Start date'))
    end_date = parse_date(item.get('End date'))
    return {
        **empty_label_row(),
        "drug_name": drug_name,
        "drug_name_raw": raw or None,
        "start_date": start_date,
        "end_date": end_date,
        "duration_days": get_duration(item.get('Duration'), start_date, end_date),
    }


def parse_date(value):
    """Returns a date, or None for empty/'NULL'/unparseable values."""
    if value is None or str(value).strip().upper() in ('', 'NULL', 'NONE'):
        return None
    parsed = pd.to_datetime(value, errors='coerce')
    return None if pd.isna(parsed) else parsed.date()


def get_duration(duration, start_date, end_date):
    """Duration in days: the model's stated Duration if it is a whole number, otherwise
    computed from the dates counting both the start and end day."""
    if isinstance(duration, int) and not isinstance(duration, bool):
        return duration
    if isinstance(duration, str) and duration.strip().isdigit():
        return int(duration.strip())
    if start_date is not None and end_date is not None:
        return (end_date - start_date).days + 1
    return None
