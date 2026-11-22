import csv
import random
from datetime import datetime
from os import path, makedirs
from click import echo, secho, get_current_context
from collections import namedtuple

PACKAGE_DIR = path.dirname(__file__)

def chunker(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def _force_color_in_workers():
    try:
        from llacie import app
        ctx = get_current_context()
        current_app = ctx.find_object(app.App)
        if current_app is None: return None
        return current_app.is_worker or None
    except RuntimeError:
        pass
    return None

def echo_err(message):
    secho(message, fg='red', err=True, color=_force_color_in_workers())

def echo_warn(message):
    secho(message, fg='yellow', err=True, color=_force_color_in_workers())

def echo_info(message):
    secho(message, fg='green', err=True, color=_force_color_in_workers())

DEFAULT_FAILURES_FILE = "logs/failed_notes.csv"
FAILURES_COLUMNS = ["timestamp", "strategy", "note_id", "episode_id", "note_feature_id", "reason"]

def _append_failure_row(failures_file, row):
    """Appends one row to a failures CSV, creating it with the header row if needed."""
    if path.dirname(failures_file): makedirs(path.dirname(failures_file), exist_ok=True)
    is_new = not path.isfile(failures_file) or path.getsize(failures_file) == 0
    with open(failures_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new: writer.writerow(FAILURES_COLUMNS)
        writer.writerow(row)

def record_failure(config, strategy_name, reason, note_id=None, episode_id=None,
        note_feature_id=None, response=None):
    """Warns about a note/episode a strategy could not process and appends it to the failures
    CSV (FAILURES_FILE in .env, default logs/failed_notes.csv), so failures can be reviewed
    after the run.

    If given, `response` (e.g. the model's raw reply) is saved to its own text file in a
    failed_responses/ folder next to the CSV, and the reason names that file.

    The current directory is not always writable (e.g. inside a container); if the failures
    file can't be written there, it falls back to ~/.local/state/llacie/, and if no writable
    location exists the failure is only reported on stderr. Problems logging a failure never
    crash the run."""
    ids = ", ".join(f"{k} {v}" for k, v in [("note", note_id), ("episode", episode_id),
        ("note feature", note_feature_id)] if v is not None)

    failures_file = (config.get('FAILURES_FILE') if config is not None else None) \
        or DEFAULT_FAILURES_FILE
    dir_name = path.dirname(failures_file)

    target_dir = dir_name or '.'
    try:
        if dir_name: makedirs(dir_name, exist_ok=True)
    except OSError:
        state_dir = path.join(path.expanduser('~'), '.local', 'state', 'llacie')
        try:
            makedirs(state_dir, exist_ok=True)
            target_dir = state_dir
            echo_warn(f"Cannot write to {dir_name or 'the current directory'}; "
                f"logging failures to {state_dir} instead")
        except OSError:
            target_dir = None

    response_note = ""
    if response is not None and target_dir is not None:
        responses_dir = path.join(target_dir, "failed_responses")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        which = "_".join(f"{k}{v}" for k, v in [("note", note_id), ("episode", episode_id)]
            if v is not None) or "unknown"
        response_file = path.join(responses_dir, f"{strategy_name}.{which}.{stamp}.txt")
        try:
            makedirs(responses_dir, exist_ok=True)
            with open(response_file, 'w') as f:
                f.write(str(response))
            response_note = f" (raw response: {response_file})"
        except OSError as e:
            response_note = f" (raw response could not be saved: {e})"

    echo_warn(f"Failed ({ids}): {reason}{response_note}")

    if target_dir is not None:
        row = [datetime.now().isoformat(timespec='seconds'), strategy_name,
            note_id, episode_id, note_feature_id, reason]
        try:
            _append_failure_row(path.join(target_dir, path.basename(failures_file)), row)
        except OSError as e:
            echo_warn(f"Could not record the failure in a file ({e})")

def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if isinstance(val, bool): return val
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))

def create_row_like(**kwargs):
    """Create a row-like object with positional and attribute access"""
    RowLike = namedtuple('RowLike', kwargs.keys())
    return RowLike(**kwargs)

def generate_unique_id(existing_ids, digits=10):
    """Generate a large random integer not in `existing_ids`. Each integer is `digits` long.
    Returns the new ID and also adds it to `existing_ids`."""
    while True:
        new_id = random.randint(10 ** digits, 10 ** (digits + 1) - 1)
        if new_id not in existing_ids:
            existing_ids.add(new_id)
            return new_id

def echo_strategies(strategies, title="Sections"):
    last_task_name = None
    echo(f"\n{(title + ':').ljust(24)} Available strategies:")
    echo("════════════════════════ ════════════════════════════════════════")
    for i, strat in enumerate(strategies):
        next_task_name = strategies[i + 1].task.name if i + 1 < len(strategies) else None
        if strat.task.name == last_task_name:
            sep = "  └─" if next_task_name != strat.task.name else "  ├─"
            task_formatted = (" " * 20)
        else: 
            sep = "◁───" if next_task_name != strat.task.name else "◁─┬─"
            task_formatted = strat.task.name.ljust(20)
        echo(f"    {task_formatted} {sep} {strat.name}")
        last_task_name = strat.task.name
    echo("")

def only_one_strategy(strategies, title="Sections"):
    if len(strategies) > 1:
        echo("Error: Your parameters matched several strategies. Here's what matched:")
        echo_strategies(strategies, title="Sections")
        echo("Use the --strategy-name option to select a single strategy.")
        return False
    elif len(strategies) == 0:
        echo("Error: No strategies matching those parameters were found.")
        return False
    return True
    