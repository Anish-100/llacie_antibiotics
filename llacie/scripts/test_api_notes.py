"""Smoke-test for the API compatibility layer (llacie/backend/api_compatible.py)
against the notes stored in the LLaCIE database.

For each of the first N notes with text, this sends the antibiotics extraction
prompt (from AntibioticsGPTOSS120BStrategy) to the UCI Health server through the
API compatibility layer, and verifies that each response is non-empty and parses
as a JSON list of antibiotic dicts.

Usage (from the repository root, with .env configured):
    python -m llacie.scripts.test_api_notes -n 3
    python -m llacie.scripts.test_api_notes -n 5 -o test_api_outputs.jsonl
    python -m llacie.scripts.test_api_notes -n 3 --dry-run   # prints prompts, no API calls
"""
import json
import re
import sys

import click
import pandas as pd
from openai import OpenAIError
from sqlalchemy import text

from ..app import App
from ..backend.api_compatible import UCI_HEALTH
from ..strategies.feature.antibiotics.gpt_oss_120b import AntibioticsGPTOSS120BStrategy
from ..utils import echo_info, echo_warn, echo_err

JSON_FENCE_PATTERN = re.compile(r'^```(?:json)?\s*|\s*```$')
JSON_FENCE_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*([\s\S]*?)```')
JSON_BRACKET_PATTERN = re.compile(r'(\[[\s\S]*?\]|\{[\s\S]*?\})')


def parse_llm_json(response):
    """Parses an LLM response as JSON, tolerating markdown code fences and
    explanatory prose before or after the JSON. Returns the parsed object,
    or None if no JSON can be parsed out of the response."""
    candidates = [response, JSON_FENCE_PATTERN.sub('', response.strip())]
    fence_match = JSON_FENCE_BLOCK_PATTERN.search(response)
    if fence_match is not None:
        candidates.append(fence_match.group(1))
    bracket_match = JSON_BRACKET_PATTERN.search(response)
    if bracket_match is not None:
        candidates.append(bracket_match.group(1))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


@click.command()
@click.option('-n', '--num-notes', default=3, show_default=True,
    help="Number of notes (with text) to test.")
@click.option('-o', '--output', type=click.Path(), default=None,
    help="Optional path to write the full results to, as JSON Lines.")
@click.option('--dry-run', is_flag=True,
    help="Print the prompts that would be sent and exit without calling the API.")
def main(num_notes, output, dry_run):
    """Smoke-test the API compatibility layer against imported notes."""
    with App() as app:
        notes_df = pd.read_sql_query(text(f'''
            SELECT "id", "NoteID", "note_text"
                FROM "{app.db.prefix}notes"
                WHERE "note_text" IS NOT NULL
                ORDER BY "id"
                LIMIT :n
            '''), app.db.conn, params={"n": num_notes})

        if len(notes_df) == 0:
            echo_err("Error: No notes with text found in the database. "
                "Run `llacie import-notes csv <path> --overwrite` first.")
            sys.exit(1)

        echo_info(f"Loaded {len(notes_df)} notes from the database")

        results = []
        if dry_run:
            for _, row in notes_df.iterrows():
                user_message = AntibioticsGPTOSS120BStrategy.LLM_USER_PROMPT.format(
                    input=row['note_text'])
                echo_info(f"note id {row['id']} (NoteID {row['NoteID']}): "
                    f"prompt would be {len(user_message)} chars")
            echo_info("Dry run, exiting without calling the API")
            return

        llm = UCI_HEALTH()
        llm.load_config()
        if not llm.config.get('API_KEY') or not llm.config.get('BASE_URL'):
            echo_err("Error: API_KEY and BASE_URL must be set in the .env file "
                "(see .env.example).")
            sys.exit(1)
        llm.SYSTEM_PROMPT = AntibioticsGPTOSS120BStrategy.LLM_SYSTEM_PROMPT
        model = llm.resolve_model()
        echo_info(f"Testing against model {model} at {llm.config.get('BASE_URL')}")

        num_passed = 0
        for _, row in notes_df.iterrows():
            record = {
                "note_id": int(row['id']),
                "NoteID": int(row['NoteID']),
                "model": model
            }
            user_message = AntibioticsGPTOSS120BStrategy.LLM_USER_PROMPT.format(
                input=row['note_text'])

            try:
                response, reasoning = llm.create_query(
                    user_message, **AntibioticsGPTOSS120BStrategy.SAMPLING_PARAMS)
            except OpenAIError as e:
                echo_err(f"note id {row['id']}: API call failed: {type(e).__name__}: {e}")
                record.update({"status": "api_error", "error": str(e)})
                results.append(record)
                continue

            record.update({"response": response, "reasoning": reasoning})
            if not response.strip():
                echo_warn(f"note id {row['id']}: empty response")
                record["status"] = "empty_response"
            else:
                parsed = parse_llm_json(response)
                if not isinstance(parsed, list):
                    echo_warn(f"note id {row['id']}: response did not parse as a JSON "
                        f"list: {response[:120]!r}")
                    record["status"] = "invalid_json"
                else:
                    drug_names = [d.get('Drug Name', '?') if isinstance(d, dict) else str(d)
                        for d in parsed]
                    echo_info(f"note id {row['id']}: OK, {len(parsed)} drugs: "
                        f"{', '.join(drug_names) if drug_names else '(none found in note)'}")
                    record.update({"status": "passed", "parsed": parsed})
                    num_passed += 1
            results.append(record)

        if output is not None:
            with open(output, 'w', encoding='utf-8') as f:
                for record in results:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            echo_info(f"Wrote full results to {output}")

        echo_info(f"{num_passed}/{len(results)} notes passed the smoke test")
        if num_passed < len(results):
            sys.exit(1)


if __name__ == '__main__':
    main()
