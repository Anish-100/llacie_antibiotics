import pandas as pd

from sqlalchemy import text

from .tasks import TaskOutputType as out_t
from .utils import echo_info, echo_warn


class AntibioticLabelsMixin:
    """LlacieDatabase methods for the antibiotic_labels table (antibiotics episode labels).

    Kept in their own module; they rely on LlacieDatabase for self.conn, self.prefix,
    self.ep_table, get_or_register_task() and append_df_to_table()."""

    ANTIBIOTIC_LABEL_COLUMNS = {
        # spreadsheet column -> key in the feature JSON that to_label_row() reads
        "drug_name": "Drug Name",
        "start_date": "Start date",
        "end_date": "End date",
        "duration_days": "Duration",
        "current": "Current",
    }

    def import_antibiotic_labels(self, strategy, input_file, sheet_name=0,
            human_username=None):
        """Imports human antibiotics labels into antibiotic_labels, one row per drug course.

        Expects a CSV or XLSX with a header row containing `FK_episode_id` (or
        `patientEncounterID`, looked up in the episodes table) and `drug_name`, and optionally
        `start_date`, `end_date` (MM/DD/YYYY), `duration_days` (or `duration`), `current`
        (true/false) and `admission_date` (checked against the episode's startDTS). A row with an empty `drug_name` means no antibiotics for that episode.
        Re-importing replaces this annotator's earlier labels for the same episodes."""
        from .strategies.episode_label.antibiotics import to_label_row, empty_label_row

        file_name = getattr(input_file, "name", str(input_file))
        if file_name.lower().endswith(".csv"):
            df = pd.read_csv(input_file)
        else:
            df = pd.read_excel(input_file, sheet_name=sheet_name)
        df = df.rename(columns={"duration": "duration_days"})
        if "FK_episode_id" not in df.columns and "patientEncounterID" in df.columns:
            df = self._add_episode_ids_from_encounters(df)
        missing = {"FK_episode_id", "drug_name"} - set(df.columns)
        if missing:
            raise RuntimeError(f"Missing required column(s): {', '.join(sorted(missing))} "
                "(FK_episode_id can instead come from patientEncounterID)")
        df = df.dropna(subset=["FK_episode_id"])

        def cell(value):
            if pd.isna(value): return None
            if isinstance(value, float) and value.is_integer(): return int(value)
            return value

        rows_by_episode = {}
        for _, row in df.iterrows():
            item = {key: cell(row[col]) for col, key in self.ANTIBIOTIC_LABEL_COLUMNS.items()
                if col in df.columns}
            label_row = to_label_row(item) if item.get("Drug Name") else empty_label_row()
            rows_by_episode.setdefault(int(row["FK_episode_id"]), []).append(label_row)
        # An episode with any drug doesn't also need the "none found" row
        for ep_id, label_rows in rows_by_episode.items():
            drugs = [r for r in label_rows if r["drug_name_raw"] is not None]
            rows_by_episode[ep_id] = drugs if drugs else [empty_label_row()]

        params = {
            "ep_ids": list(rows_by_episode),
            "task_id": strategy.task_id,
            "human_username": human_username
        }
        self.conn.execute(text(f"""
            DELETE FROM "{self.prefix}antibiotic_labels"
                WHERE "FK_episode_id" = ANY(:ep_ids)
                    AND "FK_task_id" = :task_id
                    AND "FK_strategy_id" IS NULL
                    AND "FK_human_annotator" = :human_username
            """), params)
        long_df = pd.DataFrame([
            {"FK_episode_id": ep_id, "FK_task_id": strategy.task_id,
                "FK_human_annotator": human_username, **label_row}
            for ep_id, label_rows in rows_by_episode.items() for label_row in label_rows])
        self.append_df_to_table(long_df, 'antibiotic_labels')  # This also .commit()'s changes

        unknown = sorted({r for r in long_df["drug_name_raw"].dropna()
            if long_df.loc[long_df["drug_name_raw"] == r, "drug_name"].isna().all()})
        if unknown:
            echo_warn(f"Not in drugs.db, imported with an empty drug_name: {', '.join(unknown)}")
        num_drugs = int(long_df["drug_name_raw"].notna().sum())
        echo_info(f"{num_drugs} antibiotic labels imported for {len(rows_by_episode)} episodes.")

    def _add_episode_ids_from_encounters(self, df):
        """Adds FK_episode_id by looking up each patientEncounterID's episode (its admitting
        encounter). Rows with an unknown encounter are dropped with a warning. If there is an
        admission_date column, warns where it differs from the episode's startDTS date."""
        encounter_ids = pd.to_numeric(df["patientEncounterID"], errors="coerce")
        episodes = pd.read_sql_query(text(f"""
            SELECT "id" AS "FK_episode_id", "admitPatientEncounterID", "startDTS"
                FROM "{self.ep_table}"
                WHERE "admitPatientEncounterID" = ANY(:encounter_ids)
            """), self.conn,
            params={"encounter_ids": [int(e) for e in encounter_ids.dropna().unique()]})
        df = df.assign(admitPatientEncounterID=encounter_ids).merge(episodes, how="left",
            on="admitPatientEncounterID")

        unknown = df.loc[df["FK_episode_id"].isna(), "patientEncounterID"].unique()
        if len(unknown):
            echo_warn(f"No episode for patientEncounterID(s), skipped: "
                f"{', '.join(str(e) for e in unknown)}")
        if "admission_date" in df.columns:
            admitted = pd.to_datetime(df["admission_date"], errors="coerce").dt.date
            started = pd.to_datetime(df["startDTS_y" if "startDTS_y" in df else "startDTS"],
                utc=True).dt.date
            differs = df.loc[admitted.notna() & started.notna() & (admitted != started),
                "patientEncounterID"].unique()
            if len(differs):
                echo_warn(f"admission_date differs from the episode's startDTS for "
                    f"patientEncounterID(s): {', '.join(str(e) for e in differs)}")
        return df

    DEFAULT_RELEVANCE_DAYS = 30

    def get_antibiotic_labels_with_relevance(self, task, strategy=None, human_username=None,
            days=DEFAULT_RELEVANCE_DAYS, include_empty=False):
        """Returns antibiotic_labels rows for one strategy (or one human annotator), skipping
        "none found" rows unless `include_empty`, with a `relevance` column:

        - `start_date` on or after the admission date (started during admission) ->
          not_relevant, whatever else is known
        - `is_current` true -> relevant; false -> not_relevant
        - otherwise, by days from `end_date` to the episode's admission (`startDTS`):
          at most `days` (or ending after admission) -> relevant, more -> not_relevant
        - no `is_current` and no `end_date` (or no `startDTS`) -> not_relevant"""
        if (strategy is None) == (human_username is None):
            raise RuntimeError("Give exactly one of a strategy or a human annotator")
        params = {
            "task_id": int(self.get_or_register_task(task).id),
            "days": int(days),
            "strategy_id": strategy.id if strategy is not None else None,
            "human_username": human_username
        }
        source_sql = ('al."FK_strategy_id" = :strategy_id AND al."FK_human_annotator" IS NULL'
            if strategy is not None else
            'al."FK_strategy_id" IS NULL AND al."FK_human_annotator" = :human_username')
        include_empty_sql = "TRUE" if include_empty else "FALSE"
        select_sql = text(f"""
            SELECT al."id", al."FK_episode_id", e."startDTS", al."FK_strategy_id",
                    al."FK_human_annotator", al."FK_note_feature_id",
                    al."drug_name", al."drug_name_raw", al."start_date", al."end_date",
                    al."duration_days", al."is_current",
                    (e."startDTS"::date - al."end_date") AS "days_ended_before_admission",
                    CASE
                        -- Started during this admission (on or after the admission date):
                        -- not a pre-admission antibiotic, so never relevant to our case
                        WHEN al."start_date" >= e."startDTS"::date THEN 'not_relevant'
                        WHEN al."is_current" IS TRUE THEN 'relevant'
                        WHEN al."is_current" IS NULL
                            AND e."startDTS"::date - al."end_date" <= :days THEN 'relevant'
                        ELSE 'not_relevant'
                    END AS "relevance"
                FROM "{self.prefix}{task.label_table}" AS al
                JOIN "{self.ep_table}" AS e ON e."id" = al."FK_episode_id"
                WHERE al."FK_task_id" = :task_id
                    AND ({include_empty_sql} OR al."drug_name_raw" IS NOT NULL)
                    AND {source_sql}
                ORDER BY al."FK_episode_id", al."id"
            """)
        df = pd.read_sql_query(select_sql, self.conn, params=params)
        # Keep whole-number columns as integers even when some values are missing
        int_cols = ["FK_strategy_id", "FK_note_feature_id", "duration_days",
            "days_ended_before_admission"]
        return df.astype({col: "Int64" for col in int_cols})

    def replace_antibiotic_labels(self, ep_label_strategy, row_values, label_rows):
        """Replaces this strategy's antibiotic_labels rows for one episode with `label_rows`,
        a list of dicts with the drug columns (drug_name, drug_name_raw, start_date, end_date,
        duration_days, is_current). Human-annotated rows are left alone."""
        if ep_label_strategy.task.output_type != out_t.EPISODE_LABEL:
            raise RuntimeError("Only strategies for creating episode labels are allowed.")

        params = {
            "episode_id": int(row_values.episode_id),
            "strategy_id": ep_label_strategy.id,
            "note_feature_id": int(row_values.note_feature_id),
            "task_id": ep_label_strategy.task_id
        }
        delete_labels_sql = text(f"""
            DELETE FROM "{self.prefix}antibiotic_labels"
                WHERE "FK_episode_id" = :episode_id
                    AND "FK_strategy_id" = :strategy_id
                    AND "FK_human_annotator" IS NULL
            """)
        self.conn.execute(delete_labels_sql, params)

        insert_label_sql = text(f"""
            INSERT INTO "{self.prefix}antibiotic_labels" ("FK_episode_id",
                "FK_note_feature_id", "FK_strategy_id", "FK_task_id", "FK_human_annotator",
                "drug_name", "drug_name_raw", "start_date", "end_date", "duration_days",
                "is_current")
            VALUES (:episode_id, :note_feature_id, :strategy_id, :task_id, NULL,
                :drug_name, :drug_name_raw, :start_date, :end_date, :duration_days,
                :is_current)
            """)
        for label_row in label_rows:
            self.conn.execute(insert_label_sql, {**params, **label_row})

        self.conn.commit()
