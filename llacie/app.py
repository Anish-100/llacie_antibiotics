import numpy as np
import pandas as pd
import math
import re
from click import echo

from .config import Config
from .db import LlacieDatabase
from .edw import EDWDatabase

from .strategies import find_strategies
from .tasks.abstract import TaskOutputType as out_t
from .evaluate import ConfusionMatrix
from .evaluate_antibiotics import evaluate_antibiotic_labels
from .text_wrangling import parse_sus_notes_csv
from .utils import echo_info, echo_warn, echo_err, echo_strategies, only_one_strategy

class App(object):
    def __init__(self):
        self.config = Config()
        self._db = None
        self._edw = None
        self.is_worker = False  

    @property
    def db(self):
        # Lazily initializes the connection to the database when needed
        if self._db is None:
            self._db = LlacieDatabase(self.config)
        return self._db
    
    @property
    def edw(self):
        # Lazily initializes the connection to the EDW database when needed
        if self._edw is None:
            self._edw = EDWDatabase(self.config)
        return self._edw

    @property
    def is_worker(self): 
        """Whether or not this is a worker instance of the App. Worker instances are created
        within bsub or slurm jobs by a master instance of the App."""
        return self._is_worker

    @is_worker.setter
    def is_worker(self, value):
        self._is_worker = bool(value)
        # Syncs with a variable in the config as well -- this way strategies can see it
        self.config['LLACIE_WORKER'] = self._is_worker


    # Allows this class to support the context manager protocol and be used
    # in `with` blocks to automatically clean up its own resources
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self._db is not None:
            self._db.close()
        if self._edw is not None:
            self._edw.close()
    

    def _rebuild_note_metadata(self, overwrite=False):
        self.db.warn_or_truncate('notes', overwrite)

        echo("Getting metadata for all episodes of interest.")
        eps_df = self.db.get_all_episodes()
        admit_pt_ec_ID_to_ep_ID = dict(
            zip(eps_df['admitPatientEncounterID'], eps_df['FK_episode_id'])
        )
        dc_pt_ec_ID_to_ep_ID = dict(
            zip(eps_df['dcPatientEncounterID'], eps_df['FK_episode_id'])
        )
        
        echo("Importing metadata for all admission, consult, and progress notes from "
            "ADMISSION encounters for each matching episode in EDW.")
        for df in self.edw.gen_admission_enc_note_metadata(eps_df['admitPatientEncounterID']):
            df['FK_episode_id'] = [
                admit_pt_ec_ID_to_ep_ID[ec_id] for ec_id in df['PatientEncounterID']
            ]
            self.db.append_df_to_table(df, 'notes')

        echo("Importing metadata for all discharge notes from DISCHARGE encounters "
            "for each matching episode in EDW.")
        for df in self.edw.gen_discharge_enc_note_metadata(eps_df['dcPatientEncounterID']):
            df['FK_episode_id'] = [
                dc_pt_ec_ID_to_ep_ID[ec_id] for ec_id in df['PatientEncounterID']
            ]
            self.db.append_df_to_table(df, 'notes')

    ##################################################################
    # High-level tasks that may be exposed as CLI commands
    ##################################################################

    def import_notes_text(self, input_path, overwrite=False):
        echo_info(f"Info: Importing notes from text file {input_path}.")
        self.db.warn_or_truncate('notes', overwrite)

        with open(input_path, 'r') as f:
            content = f.read()
        # Can handle two different types of notes. 
        #notes = re.split(r'\n#{10,}\n', content)
        notes = re.findall(r'NOTE_ID:.*?\n={10,}\n(.*?)(?=\n={10,}\n|\Z)', content, re.DOTALL)
        notes = [note.strip() for note in notes if note.strip()]
        echo_info(f"Found {len(notes)} notes to import.")

        if len(notes) == 0:
            echo_warn("Warn: No notes found in the file.")
            return

        self.db.import_notes(notes)
        echo_info(f"Info: Successfully imported {len(notes)} notes.")
    

    def import_notes_csv(self, input_path, overwrite=False):
        echo_info(f"Info: Importing notes from CSV file {input_path}.")
        if self._csv_is_sus_export(input_path):
            # Raw notes.sus export: repair its broken rows and strip the NoteID
            # prefix (LN09019955 -> 9019955) to get integer database NoteIDs
            records = parse_sus_notes_csv(input_path)
            if len(records) == 0:
                return echo_warn("Warn: No notes found in the file.")
            df = pd.DataFrame(records).rename(
                columns={"PatientEncounterID": "patientEncounterID"})
            echo_info(f"Info: Detected notes.sus export format ({len(df)} notes).")
        else:
            df = pd.read_csv(input_path, dtype={"patientEncounterID": str, "NoteID": str})
        missing = set(self.db.NOTES_CSV_COLUMNS) - set(df.columns)
        if missing:
            return echo_err(f"Error: Missing required column(s): {', '.join(sorted(missing))}")
        df = df.dropna(subset=["patientEncounterID", "NoteID", "note_text"])
        if len(df) == 0:
            return echo_warn("Warn: No notes found in the file.")
        self.db.warn_or_truncate('notes', overwrite)

        num_new, num_reused = self.db.import_notes_csv(df)
        echo_info(f"Info: Imported {len(df)} notes for {num_new + num_reused} encounters "
            f"({num_new} new episodes, {num_reused} existing).")

    @staticmethod
    def _csv_is_sus_export(input_path):
        """A raw notes.sus export is recognized by its letter-prefixed NoteIDs
        (e.g. LN09019955); cleaned CSVs use integer NoteIDs."""
        if 'NoteID' not in pd.read_csv(input_path, nrows=0).columns:
            return False
        note_ids = pd.read_csv(input_path, usecols=['NoteID'], dtype=str)['NoteID'].dropna()
        return bool(note_ids.str.match(r'^[A-Za-z]').any())


    def import_notes_edw(self, overwrite=False, resume_note_text=False):
        if not resume_note_text:
            self._rebuild_note_metadata(overwrite)

        echo_info("Info: Updating the notes table with full note text from EDW.")
        notes_df = self.db.get_notes_without_text()
        for df in self.edw.gen_note_text(notes_df['NoteID']):
            self.db.update_table_from_df(df, 'notes', 'NoteID')


    ##########################
    ####### SECTIONS #########
    ##########################

    def list_sections_with_strategies(self, section_name=None, strategy_name=None):
        strategies = find_strategies(output_type=out_t.SECTION, task=section_name,
            name_glob=strategy_name)
        if len(strategies) == 0:
            return echo_err("Error: No strategies matching those parameters were found.")
        
        echo_strategies(strategies, title="Sections")


    def run_section_strategy(self, section_name=None, strategy_name=None, dry_run=False):
        strategies = find_strategies(output_type=out_t.SECTION, task=section_name, 
            name_glob=strategy_name)
        if not only_one_strategy(strategies, "Sections"): return

        strat = strategies[0](self.db, self.config)
        ids = strat.get_unfinished_ids()
        echo_info(f"Info: {len(ids)} notes will undergo section extraction")
        if dry_run: echo_info("Info: Dry run, exiting")
        else: strat.run(ids)
    

    ##########################
    ####### FEATURES #########
    ##########################

    def list_features_with_strategies(self, feature_name=None, strategy_name=None):
        strategies = find_strategies(output_type=out_t.FEATURE, task=feature_name,
            name_glob=strategy_name)
        if len(strategies) == 0:
            return echo_err("Error: No strategies matching those parameters were found.")

        echo_strategies(strategies, title="Features")


    def run_feature_strategy(self, feature_name=None, strategy_name=None, dry_run=False,
            annotated_for_task=None, max_note_ids=None, redo_older_than=None, note_ids=tuple()):
        strategies = find_strategies(output_type=out_t.FEATURE, task=feature_name, 
            name_glob=strategy_name)
        if not only_one_strategy(strategies, "Features"): return

        strat = strategies[0](self.db, self.config)
        # FIXME: Should have strategies check whether prereqs are finished.

        note_ids = np.array(note_ids)
        if len(note_ids) == 0:
            note_ids = strat.get_unfinished_ids(newer_than=redo_older_than)

        if not self.is_worker:
            if annotated_for_task is not None:
                annot_strategies = find_strategies(task=annotated_for_task)
                if len(annot_strategies) == 0:
                    return echo_err(f"No task exists with that name: {annotated_for_task}")
                annot_task = annot_strategies[0].task
                annot_note_ids = self.db.get_ids_annotated_for_task(annot_task, strat.task)
                note_ids = np.intersect1d(note_ids, annot_note_ids)

            echo_info(f"Info: {len(note_ids)} notes will undergo feature extraction")

        strat.run(note_ids, max_note_ids=max_note_ids, dry_run=dry_run)


    ################################
    ####### EPISODE LABELS #########
    ################################

    def list_episode_labels_with_strategies(self, episode_label_task=None, strategy_name=None):
        strategies = find_strategies(output_type=out_t.EPISODE_LABEL, task=episode_label_task,
            name_glob=strategy_name)
        if len(strategies) == 0:
            return echo_err("Error: No strategies matching those parameters were found.")

        echo_strategies(strategies, title="Episode labels")


    def run_episode_label_strategy(self, episode_label_task=None, strategy_name=None, 
            dry_run=False, annotated_for_task=None):
        strategies = find_strategies(output_type=out_t.EPISODE_LABEL, task=episode_label_task, 
            name_glob=strategy_name)
        if not only_one_strategy(strategies, "Episode labels"): return

        strat = strategies[0](self.db, self.config)
        # FIXME: Should have strategies check whether prereqs are finished.
        ep_ids = strat.get_unfinished_ids()

        if annotated_for_task is not None:
            annot_strategies = find_strategies(task=annotated_for_task)
            if len(annot_strategies) == 0:
                return echo_err(f"No task exists with that name: {annotated_for_task}")
            annot_task = annot_strategies[0].task
            annot_ep_ids = self.db.get_ids_annotated_for_task(annot_task, strat.task)
            ep_ids = np.intersect1d(ep_ids, annot_ep_ids)

        echo_info(f"Info: {len(ep_ids)} episodes will undergo label creation")
        if dry_run: echo_info("Info: Dry run, exiting")
        else: strat.run(ep_ids)


    def _echo_confusion_matrices(self, df_human, df_pred, vocab, other_human=None, n_resamples=None):
        label = f"of {other_human}'s" if other_human is not None else "LLM"
        echo(f"=============== Using top 10 {label} responses ================")
        conf_mat = ConfusionMatrix.from_episode_labels(df_human, df_pred, vocab, 
            max_line_num=10, other_human=other_human, n_resamples=n_resamples)
        echo(conf_mat)
        echo(f"=============== Using all {label} responses ================")
        conf_mat = ConfusionMatrix.from_episode_labels(df_human, df_pred, vocab, 
            max_line_num=math.inf, other_human=other_human, n_resamples=n_resamples)
        echo(conf_mat)


    def _evaluate_antibiotic_labels(self, strategy, human_username, other_human=None,
            also_labeled_by=None, n_resamples=None, days=None):
        """Compares a strategy's (or `other_human`'s) antibiotic_labels against
        `human_username`'s, over the episodes that human labeled."""
        if human_username is None:
            return echo_err("Error: Evaluating antibiotics labels needs -u, the human "
                "annotator to compare against.")
        task = strategy.task
        if days is None:
            days = int(self.config.get("RELEVANCE_DAYS") or self.db.DEFAULT_RELEVANCE_DAYS)
        def labels(**source):
            return self.db.get_antibiotic_labels_with_relevance(task, days=days,
                include_empty=True, **source)

        df_human = labels(human_username=human_username)
        ep_ids = set(df_human["FK_episode_id"])
        if also_labeled_by is not None:
            ep_ids &= set(labels(human_username=also_labeled_by)["FK_episode_id"])
        if other_human is not None:
            df_pred = labels(human_username=other_human)
            # For inter-labeler reliability, only compare episodes both humans labeled
            ep_ids &= set(df_pred["FK_episode_id"])
            source = f"{other_human}'s labels"
        else:
            df_pred = labels(strategy=strategy)
            source = f"Strategy {strategy.name}"
        if len(ep_ids) == 0:
            return echo_warn(f"Warn: No episodes labeled by {human_username} to compare "
                f"for {task.name}, skipping")

        ep_ids = sorted(ep_ids)
        num_unlabeled = len(set(ep_ids) - set(df_pred["FK_episode_id"]))
        echo_info(f"{source} vs {human_username}: {len(ep_ids)} episodes, relevance window "
            f"{days} days" + (f" ({num_unlabeled} episodes have no labels from this "
            "strategy, counted as nothing found)" if num_unlabeled else ""))
        report, _ = evaluate_antibiotic_labels(df_human, df_pred, ep_ids, n_resamples)
        echo(report)


    def export_episode_labels(self, output, episode_label_task=None, strategy_name=None,
            human_username=None, days=None, relevant_only=False):
        if human_username is not None:
            strategies = find_strategies(output_type=out_t.EPISODE_LABEL,
                task=episode_label_task)
            if len(strategies) == 0:
                return echo_err("Error: No episode labelling task by that name was found.")
            task, strategy = strategies[0].task, None
        else:
            strategies = find_strategies(output_type=out_t.EPISODE_LABEL,
                task=episode_label_task, name_glob=strategy_name)
            if not only_one_strategy(strategies, "Episode labels"): return
            strategy = strategies[0](self.db, self.config)
            task = strategy.task
        if task.label_table != "antibiotic_labels":
            return echo_err(f"Error: export only supports antibiotics tasks for now, "
                f"not {task.name}")

        if days is None:
            days = int(self.config.get("RELEVANCE_DAYS") or self.db.DEFAULT_RELEVANCE_DAYS)
        df = self.db.get_antibiotic_labels_with_relevance(task, strategy, human_username,
            days)
        if relevant_only:
            df = df[df["relevance"] == "relevant"]
        df.to_csv(output, index=False)
        counts = df["relevance"].value_counts().to_dict()
        source = strategy.name if strategy is not None else f"annotator {human_username}"
        echo_info(f"Exported {len(df)} labels from {source} for "
            f"{df['FK_episode_id'].nunique()} episodes (relevance window {days} days): "
            + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))


    def evaluate_episode_label_strategies(self, episode_label_task=None, strategy_name=None,
            human_username=None, other_human=None, also_labeled_by=None, bootstrap_samples=None,
            days=None):
        strategies = find_strategies(output_type=out_t.EPISODE_LABEL, task=episode_label_task,
            name_glob=strategy_name)
        if len(strategies) == 0:
            return echo_err("Error: No strategies matching those parameters were found.")

        human_annotator = human_username if human_username is not None else True
        seen_tasks = set()

        for strategy_cls in strategies:
            strategy = strategy_cls(self.db, self.config)
            if strategy.task.label_table == "antibiotic_labels":
                if other_human is not None:
                    if strategy.task.name in seen_tasks: continue
                    seen_tasks.add(strategy.task.name)
                self._evaluate_antibiotic_labels(strategy, human_username, other_human,
                    also_labeled_by, bootstrap_samples, days)
                continue
            vocab = strategy.task.vocab

            ep_ids = None
            if also_labeled_by is not None:
                df_also_labeled = self.db.get_episode_labels(strategy, human_annotated=also_labeled_by)
                ep_ids = df_also_labeled["FK_episode_id"].unique()

            df_human = self.db.get_episode_labels(strategy, ep_ids, human_annotated=human_annotator)
            ep_ids = df_human["FK_episode_id"].unique()

            if other_human is not None:
                if strategy.task.name in seen_tasks: continue
                seen_tasks.add(strategy.task.name)
                echo_info(f"Task {strategy.task.name}, {len(ep_ids)} episodes, vocab size {len(vocab)}")

                df_pred = self.db.get_episode_labels(strategy, ep_ids, human_annotated=other_human)
                if len(df_pred) == 0:
                    echo_warn(f"Warn: No 2nd human annotations for {strategy.task.name}, skipping")
                    continue
                # When evaluating inter-labeler reliability, don't penalize for any episodes not
                #   labeled by the other human
                other_human_ep_ids = df_pred["FK_episode_id"].unique()
                df_human = df_human[df_human["FK_episode_id"].isin(other_human_ep_ids)]
            else:
                echo_info(f"Strategy {strategy.name}, {len(ep_ids)} episodes, vocab size {len(vocab)}")
                
                df_pred = self.db.get_episode_labels(strategy, ep_ids, human_annotated=False)
                if len(df_pred) == 0:
                    echo_warn(f"Warn: No episode labels created for {strategy.name}, skipping")
                    continue

            if len(df_human) == 0:
                echo_warn(f"Warn: No human annotations for {strategy.task.name}, skipping")
                continue
            self._echo_confusion_matrices(df_human, df_pred, vocab, other_human, bootstrap_samples)