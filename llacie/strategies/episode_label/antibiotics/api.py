from . import AbstractAntibioticsStrategy
from ...feature.antibiotics.api import AntibioticsApiStrategy
from ...custom_health_server import api_model_label
from ....tasks.episode_label import AntibioticsEpisodeLabelTask

class AntibioticsEpisodeLabelApiStrategy(
        AbstractAntibioticsStrategy):
    """\
    Converts the feature.antibiotics.api.<model> note features into antibiotic_labels rows,
    for the model currently set in MODEL_NAME.
    """
    task = AntibioticsEpisodeLabelTask
    name = f"episode_label.antibiotics_eplab1.api.{api_model_label()}"
    version = "0.0.4"
    prereq_tasks = []

    BATCH_SIZE = 1000


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        self.feat_strat = AntibioticsApiStrategy(db, config, **options)

        