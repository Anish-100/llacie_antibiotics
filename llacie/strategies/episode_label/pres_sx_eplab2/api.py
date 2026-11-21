from . import AbstractPresentingSymptomsEpisodeLabelV2Strategy
from ...feature.presenting_sx.api import PresentingSxFeatureApiStrategy
from ...custom_health_server import api_model_label
from ....tasks.episode_label import PresentingSymptomsEpisodeLabelV2Task

class PresentingSxEpisodeLabelApiStrategy(
        AbstractPresentingSymptomsEpisodeLabelV2Strategy):
    """\
    Converts the feature.presenting_sx.api.<model> note features into episode labels,
    for the model currently set in MODEL_NAME.
    """
    task = PresentingSymptomsEpisodeLabelV2Task
    name = f"episode_label.pres_sx_eplab2.api.{api_model_label()}"
    version = "0.0.1"
    prereq_tasks = []

    BATCH_SIZE = 1000


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        self.feat_strat = PresentingSxFeatureApiStrategy(db, config, **options)