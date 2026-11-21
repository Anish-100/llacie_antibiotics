from textwrap import dedent

from ...custom_health_server import AbstractUciHealthStrategy, api_model_label
from ...section.hpi_short.regex_temp import ShortHPISectionRegexStrategy
from ....tasks.feature import PresentingSymptomsFeatureTask

class PresentingSxFeatureApiStrategy(AbstractUciHealthStrategy):
    """\
    Extracts the `presenting_sx` feature with whichever model the API server is configured for
    (MODEL_NAME in .env). The model is part of the strategy name, so each model's results are
    stored separately.

    Also uses the `hpi_short` sections created by the section.hpi_short.regex strategy.
    """
    task = PresentingSymptomsFeatureTask
    name = f"feature.presenting_sx.api.{api_model_label()}"
    version = "0.0.1"
    prereq_tasks = []

    SECTION_STRATEGY_CLASS = ShortHPISectionRegexStrategy

    # SAMPLING_PARAMS = {
    #     "temperature": 0.1,
    #     "top_p": 0.8,
    #     "repetition_penalty": 1.05
    # }

    LLM_SYSTEM_PROMPT = dedent("""\
        You are a clinical researcher that reads medical charts and answers questions about them. 
        Use only the information in the text provided to answer the question.
        If a patient denies something, do not include it in your answer.
        After you provide an answer, you immediately stop talking.""")
    
    LLM_USER_PROMPT = dedent("""\
        Read the following patient history and list the patient's presenting symptoms.
        Include only symptoms present now or reported for the days to weeks leading up to admission.
        Ignore any symptoms from past medical history or prior hospital admissions.
        Give your answer as a JSON array containing up to ten strings.
        Each string contains between one and three words.
        
        {input}""")
