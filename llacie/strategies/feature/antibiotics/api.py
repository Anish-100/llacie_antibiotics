from textwrap import dedent

from ...custom_health_server import AbstractUciHealthStrategy, api_model_label
from ...section.medicines.all_sections import AntibioticsNoStrategy
from ....tasks.feature import AntibioticsFeatureTask


class AntibioticsApiStrategy(AbstractUciHealthStrategy):
    """\
    Extracts the `antibiotics` feature with whichever model the API server is configured for
    (MODEL_NAME in .env). The model is part of the strategy name, so each model's results are
    stored separately.
    """
    task = AntibioticsFeatureTask
    name = f"feature.antibiotics.api.{api_model_label()}"
    version = "0.0.1"
    prereq_tasks = []

    SECTION_STRATEGY_CLASS = AntibioticsNoStrategy


    # Model context length. Note that Vllm will not generate past the end of the context window.
    LLM_MAX_MODEL_LENGTH = 120000
    LLM_SYSTEM_PROMPT = dedent("""\
        You are a clinical researcher that reads medical charts and answers questions about them. 
        Use only the information in the text provided to answer the question.
        If a patient denies something, do not include it in your answer.
        After you provide an answer, you immediately stop talking.
        Remember, do not infer what is given beyong the notes, only provide info that is within the note itself. 
        Use the drug name mentioned in the note, nothing else. 
        """)
    LLM_USER_PROMPT = dedent("""\
Read the following patient history and list the antimicrobial (antibacterial, antiviral, 
antifungal) medications the patient was taking before coming to the hospital.

Include only antimicrobials prescribed or taken prior to this hospital admission, such 
as those started by a primary care provider, urgent care, or during a recent prior 
hospitalization that has since ended.

Do not include antimicrobials started for the first time during this admission.

Exclusions — do NOT extract:
- Antiretroviral therapy (ART) for HIV (e.g., bictegravir, emtricitabine, tenofovir, 
  efavirenz, or any fixed-dose combination ART tablet)
- Hepatitis C direct-acting antivirals (e.g., sofosbuvir, velpatasvir, ledipasvir)

Return a JSON list of dicts. Each dict must follow this format:
(
  "Drug Name": str,   // Base drug name. For combinations, join with underscore 
                      // (e.g., amoxicillin_clavulanate). Strip modifiers: DS, XR, 
                      // ER, SR, CR, XL, LA, HCl.
  "Route": str,       // e.g., PO, IV, IM, inhaled
  "Dose": str,        // e.g., 500mg, 2g — leave empty if unknown
  "Frequency": str,   // e.g., BID, daily, q8h — leave empty if unknown
  "Duration": int,    // Duration in days — leave empty if unknown
  "Start date": str,  // MM/DD/YYYY — leave empty if unknown
  "End date": str     // MM/DD/YYYY — leave empty if unknown
)

If a value is unknown, leave it empty. Do not guess.
        {input}""")
