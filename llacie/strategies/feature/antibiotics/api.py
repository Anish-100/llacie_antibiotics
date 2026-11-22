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
    version = "0.0.2"
    prereq_tasks = []

    SECTION_STRATEGY_CLASS = AntibioticsNoStrategy


    # Model context length. Note that Vllm will not generate past the end of the context window.
    LLM_MAX_MODEL_LENGTH = 120000
    LLM_SYSTEM_PROMPT = dedent("""\
        You are a clinical researcher that reads medical charts and answers questions about them. 
        Use only the information in the text provided to answer the question.
        If a patient denies something, do not include it in your answer.
        After you provide an answer, you immediately stop talking.
        Remember, do not infer what is given beyond the notes, only provide info that is within the note itself.
        Name each drug by its generic name, even when the note uses a brand name or an abbreviation.
        """)
    LLM_USER_PROMPT = dedent("""\
Read the following patient history and list the antibiotics (antibacterial medications)
the patient was taking before coming to the hospital.

Include only antibiotics prescribed or taken prior to this hospital admission, such
as those started by a primary care provider, urgent care, or during a recent prior
hospitalization that has since ended.

Do NOT extract:
- Antibiotics started for the first time during this admission, including those given in
  the emergency department, at admission, or recommended or planned in the assessment
- Antifungals (e.g., fluconazole, micafungin), antivirals (e.g., acyclovir, oseltamivir,
  HIV or hepatitis C medications) and antiparasitics

Return a JSON list of dicts. Each dict must follow this format:
(
  "Drug Name": str,   // Generic drug name in lowercase. Never a brand name (Keflex ->
                      // cephalexin, Levaquin -> levofloxacin) or an abbreviation (CTX,
                      // TMP-SMX); write the regimen's drugs as separate dicts (RIPE ->
                      // rifampin, isoniazid, pyrazinamide, ethambutol). For combinations,
                      // join with underscore (amoxicillin_clavulanate,
                      // piperacillin_tazobactam, trimethoprim_sulfamethoxazole).
                      // Strip salts and modifiers: DS, XR, ER, SR, CR, XL, LA, HCl,
                      // monohydrate, macrocrystals.
                      // Vancomycin only: write vancomycin_po or vancomycin_iv for its
                      // route, or vancomycin if the note does not say.
  "Route": str,       // e.g., PO, IV, IM, inhaled
  "Dose": str,        // e.g., 500mg, 2g — leave empty if unknown
  "Frequency": str,   // e.g., BID, daily, q8h — leave empty if unknown
  "Duration": int,    // Duration in days — leave empty if unknown
  "Start date": str,  // MM/DD/YYYY — leave empty if unknown
  "End date": str,    // MM/DD/YYYY — leave empty if unknown
  "Current": bool     // true if the patient was still taking it when admitted, false if
                      // it was stopped before admission — leave empty if the note does not say
)

List each course of a drug once. If a value is unknown, leave it empty. Do not guess.
Return nothing except the above json.
        {input}""")
