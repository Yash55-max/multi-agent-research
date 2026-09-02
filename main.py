from typing import TypedDict, Annotated
import operator
import time
import re
from datetime import date

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"

MAX_REVISIONS = 2
MAX_SEARCH_RESULTS = 3

MAX_SEARCH_CONTEXT_CHARS = 4500
MAX_NOTE_CHARS = 1800
MAX_CRITIC_INPUT_CHARS = 5000
MAX_WRITER_INPUT_CHARS = 7000

REQUEST_DELAY = 2


# ============================================================
# SHARED STATE
# ============================================================

class ResearchState(TypedDict):
    """Shared state passed between all agents."""

    topic: str
    sub_questions: list[str]

    # operator.add allows LangGraph state accumulation.
    # We deduplicate before critic/writer processing.
    research_notes: Annotated[list[dict], operator.add]

    critique: str
    needs_more_research: bool
    final_report: str
    revision_count: int


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text: str, max_chars: int) -> str:
    """Normalize whitespace and safely limit text size."""

    text = re.sub(r"\s+", " ", str(text)).strip()

    if len(text) > max_chars:
        return text[:max_chars] + "..."

    return text


def deduplicate_notes(notes: list[dict]) -> list[dict]:
    """
    Keep the latest note for each question.

    This prevents repeated Critic -> Researcher cycles from
    continuously increasing the state and prompt size.
    """

    latest_notes = {}

    for note in notes:
        question = note.get("question", "").strip()

        if question:
            latest_notes[question] = note

    return list(latest_notes.values())


def extract_search_context(search_results) -> str:
    """
    Convert Tavily results into compact evidence.

    Only title, URL, and a limited content excerpt are passed
    to the LLM to prevent Groq request-size errors.
    """

    if not search_results:
        return "No search results were returned."

    if isinstance(search_results, dict):
        items = search_results.get("results", [])

    elif isinstance(search_results, list):
        items = search_results

    else:
        return "No usable search results were returned."

    evidence_blocks = []

    content_limit = max(
        500,
        MAX_SEARCH_CONTEXT_CHARS // max(len(items), 1)
    )

    for index, item in enumerate(
        items[:MAX_SEARCH_RESULTS],
        start=1
    ):

        if not isinstance(item, dict):
            continue

        title = clean_text(
            item.get("title", "Untitled"),
            200
        )

        url = clean_text(
            item.get("url", ""),
            500
        )

        content = (
            item.get("content")
            or item.get("snippet")
            or item.get("raw_content")
            or ""
        )

        content = clean_text(
            content,
            content_limit
        )

        evidence_blocks.append(
            f"""
SOURCE {index}
Title: {title}
URL: {url}
Content: {content}
""".strip()
        )

    if not evidence_blocks:
        return "No usable evidence was extracted."

    return "\n\n".join(evidence_blocks)


def get_notes_for_prompt(
    notes: list[dict],
    max_chars: int,
    finding_limit: int
) -> str:
    """
    Convert research notes into a bounded prompt.

    The total output is truncated to avoid Groq TPM/request limits.
    """

    blocks = []

    for note in notes:

        question = clean_text(
            note.get("question", ""),
            400
        )

        findings = clean_text(
            note.get("findings", ""),
            finding_limit
        )

        blocks.append(
            f"""
QUESTION:
{question}

FINDINGS:
{findings}
""".strip()
        )

    return clean_text(
        "\n\n".join(blocks),
        max_chars
    )


# ============================================================
# PLANNER AGENT
# ============================================================

planner_llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
)


class SubQuestions(BaseModel):
    """Structured output expected from the planner."""

    sub_questions: list[str] = Field(
        description=(
            "Exactly 5 focused, specific, non-overlapping "
            "and searchable research questions."
        )
    )


planner_structured = planner_llm.with_structured_output(
    SubQuestions
)


def planner_node(state: ResearchState) -> dict:
    """Break the topic into five research questions."""

    print(
        f"\n[Planner] Breaking down topic: "
        f"{state['topic']}"
    )

    prompt = f"""
Break the following research topic into exactly 5 focused,
specific, non-overlapping, searchable research questions.

The questions should collectively cover the most important
technical, regulatory, commercial, economic, and safety aspects.

Avoid questions that are too broad.

Topic:
{state["topic"]}

Return exactly 5 questions.
"""

    result = planner_structured.invoke(prompt)

    questions = result.sub_questions[:5]

    print(
        f"[Planner] Generated "
        f"{len(questions)} sub-questions:"
    )

    for index, question in enumerate(
        questions,
        start=1
    ):
        print(f"  {index}. {question}")

    return {
        "sub_questions": questions
    }


# ============================================================
# RESEARCHER AGENT
# ============================================================

search_tool = TavilySearch(
    max_results=MAX_SEARCH_RESULTS
)

researcher_llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
)


def researcher_node(state: ResearchState) -> dict:
    """
    Research each question.

    Tavily performs search.
    The LLM receives evidence as plain text and has no tools bound.
    """

    questions = state.get("sub_questions", [])

    print(
        f"\n[Researcher] Researching "
        f"{len(questions)} questions"
    )

    previous_notes = deduplicate_notes(
        state.get("research_notes", [])
    )

    previous_note_map = {
        note.get("question"): note.get("findings")
        for note in previous_notes
        if note.get("question")
    }

    critique = clean_text(
        state.get("critique", ""),
        900
    )

    notes = []

    for question in questions:

        print(
            f"\n[Researcher] Question: "
            f"{question}"
        )

        # ----------------------------------------------------
        # STEP 1: SEARCH
        # ----------------------------------------------------

        try:

            search_results = search_tool.invoke(
                {"query": question}
            )

            search_context = extract_search_context(
                search_results
            )

        except Exception as error:

            print(
                f"[Researcher] Search failed: {error}"
            )

            search_context = (
                "No search evidence was available because "
                "the search request failed."
            )

        # ----------------------------------------------------
        # STEP 2: PREVIOUS FINDINGS
        # ----------------------------------------------------

        previous_findings = previous_note_map.get(
            question,
            ""
        )

        # Only use previous findings when revising.
        # Keep them small to avoid growing prompts.

        previous_findings = clean_text(
            previous_findings,
            700
        )

        # ----------------------------------------------------
        # STEP 3: SYNTHESIS
        # ----------------------------------------------------

        synthesis_prompt = f"""
You are a research analyst.

Today's date: {date.today().isoformat()}

Research question:
{question}

Critic feedback:
{critique if critique else "No previous critique."}

Previous findings:
{previous_findings if previous_findings else "None."}

Search evidence:
{search_context}

Instructions:

- Use ONLY the search evidence provided above.
- Do NOT call tools.
- Do NOT invent facts, dates, numbers, companies, projects,
  approvals, or regulatory decisions.
- Do NOT use outside knowledge.
- If evidence is incomplete, explicitly state the limitation.
- Preserve complete URLs exactly.
- Only include a claim when supported by the evidence.
- Keep the response under 220 words.

Return exactly this format:

Summary:
<concise answer>

Evidence:
- Claim: <supported claim>
  Source: <full URL>

Limitations:
<missing evidence or uncertainty>
"""

        try:

            response = researcher_llm.invoke(
                synthesis_prompt
            )

            answer = clean_text(
                response.content,
                MAX_NOTE_CHARS
            )

        except Exception as error:

            print(
                f"[Researcher] Synthesis failed: "
                f"{error}"
            )

            answer = (
                "Summary: Research synthesis failed.\n\n"
                "Evidence:\n"
                "- No usable synthesized evidence was produced.\n\n"
                "Limitations:\n"
                "The evidence could not be synthesized during "
                "this iteration."
            )

        notes.append(
            {
                "question": question,
                "findings": answer,
            }
        )

        print(
            f"[Researcher] Completed: "
            f"{question[:65]}..."
        )

        time.sleep(REQUEST_DELAY)

    return {
        "research_notes": notes
    }


# ============================================================
# CRITIC AGENT
# ============================================================

critic_llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
)


class CriticReview(BaseModel):
    """Structured output expected from the critic."""

    needs_more_research: bool = Field(
        description=(
            "True only when major evidence gaps prevent "
            "writing a grounded concise report."
        )
    )

    critique: str = Field(
        description=(
            "Concise actionable critique under 120 words."
        )
    )


critic_structured = critic_llm.with_structured_output(
    CriticReview
)


def critic_node(state: ResearchState) -> dict:
    """
    Review unique research notes.

    Uses bounded input to prevent request-size and TPM errors.
    """

    unique_notes = deduplicate_notes(
        state.get("research_notes", [])
    )

    revision_count = state.get(
        "revision_count",
        0
    )

    print(
        f"\n[Critic] Reviewing "
        f"{len(unique_notes)} unique research notes"
    )

    # Hard stop before making another LLM request.
    if revision_count >= MAX_REVISIONS:

        print(
            "[Critic] Maximum revision limit reached."
        )

        return {
            "needs_more_research": False,
            "critique": (
                "Maximum revision limit reached. "
                "Write the final report using only supported "
                "evidence and clearly state limitations."
            ),
            "revision_count": revision_count,
            "research_notes": unique_notes,
        }

    notes_text = get_notes_for_prompt(
        unique_notes,
        MAX_CRITIC_INPUT_CHARS,
        750
    )

    review_prompt = f"""
You are a strict research quality reviewer.

Topic:
{state["topic"]}

Research notes:
{notes_text}

Evaluate only:

1. Are the five major questions substantially addressed?
2. Are important factual claims tied to evidence?
3. Are usable source URLs present?
4. Are important uncertainties stated?
5. Are there obvious contradictions?

Approve the research if it is sufficient for a concise,
grounded report.

Do NOT demand exhaustive statistics or perfect coverage.

Return needs_more_research=true ONLY if major evidence gaps
prevent responsible reporting.

Keep critique under 120 words.
"""

    try:

        review = critic_structured.invoke(
            review_prompt
        )

        needs_more_research = (
            review.needs_more_research
        )

        critique = clean_text(
            review.critique,
            1000
        )

    except Exception as error:

        print(
            f"[Critic] Review failed: {error}"
        )

        needs_more_research = False

        critique = (
            "Critic review failed. Proceed using only "
            "the available evidence."
        )

    new_revision_count = revision_count

    if needs_more_research:

        new_revision_count += 1

        print(
            "[Critic] Additional research required."
        )

    else:

        print(
            "[Critic] Research approved."
        )

    print(
        f"[Critic] Feedback: {critique}"
    )

    return {
        "needs_more_research": needs_more_research,
        "critique": critique,
        "revision_count": new_revision_count,
        "research_notes": unique_notes,
    }


# ============================================================
# WRITER AGENT
# ============================================================

writer_llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.2,
)


def writer_node(state: ResearchState) -> dict:
    """
    Generate the final report using only research findings.
    """

    print(
        "\n[Writer] Generating final research report"
    )

    unique_notes = deduplicate_notes(
        state.get("research_notes", [])
    )

    notes_text = get_notes_for_prompt(
        unique_notes,
        MAX_WRITER_INPUT_CHARS,
        1100
    )

    prompt = f"""
You are a professional research report writer.

Topic:
{state["topic"]}

Research findings:
{notes_text}

Rules:

- Use ONLY information contained in the findings.
- Do NOT add outside knowledge.
- Do NOT invent facts, dates, statistics, companies,
  regulatory approvals, or sources.
- Preserve uncertainty.
- If evidence is incomplete, explicitly state the limitation.
- Do not create fake citations such as [1] or [2].
- Only list URLs that appear in the findings.
- Remove duplicate URLs.
- Keep the report concise and professional.

Use exactly this structure:

# Introduction

# Key Findings

## Technology and Development

## Deployment and Regulation

## Economics and Commercialization

## Safety and Environmental Considerations

# Limitations and Uncertainty

# Conclusion

# Sources

For Sources:
- Include only complete URLs beginning with http:// or https://
- One URL per line
- Remove duplicates
- Do not invent URLs
"""

    try:

        result = writer_llm.invoke(prompt)

        report = result.content.strip()

        if not report:
            raise ValueError(
                "Writer returned an empty response."
            )

        print(
            "[Writer] Report generated successfully"
        )

    except Exception as error:

        print(
            f"[Writer] Report generation failed: "
            f"{error}"
        )

        report = (
            "# Report Generation Failed\n\n"
            "The final report could not be generated."
        )

    return {
        "final_report": report
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_critic(
    state: ResearchState
) -> str:
    """
    Route:

    Critic -> Researcher when more evidence is required.
    Critic -> Writer otherwise.
    """

    needs_more_research = state.get(
        "needs_more_research",
        False
    )

    revision_count = state.get(
        "revision_count",
        0
    )

    print(
        f"\n[Router] "
        f"needs_more_research={needs_more_research}, "
        f"revision_count={revision_count}"
    )

    if (
        needs_more_research
        and revision_count < MAX_REVISIONS
    ):

        print(
            "[Router] Sending task back to Researcher"
        )

        return "researcher"

    print(
        "[Router] Sending task to Writer"
    )

    return "writer"


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        ResearchState
    )

    graph.add_node(
        "planner",
        planner_node
    )

    graph.add_node(
        "researcher",
        researcher_node
    )

    graph.add_node(
        "critic",
        critic_node
    )

    graph.add_node(
        "writer",
        writer_node
    )

    graph.set_entry_point(
        "planner"
    )

    graph.add_edge(
        "planner",
        "researcher"
    )

    graph.add_edge(
        "researcher",
        "critic"
    )

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "researcher": "researcher",
            "writer": "writer",
        }
    )

    graph.add_edge(
        "writer",
        END
    )

    return graph.compile()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app = build_graph()

    print(
        "\nGRAPH STRUCTURE:\n"
    )

    print(
        app.get_graph().draw_mermaid()
    )

    initial_state = {
        "topic": (
            "the current state of "
            "small modular nuclear reactors"
        ),
        "sub_questions": [],
        "research_notes": [],
        "critique": "",
        "needs_more_research": False,
        "revision_count": 0,
        "final_report": "",
    }

    result = app.invoke(
        initial_state
    )

    print("\n")
    print("=" * 70)
    print("FINAL RESEARCH REPORT")
    print("=" * 70)
    print("\n")

    print(
        result.get(
            "final_report",
            "No report was generated."
        )
    )