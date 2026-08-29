from typing import TypedDict, Annotated
import operator
import time
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
# SHARED STATE
# ============================================================

class ResearchState(TypedDict):
    """
    Shared state passed between all agents in the graph.
    """

    topic: str
    sub_questions: list[str]

    research_notes: Annotated[
        list[dict],
        operator.add
    ]

    critique: str
    needs_more_research: bool
    final_report: str
    revision_count: int


# ============================================================
# PLANNER AGENT
# ============================================================

planner_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


class SubQuestions(BaseModel):
    """
    Structured output expected from the Planner.
    """

    sub_questions: list[str] = Field(
        description=(
            "3-5 focused, specific and searchable "
            "research sub-questions that together "
            "cover the topic."
        )
    )


planner_structured = planner_llm.with_structured_output(
    SubQuestions
)


def planner_node(state: ResearchState) -> dict:
    """
    Uses Groq to break the research topic into
    focused sub-questions.
    """

    print(
        f"\n[Planner] Breaking down topic: "
        f"{state['topic']}"
    )

    result = planner_structured.invoke(
        f"""
Break down this research topic into 3-5 specific,
non-overlapping, searchable research sub-questions.

The questions should collectively cover the important
aspects of the topic.

Research topic:
{state["topic"]}
"""
    )

    print(
        f"[Planner] Generated "
        f"{len(result.sub_questions)} sub-questions:"
    )

    for index, question in enumerate(
        result.sub_questions,
        start=1
    ):
        print(f"  {index}. {question}")

    return {
        "sub_questions": result.sub_questions
    }


# ============================================================
# RESEARCHER AGENT
# ============================================================

search_tool = TavilySearch(
    max_results=2
)


researcher_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


def researcher_node(state: ResearchState) -> dict:
    """
    Researches each Planner-generated question.

    Flow:

        Question
            ↓
        Tavily Search
            ↓
        Groq synthesis
            ↓
        Research Note
    """

    print(
        f"\n[Researcher] Researching "
        f"{len(state['sub_questions'])} questions"
    )

    notes = []

    for question in state["sub_questions"]:

        print(
            f"\n[Researcher] Question: "
            f"{question}"
        )

        # ----------------------------------------------------
        # STEP 1: Search the web
        # ----------------------------------------------------

        search_results = search_tool.invoke(
            {
                "query": question
            }
        )

        # ----------------------------------------------------
        # STEP 2: Ask Groq to synthesize the results
        # ----------------------------------------------------

        synthesis_prompt = f"""
You are a research analyst.

Today's date is {date.today().isoformat()}.

Research question:
{question}

Web search results:
{search_results}

Instructions:
- Answer the research question using only the
  information contained in the search results.
- Do not invent facts.
- Prefer recent and authoritative information.
- Clearly summarize the most important evidence.
- Mention uncertainty where appropriate.
- Include the source URLs.
- Keep the answer under 300 words.
"""

        response = researcher_llm.invoke(
            synthesis_prompt
        )

        answer = response.content

        # ----------------------------------------------------
        # STEP 3: Store the result
        # ----------------------------------------------------

        notes.append(
            {
                "question": question,
                "findings": answer,
            }
        )

        print(
            f"[Researcher] Completed: "
            f"{question[:60]}..."
        )

        # Prevent excessive request frequency
        time.sleep(2)

    return {
        "research_notes": notes
    }


# ============================================================
# CRITIC AGENT
# ============================================================

def critic_node(state: ResearchState) -> dict:
    """
    Day 3 stub.

    The real Critic will be implemented on Day 4.
    """

    print(
        f"\n[Critic] Reviewing "
        f"{len(state['research_notes'])} research notes"
    )

    print(
        "[Critic] Day 3 stub: research approved."
    )

    return {
        "needs_more_research": False,
        "critique": "",
        "revision_count": state["revision_count"],
    }


# ============================================================
# WRITER AGENT
# ============================================================

def writer_node(state: ResearchState) -> dict:
    """
    Day 3 Writer stub.

    The real Writer will be implemented later.
    """

    print("\n[Writer] Writing final report")

    report = f"""
Research Report
===============

Topic:
{state["topic"]}

Research Notes:
"""

    for note in state["research_notes"]:

        report += f"""

Question:
{note["question"]}

Findings:
{note["findings"]}
"""

    return {
        "final_report": report
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_critic(state: ResearchState) -> str:
    """
    Routes to Researcher if more research is needed.
    Otherwise sends the workflow to Writer.
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
        and revision_count < 3
    ):
        print(
            "[Router] Sending task back "
            "to Researcher"
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

    graph = StateGraph(ResearchState)

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

    graph.set_entry_point("planner")

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

    print("\nGRAPH STRUCTURE:\n")
    print(app.get_graph().draw_mermaid())

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

    result = app.invoke(initial_state)

    print("\n")
    print("=" * 70)
    print("DAY 3 RESEARCH RESULTS")
    print("=" * 70)

    for note in result["research_notes"]:

        print(
            f"\nQ: {note['question']}"
        )

        print(note["findings"])