"""
Unit & Integration Tests for Agent Intent Classification, Guardrails, and Router.
Verifies that question inquiries never trigger code execution or unwanted charts.
"""

import pytest
from backend.agents.workflow import (
    understand_query_node,
    build_workflow,
    generate_suggestions_node,
    is_valid_suggested_question,
    generate_grounded_questions,
)


def test_screenshot_meta_question_query():
    """
    Direct test for the user screenshot bug:
    'what type of questions that i can ask on this dataset (plotting questions)?'
    Must be classified as META_QUESTIONS with needs_visualization=False.
    """
    state = {
        "query": "what type of questions that i can ask on this dataset (plotting questions)?",
        "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
        "numeric_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "categorical_columns": ["species"],
    }
    result = understand_query_node(state)
    assert result["intent"] == "META_QUESTIONS"
    assert result["needs_visualization"] is False
    assert result["off_topic"] is False
    assert result["needs_clarification"] is False


def test_general_question_ideas_query():
    state = {
        "query": "suggest some questions i can ask about this dataset",
        "columns": ["age", "fare", "survived"],
    }
    result = understand_query_node(state)
    assert result["intent"] == "META_QUESTIONS"
    assert result["needs_visualization"] is False


def test_conversational_followup_guardrail():
    state = {
        "query": "what does this graph indicate?",
        "columns": ["age", "fare"],
    }
    result = understand_query_node(state)
    assert result["intent"] == "EXPLANATION"
    assert result["needs_visualization"] is False


def test_underspecified_facet_guardrail():
    state = {
        "query": "create a facetgrid plot",
        "columns": ["age", "fare", "class"],
    }
    result = understand_query_node(state)
    assert result["needs_clarification"] is True
    assert "facet grid" in result["clarification_message"].lower()


def test_data_analysis_mode_rejects_what_is_python():
    """
    Direct test for user screenshot query 1:
    'what is python?' in Data Analysis mode must be flagged as OFF_TOPIC.
    """
    state = {
        "query": "what is python?",
        "chat_mode": "analysis",
        "columns": ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Fare"],
    }
    result = understand_query_node(state)
    assert result["intent"] == "OFF_TOPIC"
    assert result["off_topic"] is True
    assert result["needs_visualization"] is False


def test_data_analysis_mode_rejects_consecutive_ones():
    """
    Direct test for user screenshot query 2:
    'generate a code on consective ones' in Data Analysis mode must be flagged as OFF_TOPIC.
    """
    state = {
        "query": "generate a code on consective ones",
        "chat_mode": "analysis",
        "columns": ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Fare"],
    }
    result = understand_query_node(state)
    assert result["intent"] == "OFF_TOPIC"
    assert result["off_topic"] is True
    assert result["needs_visualization"] is False


def test_data_analysis_mode_rejects_what_is_bar_chart_and_definitions():
    """
    Direct test for user query:
    'what is bar chart' in Data Analysis mode must NOT generate a plot.
    It is a conceptual/definition question and should be rejected with OFF_TOPIC
    so the user is redirected to Chat (Discuss Insights) mode.
    """
    cols = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Fare"]
    concept_queries = [
        "what is bar chart",
        "what is a bar chart?",
        "what is histogram",
        "what is a box plot",
        "define scatter plot",
        "explain standard deviation",
        "what is correlation",
    ]
    for q in concept_queries:
        state = {
            "query": q,
            "chat_mode": "analysis",
            "columns": cols,
        }
        result = understand_query_node(state)
        assert result["intent"] == "OFF_TOPIC", f"Query '{q}' should be OFF_TOPIC, but got {result['intent']}"
        assert result["off_topic"] is True
        assert result["needs_visualization"] is False


def test_actionable_vs_definition_intent():
    """
    Ensure actionable plotting commands work, while pure definition queries are blocked.
    """
    cols = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Fare"]
    # Actionable:
    for actionable_q in ["plot a bar chart of pclass", "bar chart of pclass", "show histogram of age"]:
        state = {
            "query": actionable_q,
            "chat_mode": "analysis",
            "columns": cols,
        }
        res = understand_query_node(state)
        assert res["intent"] == "VISUALIZATION", f"Actionable query '{actionable_q}' should be VISUALIZATION, got {res['intent']}"
        assert res["needs_visualization"] is True
        assert res["off_topic"] is False



def test_data_analysis_mode_rejects_general_knowledge_and_chitchat():
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    for q in ["what is the capital of france", "hello there", "tell me a joke", "write a python function to reverse a string"]:
        state = {
            "query": q,
            "chat_mode": "analysis",
            "columns": cols,
        }
        res = understand_query_node(state)
        assert res["intent"] == "OFF_TOPIC", f"Query '{q}' should be OFF_TOPIC but was {res.get('intent')}"
        assert res["off_topic"] is True


def test_data_analysis_mode_allows_valid_dataset_queries():
    cols = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Fare"]
    # 1. Visualization query with column
    state1 = {
        "query": "plot a histogram of fare",
        "chat_mode": "analysis",
        "columns": cols,
    }
    res1 = understand_query_node(state1)
    assert res1["intent"] == "VISUALIZATION"
    assert res1["off_topic"] is False
    assert res1["needs_visualization"] is True

    # 2. Statistical query with column
    state2 = {
        "query": "what is the average age of passengers?",
        "chat_mode": "analysis",
        "columns": cols,
    }
    res2 = understand_query_node(state2)
    assert res2["intent"] == "DATA_ONLY"
    assert res2["off_topic"] is False
    assert res2["needs_visualization"] is False

    # 3. General dataset correlation query
    state3 = {
        "query": "show correlation heatmap",
        "chat_mode": "analysis",
        "columns": cols,
    }
    res3 = understand_query_node(state3)
    assert res3["intent"] == "VISUALIZATION"
    assert res3["off_topic"] is False


def test_workflow_structure_and_nodes():
    app = build_workflow()
    assert app is not None


def test_screenshot_plot_a_distribution_requires_clarification():
    """
    Direct test for the user screenshot bug:
    'plot a distribution' without column must NOT directly plot Age.
    It MUST trigger needs_clarification with suggested questions.
    """
    state = {
        "query": "plot a distribution",
        "chat_mode": "analysis",
        "columns": ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Fare"],
        "numeric_columns": ["PassengerId", "Pclass", "Age", "Fare"],
        "categorical_columns": ["Survived", "Name", "Sex"],
    }
    result = understand_query_node(state)
    assert result["needs_clarification"] is True
    assert result["needs_visualization"] is False
    assert result["intent"] == "CLARIFICATION"
    assert result["off_topic"] is False
    
    # Must contain "These are some questions you may ask"
    msg = result["clarification_message"].lower()
    assert "these are some questions you may ask" in msg
    assert "plot a distribution" in msg or "distribution" in msg
    
    # Must populate suggested questions list
    assert len(result["suggested_questions"]) >= 3
    assert any("age" in sq.lower() for sq in result["suggested_questions"])


def test_ambiguous_visualization_queries_require_clarification():
    """Ambiguous chart queries without columns must ask for clarification and provide questions."""
    cols = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Fare"]
    num_cols = ["PassengerId", "Pclass", "Age", "Fare"]
    cat_cols = ["Survived", "Name", "Sex"]

    ambiguous_queries = [
        "plot a distribution",
        "plot distribution",
        "plot a histogram",
        "draw a plot",
        "plot a bar chart",
        "make a box plot",
        "visualize",
    ]

    for q in ambiguous_queries:
        state = {
            "query": q,
            "chat_mode": "analysis",
            "columns": cols,
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
        }
        res = understand_query_node(state)
        assert res["needs_clarification"] is True, f"Query '{q}' should need clarification"
        assert res["needs_visualization"] is False
        assert "these are some questions you may ask" in res["clarification_message"].lower()
        assert len(res["suggested_questions"]) >= 3


def test_specific_column_distribution_plots_without_clarification():
    """When user specifies column 'age', it must plot without asking for clarification."""
    state = {
        "query": "plot a distribution of age",
        "chat_mode": "analysis",
        "columns": ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Fare"],
        "numeric_columns": ["PassengerId", "Pclass", "Age", "Fare"],
        "categorical_columns": ["Survived", "Name", "Sex"],
    }
    result = understand_query_node(state)
    assert result["needs_clarification"] is False
    assert result["needs_visualization"] is True
    assert result["intent"] == "VISUALIZATION"


def test_is_valid_suggested_question_rejects_dataset_descriptions():
    """Verifies that non-question descriptions and preamble text are strictly rejected."""
    bad_inputs = [
        "The dataset you've provided is a subset of The Kaggle House Prices dataset, which is commonly used for regression tasks.",
        "Id: A unique identifier for each house in the dataset.",
        "MSSubClass: The type of dwelling.",
        "Here's a breakdown of the key features:",
        "Here are some questions you can ask:",
    ]
    for bad in bad_inputs:
        assert is_valid_suggested_question(bad) is False, f"Should reject: {bad}"

    good_inputs = [
        "Plot the distribution of SalePrice",
        "What is the average SalePrice grouped by Neighborhood?",
        "Scatter plot of GrLivArea vs SalePrice",
        "Show a correlation heatmap across all numerical columns",
        "How does OverallQual relate to SalePrice?",
    ]
    for good in good_inputs:
        assert is_valid_suggested_question(good) is True, f"Should accept: {good}"


def test_generate_suggestions_node_fallback_quality():
    """Verifies that generate_suggestions_node always provides high quality grounded questions."""
    state = {
        "query": "",
        "chat_mode": "analysis",
        "columns": ["Id", "MSSubClass", "LotArea", "OverallQual", "Neighborhood", "SalePrice"],
        "numeric_columns": ["Id", "MSSubClass", "LotArea", "OverallQual", "SalePrice"],
        "categorical_columns": ["Neighborhood"],
        "dataset_info": "Columns: Id, MSSubClass, LotArea, OverallQual, Neighborhood, SalePrice",
        "provider": "dummy",
        "model_name": "dummy",
        "api_key": "dummy",
    }
    result = generate_suggestions_node(state)
    questions = result.get("suggested_questions", [])
    assert len(questions) >= 3
    # None of the questions should contain descriptive preambles or column definitions
    for q in questions:
        assert is_valid_suggested_question(q) is True
        assert not any(p in q.lower() for p in ["dataset", "breakdown", "dwelling", "unique identifier"])


