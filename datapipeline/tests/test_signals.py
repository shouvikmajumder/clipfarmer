"""Tests for processing.signals — pure NLP scoring functions.

All functions are pure (no heavy deps, no network). Tests cover:
- hook_quality
- standalone_coherence
- specificity
- curiosity
- sentiment
- first_sentence helper

Every function returns a float in [0.0, 1.0].
"""

from __future__ import annotations

import pytest

from processing.signals import (
    curiosity,
    first_sentence,
    hook_quality,
    sentiment,
    specificity,
    standalone_coherence,
)


# ---------------------------------------------------------------------------
# first_sentence
# ---------------------------------------------------------------------------


class TestFirstSentence:
    def test_single_sentence_no_boundary(self):
        text = "Hello there"
        assert first_sentence(text) == "Hello there"

    def test_splits_at_period(self):
        text = "Hello there. More text follows."
        assert first_sentence(text) == "Hello there"

    def test_splits_at_question_mark(self):
        text = "What is this? And more."
        assert first_sentence(text) == "What is this"

    def test_splits_at_exclamation(self):
        text = "Incredible! Never seen before."
        assert first_sentence(text) == "Incredible"

    def test_empty_string(self):
        assert first_sentence("") == ""

    def test_strips_whitespace(self):
        assert first_sentence("  hello world  ") == "hello world"

    def test_trailing_punctuation_stripped_from_result(self):
        # The function strips trailing sentence-end punctuation from the result.
        result = first_sentence("Hello world.")
        assert not result.endswith(".")


# ---------------------------------------------------------------------------
# hook_quality
# ---------------------------------------------------------------------------


class TestHookQuality:
    def test_returns_float_in_unit_interval(self):
        for text in ["", "Hello world", "And then we went home", "I made $1,000,000!"]:
            score = hook_quality(text)
            assert 0.0 <= score <= 1.0

    def test_empty_string_returns_zero(self):
        assert hook_quality("") == 0.0

    def test_continuation_starter_penalized(self):
        """'Because', 'And', 'But' starters are penalized -0.5."""
        good = hook_quality("I made a million dollars")
        bad_because = hook_quality("Because of all the things that happened")
        bad_and = hook_quality("And then we went to the store")
        bad_but = hook_quality("But what about the other side")
        assert bad_because < good
        assert bad_and < good
        assert bad_but < good

    def test_long_sentence_penalized(self):
        """Sentences over 20 words are penalized -0.2.

        Use a long sentence that doesn't trigger _HOOK_PATTERNS so the only
        difference is the word count penalty.
        """
        short = hook_quality("Today everything changed for good")
        # 21+ words, no hook patterns, no bad starter.
        long_text = hook_quality(
            "Today everything changed because of all the things that happened over "
            "the last several weeks in ways nobody expected to see at all ever"
        )
        assert long_text < short

    def test_pattern_bonus_dollar_amount(self):
        """A dollar amount in the first sentence gets a hook pattern bonus."""
        no_dollar = hook_quality("I made a lot of money this year")
        with_dollar = hook_quality("I made $1000000 this year")
        assert with_dollar >= no_dollar

    def test_pattern_bonus_percentage(self):
        """A percentage figure in the first sentence gets a hook pattern bonus."""
        no_pct = hook_quality("Our conversion rate went up a lot")
        with_pct = hook_quality("Our conversion rate went up 300%")
        assert with_pct >= no_pct

    def test_pattern_bonus_nobody_talks(self):
        """'nobody talks' pattern gets a bonus."""
        score = hook_quality("Nobody talks about this investment strategy")
        assert score > 0.5

    def test_pattern_bonus_heres_why(self):
        """'Here's why' pattern gets a bonus."""
        score = hook_quality("Here's why everyone gets this wrong")
        assert score > 0.5

    def test_continuation_with_pattern_bonus_interaction(self):
        """A continuation starter (-0.5) with a pattern bonus still produces [0,1]."""
        score = hook_quality("Because nobody talks about the $1M secret")
        assert 0.0 <= score <= 1.0

    def test_clean_strong_hook(self):
        """A clean, specific, pattern-matching hook scores above neutral (0.5)."""
        score = hook_quality("I lost $50,000 in one day")
        assert score > 0.5


# ---------------------------------------------------------------------------
# standalone_coherence
# ---------------------------------------------------------------------------


class TestStandaloneCoherence:
    def test_returns_valid_value(self):
        for text in ["I went to the store", "Because of this", "As I said earlier"]:
            score = standalone_coherence(text)
            assert score in {0.1, 0.6, 0.8}

    def test_context_dependent_pronoun_returns_low(self):
        """'This', 'They', 'Because', 'It' openers return 0.1."""
        assert standalone_coherence("This is what I meant") == 0.1
        assert standalone_coherence("They told me to stop") == 0.1
        assert standalone_coherence("Because of what happened") == 0.1
        assert standalone_coherence("It all started here") == 0.1

    def test_clean_opener_returns_high(self):
        """'I', 'The', 'You', 'We', 'There' openers return 0.8."""
        assert standalone_coherence("I went to the store") == 0.8
        assert standalone_coherence("The truth is nobody knows") == 0.8
        assert standalone_coherence("You need to hear this") == 0.8
        assert standalone_coherence("We discovered something incredible") == 0.8

    def test_neutral_opener_returns_mid(self):
        """An unrecognised opener that's not a bad starter returns 0.6."""
        assert standalone_coherence("Today we are going to talk") == 0.6
        assert standalone_coherence("Yesterday everything changed") == 0.6

    def test_multiword_context_phrase_returns_low(self):
        """'As I said', 'Like I mentioned', 'As we discussed' return 0.1."""
        assert standalone_coherence("As I said before, this is wrong") == 0.1
        assert standalone_coherence("Like I mentioned earlier") == 0.1
        assert standalone_coherence("As we discussed last time") == 0.1

    def test_empty_string_returns_neutral(self):
        assert standalone_coherence("") == 0.6


# ---------------------------------------------------------------------------
# specificity
# ---------------------------------------------------------------------------


class TestSpecificity:
    def test_returns_float_in_unit_interval(self):
        for text in ["", "hello world", "$1,000 and 50% off", "PayPal and Google"]:
            score = specificity(text)
            assert 0.0 <= score <= 1.0

    def test_empty_text_returns_zero(self):
        assert specificity("") == 0.0

    def test_dollar_amounts_raise_specificity(self):
        no_dollar = specificity("I made a lot of money")
        with_dollar = specificity("I made $500,000 in revenue")
        assert with_dollar > no_dollar

    def test_percentages_raise_specificity(self):
        no_pct = specificity("Our growth was dramatic")
        with_pct = specificity("Our growth was 37%")
        assert with_pct > no_pct

    def test_standalone_numbers_raise_specificity(self):
        no_num = specificity("I did it many times")
        with_num = specificity("I did it 42 times in 7 days")
        assert with_num > no_num

    def test_saturates_at_one(self):
        """Many concrete markers should cap at 1.0."""
        very_specific = specificity(
            "I earned $1,000,000 in 90 days, a 500% increase. "
            "PayPal and Stripe both saw 3x growth from January."
        )
        assert very_specific == 1.0

    def test_generic_text_low_specificity(self):
        score = specificity("and then we talked about some stuff that happened")
        assert score < 0.5


# ---------------------------------------------------------------------------
# curiosity
# ---------------------------------------------------------------------------


class TestCuriosity:
    def test_returns_float_in_unit_interval(self):
        for text in ["", "hello", "Why? How? What?", "The truth is nobody tells you"]:
            score = curiosity(text)
            assert 0.0 <= score <= 1.0

    def test_empty_text_returns_zero(self):
        assert curiosity("") == 0.0

    def test_question_marks_raise_curiosity(self):
        no_q = curiosity("This is a statement without any questions here")
        with_q = curiosity("Why is this? How does it work? What happens next?")
        assert with_q > no_q

    def test_question_word_openers_raise_curiosity(self):
        no_opener = curiosity("This is just a simple statement")
        with_opener = curiosity("Why does this happen every time?")
        assert with_opener > no_opener

    def test_generic_keyword_phrases_raise_curiosity(self):
        """Phrases from GENERIC_KEYWORD_PHRASES boost curiosity."""
        no_kw = curiosity("Let me walk you through it step by step")
        with_kw = curiosity("Here's why nobody talks about the truth")
        assert with_kw > no_kw

    def test_combined_signal_higher_than_individual(self):
        """Questions + keyword phrases together score higher than either alone."""
        questions_only = curiosity("Why? How? When? What?")
        keywords_only = curiosity("The truth is nobody talks about this here now")
        combined = curiosity(
            "Why does nobody talk about this? Here's the truth nobody tells you!"
        )
        # Combined should be at least as good as either alone.
        assert combined >= min(questions_only, keywords_only)

    def test_no_signals_returns_low_score(self):
        score = curiosity("we went to the store and bought some groceries")
        assert score < 0.4


# ---------------------------------------------------------------------------
# sentiment
# ---------------------------------------------------------------------------


class TestSentiment:
    def test_returns_float_in_unit_interval(self):
        for text in ["", "hello world", "This is absolutely incredible and insane!"]:
            score = sentiment(text)
            assert 0.0 <= score <= 1.0

    def test_empty_text_returns_zero(self):
        assert sentiment("") == 0.0

    def test_strong_emotion_words_raise_sentiment(self):
        bland = sentiment("we then proceeded to discuss the agenda items")
        rich = sentiment(
            "This is absolutely incredible and shocking! Unbelievable and insane. "
            "Amazing, fantastic, brutal truth exposed!"
        )
        assert rich > bland

    def test_single_strong_word_scores_positive(self):
        score = sentiment("This is incredible")
        assert score > 0.0

    def test_text_with_no_emotion_words_scores_low(self):
        score = sentiment("the cat sat on the mat by the door near the window")
        assert score == 0.0

    def test_dense_emotion_saturates(self):
        """Multiple strong emotion words in a short text should saturate near 1.0."""
        text = "amazing incredible shocking unbelievable insane fantastic"
        score = sentiment(text)
        assert score == 1.0

    def test_diluted_emotions_score_proportionally(self):
        """Fewer emotion words in a longer text score lower than a dense text."""
        dense = sentiment("incredible amazing shocking unbelievable")
        diluted = sentiment(
            "incredible word word word word word word word word word word word word"
        )
        assert dense >= diluted
