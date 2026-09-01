from lib.envelope import EnvelopeBuilder


def test_glossary_modes_override_near_case_and_language_filtering():
    glossary = {
        "terms": {
            "Exact": {"mode": "exact", "t": {"es": "Exacto", "fr": "Exact"}},
            "Chest": {"mode": "stem", "t": {"es": "Cofre", "fr": "Coffre"}},
            "Chaos": {
                "mode": "stem",
                "cs": True,
                "near": r"\bFaction\b",
                "t": {"es": "Caos"},
            },
            "Ignored": {"mode": "skip", "t": {"es": "Ignorado"}},
            "Brand": {"mode": "keep"},
            "Hero": {"mode": "exact", "t": {"es": "Viejo"}},
        },
        "manual": {
            "Hero": {"mode": "stem", "t": {"es": "Héroe", "fr": "Héros"}}
        },
    }
    phrases = [
        "Exact",
        "Exact value",
        "Open Chests",
        "Chaos Faction",
        "chaos Faction",
        "Chaos everywhere",
        "Ignored",
        "Brand name",
        "Heroic",
    ]
    rows = [{"en": phrase, "es": ""} for phrase in phrases]
    builder = EnvelopeBuilder(glossary, rows, "en", ["es"], [])
    batch = builder.build(
        [(phrase, None) for phrase in phrases],
        {phrase: index for index, phrase in enumerate(phrases)},
    ).model_dump(exclude_none=True)

    assert batch["glossary"] == [
        {"term": "Exact", "mode": "exact", "t": {"es": "Exacto"}},
        {"term": "Chest", "mode": "stem", "t": {"es": "Cofre"}},
        {
            "term": "Chaos",
            "mode": "stem",
            "cs": True,
            "near": r"\bFaction\b",
            "t": {"es": "Caos"},
        },
        {"term": "Brand", "mode": "keep"},
        {"term": "Hero", "mode": "stem", "t": {"es": "Héroe"}},
    ]
    assert all(set(entry.get("t", {})) <= {"es"} for entry in batch["glossary"])


def test_references_and_neighbor_examples_are_optional():
    glossary = {
        "terms": {
            "Tina": {"mode": "stem", "t": {"pl": "Tina"}},
            "Rayne": {"mode": "stem", "t": {"pl": "Rayne"}},
        }
    }
    rows = [
        {"en": "Tina Pack", "ru": "Набор Тины", "pl": ""},
        {"en": "Rayne Pack", "ru": "", "pl": "Pakiet Rayne"},
        {"en": "Standalone", "ru": "", "pl": "Samodzielny"},
    ]
    builder = EnvelopeBuilder(glossary, rows, "en", ["pl"], ["ru"])

    batch = builder.build(
        [("Tina Pack", "Shop item"), ("Standalone", None)],
        {"Tina Pack": 0, "Standalone": 2},
    ).model_dump(exclude_none=True)

    assert batch["phrases"][0] == {
        "phrase": "Tina Pack",
        "context": "Shop item",
        "reference": {"ru": "Набор Тины"},
        "examples": [{"phrase": "Rayne Pack", "t": {"pl": "Pakiet Rayne"}}],
    }
    assert batch["phrases"][1] == {"phrase": "Standalone"}


def test_glossary_is_limited_by_phrase_frequency_and_may_be_absent():
    terms = {
        f"Token{index:02}": {"mode": "stem", "t": {"es": str(index)}}
        for index in range(21)
    }
    all_terms = " ".join(terms)
    rows = [{"en": all_terms}, {"en": "Token20 again"}]
    builder = EnvelopeBuilder({"terms": terms}, rows, "en", ["es"], [])

    batch = builder.build(
        [(all_terms, None), ("Token20 again", None)],
        {all_terms: 0, "Token20 again": 1},
    ).model_dump(exclude_none=True)

    assert len(batch["glossary"]) == 20
    assert batch["glossary"][0]["term"] == "Token20"
    assert "Token19" not in {entry["term"] for entry in batch["glossary"]}

    no_glossary = EnvelopeBuilder({}, [{"en": "Hello"}], "en", ["es"], [])
    assert no_glossary.build([("Hello", None)], {"Hello": 0}).model_dump(
        exclude_none=True
    ) == {"phrases": [{"phrase": "Hello"}]}
