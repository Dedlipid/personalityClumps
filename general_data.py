# Define dimension names
dimension_names = [
    "agreeableness_overall",
    "altruism",
    "cooperation",
    "modesty",
    "morality",
    "sympathy",
    "trust",
    "conscientiousness_overall",
    "achievement_striving",
    "cautiousness",
    "dutifulness",
    "orderliness",
    "self_discipline",
    "self_efficacy",
    "extraversion_overall",
    "activity_level",
    "assertiveness",
    "cheerfulness",
    "excitement_seeking",
    "friendliness",
    "gregariousness",
    "neuroticism_overall",
    "anger",
    "anxiety",
    "depression",
    "immoderation",
    "self_consciousness",
    "vulnerability",
    "openness_overall",
    "adventurousness",
    "artistic_interests",
    "emotionality",
    "imagination",
    "intellect",
    "liberalism",
]

# Indices for overall scores
overall_indices = [i for i, name in enumerate(dimension_names) if "overall" in name]

# Indices excluding overall scores
no_overall_indices = [
    i for i in range(len(dimension_names)) if i not in overall_indices
]

# Define the template for the personality profile
template = {
    "extraversion": {
        "overall": 0,
        "sub": {
            "activity_level": 0,
            "assertiveness": 0,
            "cheerfulness": 0,
            "excitement_seeking": 0,
            "friendliness": 0,
            "gregariousness": 0,
        },
    },
    "agreeableness": {
        "overall": 0,
        "sub": {
            "altruism": 0,
            "cooperation": 0,
            "modesty": 0,
            "morality": 0,
            "sympathy": 0,
            "trust": 0,
        },
    },
    "conscientiousness": {
        "overall": 0,
        "sub": {
            "achievement_striving": 0,
            "cautiousness": 0,
            "dutifulness": 0,
            "orderliness": 0,
            "self_discipline": 0,
            "self_efficacy": 0,
        },
    },
    "neuroticism": {
        "overall": 0,
        "sub": {
            "anger": 0,
            "anxiety": 0,
            "depression": 0,
            "immoderation": 0,
            "self_consciousness": 0,
            "vulnerability": 0,
        },
    },
    "openness": {
        "overall": 0,
        "sub": {
            "adventurousness": 0,
            "artistic_interests": 0,
            "emotionality": 0,
            "imagination": 0,
            "intellect": 0,
            "liberalism": 0,
        },
    },
}
