import numpy as np
from personality_profile import PersonalityProfile
from raw_data import names
from raw_dark import dark_names


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

dark_dimension_names = [    
    "narcissist",
    "machievellian",
    "psychopath",
    "average"
]
# Indices for overall scores
overall_indices = [i for i, name in enumerate(dimension_names) if "overall" in name]

# Indices excluding overall scores
no_overall_indices = [
    i for i in range(len(dimension_names)) if i not in overall_indices
]

data = np.array([PersonalityProfile(profile).to_np_array() for profile in names.values()])
labels = list(names.keys())

# Dark Triad data
dark_data = np.array([list(profile.values()) for profile in dark_names.values()])
dark_labels = list(dark_names.keys())

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
