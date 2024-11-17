from personality_profile import PersonalityProfile


def extract_features(profile: PersonalityProfile) -> list:
    """Extracts features from a personality profile for PCA."""
    features = []
    for trait, values in profile.items():
        features.append(values["overall"])
        features.extend(values["sub"].values())
    return features
