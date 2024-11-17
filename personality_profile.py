from typing import TypedDict, Union

import numpy as np


# Define the structure of the profile using TypedDict
class SubTrait(TypedDict):
    altruism: float
    cooperation: float
    modesty: float
    morality: float
    sympathy: float
    trust: float


class ConscientiousnessSubTrait(TypedDict):
    achievement_striving: float
    cautiousness: float
    dutifulness: float
    orderliness: float
    self_discipline: float
    self_efficacy: float


class ExtraversionSubTrait(TypedDict):
    activity_level: float
    assertiveness: float
    cheerfulness: float
    excitement_seeking: float
    friendliness: float
    gregariousness: float


class NeuroticismSubTrait(TypedDict):
    anger: float
    anxiety: float
    depression: float
    immoderation: float
    self_consciousness: float
    vulnerability: float


class OpennessSubTrait(TypedDict):
    adventurousness: float
    artistic_interests: float
    emotionality: float
    imagination: float
    intellect: float
    liberalism: float


class Trait(TypedDict):
    overall: float
    sub: Union[
        SubTrait,
        ConscientiousnessSubTrait,
        ExtraversionSubTrait,
        NeuroticismSubTrait,
        OpennessSubTrait,
    ]


class PersonalityProfileTemplate(TypedDict):
    agreeableness: Trait
    conscientiousness: Trait
    extraversion: Trait
    neuroticism: Trait
    openness: Trait


class PersonalityProfile:
    def __init__(self, profile: PersonalityProfileTemplate = None):
        # Initialize the profile with the template structure or a provided profile
        self.profile: PersonalityProfileTemplate = (
            profile
            if profile
            else {
                "agreeableness": {
                    "overall": 0.0,
                    "sub": {
                        "altruism": 0.0,
                        "cooperation": 0.0,
                        "modesty": 0.0,
                        "morality": 0.0,
                        "sympathy": 0.0,
                        "trust": 0.0,
                    },
                },
                "conscientiousness": {
                    "overall": 0.0,
                    "sub": {
                        "achievement_striving": 0.0,
                        "cautiousness": 0.0,
                        "dutifulness": 0.0,
                        "orderliness": 0.0,
                        "self_discipline": 0.0,
                        "self_efficacy": 0.0,
                    },
                },
                "extraversion": {
                    "overall": 0.0,
                    "sub": {
                        "activity_level": 0.0,
                        "assertiveness": 0.0,
                        "cheerfulness": 0.0,
                        "excitement_seeking": 0.0,
                        "friendliness": 0.0,
                        "gregariousness": 0.0,
                    },
                },
                "neuroticism": {
                    "overall": 0.0,
                    "sub": {
                        "anger": 0.0,
                        "anxiety": 0.0,
                        "depression": 0.0,
                        "immoderation": 0.0,
                        "self_consciousness": 0.0,
                        "vulnerability": 0.0,
                    },
                },
                "openness": {
                    "overall": 0.0,
                    "sub": {
                        "adventurousness": 0.0,
                        "artistic_interests": 0.0,
                        "emotionality": 0.0,
                        "imagination": 0.0,
                        "intellect": 0.0,
                        "liberalism": 0.0,
                    },
                },
            }
        )

    def validate_and_update_profile(
        self, new_profile: PersonalityProfileTemplate
    ) -> None:
        """Validate and update the entire profile."""
        for trait, values in new_profile.items():
            # Validate overall score
            overall_value = values.get("overall")
            self.profile[trait]["overall"] = self.validate_value(overall_value)

            # Validate sub-traits
            for sub_trait, sub_value in values.get("sub", {}).items():
                self.profile[trait]["sub"][sub_trait] = self.validate_value(sub_value)

    def validate_value(self, value: Union[int, float]) -> float:
        """Ensure the value is between 0 and 1, or convert positive integers by dividing by 100."""
        if isinstance(value, float) and 0 < value <= 1:
            return round(float(value) * 100)
        elif isinstance(value, int) and value >= 0:
            return value
        else:
            raise ValueError(
                "Value must be a float between 0 and 1, or a positive integer."
            )

    def extract(self) -> list:
        """Convert the profile to a list for PCA analysis."""
        features = []
        for trait, values in self.profile.items():
            features.append(values["overall"])
            features.extend(values["sub"].values())
        return features

    def to_np_array(self) -> np.array:
        """Convert the profile to a numpy array for PCA analysis."""
        return np.array(self.extract())

    def overall_to_np_array(self) -> np.array:
        """Convert the overall scores of the profile to a numpy array for PCA analysis."""
        return np.array([trait["overall"] for trait in self.profile.values()])

    def excluding_overall_to_np_array(self) -> np.array:
        """Convert the profile to a numpy array excluding overall scores for PCA analysis."""
        return np.array(
            [trait["sub"].values() for trait in self.profile.values()]
        ).flatten()
