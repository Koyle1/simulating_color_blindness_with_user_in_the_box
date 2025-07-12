from enum import Enum, auto

#color blindness simulation
class Trichromatic_view(Enum):
    NONE = auto()
    ANOMALOUS_TRICHROMACY = auto()
    DICHROMATIC = auto()
    ACHROMATOPSIA = auto()
    BLUE_CONE_MONOCHROMANCY = auto()

class Deficiency(Enum):
    NONE = auto()
    PROTAN = auto()
    DEUTAN = auto()
    TRITAN = auto()