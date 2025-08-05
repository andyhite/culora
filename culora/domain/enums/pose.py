"""Pose estimation enums."""

from enum import Enum


class PoseCategory(str, Enum):
    """Pose categories based on body position and posture."""

    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    KNEELING = "kneeling"
    CROUCHING = "crouching"
    UNKNOWN = "unknown"


class PoseOrientation(str, Enum):
    """Body orientation relative to camera."""

    FRONTAL = "frontal"
    PROFILE = "profile"
    THREE_QUARTER = "three_quarter"
    BACK = "back"
    UNKNOWN = "unknown"


class ArmPosition(str, Enum):
    """Arm positioning categories."""

    RAISED = "raised"
    EXTENDED = "extended"
    CROSSED = "crossed"
    AT_SIDES = "at_sides"
    ON_HIPS = "on_hips"
    BEHIND_BACK = "behind_back"
    UNKNOWN = "unknown"


class LegPosition(str, Enum):
    """Leg positioning categories."""

    STRAIGHT = "straight"
    BENT = "bent"
    CROSSED = "crossed"
    SPREAD = "spread"
    ONE_RAISED = "one_raised"
    UNKNOWN = "unknown"


class PoseSymmetry(str, Enum):
    """Pose symmetry classification."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    UNKNOWN = "unknown"


class PoseDynamism(str, Enum):
    """Pose dynamism level."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    ACTION = "action"
    UNKNOWN = "unknown"
