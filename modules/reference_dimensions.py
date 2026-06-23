"""
Indian reference dimensions for object-based flood depth estimation.

This module stores average human body dimensions, vehicle heights, and
stop sign dimensions for use as physical reference anchors when estimating
water depth from detected objects.
"""

HUMAN_BODY_PART_HEIGHTS = {
    "knee_height": {
        "male_cm": 40,
        "female_cm": 38,
        "description": "Distance from the bottom of the feet to the top of the knee"
    },
    "waist_height": {
        "male_cm": 90,
        "female_cm": 80,
        "description": "Distance from the ground to a person's waist"
    },
    "shoulder_height": {
        "male_cm": 135,
        "female_cm": 125,
        "description": "Distance from the ground to the top of a person's shoulder"
    },
    "head_total_height": {
        "male_cm": 165,
        "female_cm": 153,
        "description": "Total height from the ground to the top of the head"
    }
}

# Total body height averages for person-based depth reference.
PERSON_REFERENCE_HEIGHT_CM = 159

VEHICLE_REFERENCE_DIMENSIONS = {
    "sedan": {
        "ground_clearance_cm": 20,
        "ride_height_cm": 60,
        "windshield_height_cm": 100,
        "total_height_cm": 140,
        "description": "Sedan average heights"
    },
    "truck": {
        "ground_clearance_cm": 50,
        "ride_height_cm": 80,
        "windshield_height_cm": 130,
        "total_height_cm": 180,
        "description": "Truck average heights"
    },
    "bus": {
        "ground_clearance_cm": 70,
        "ride_height_cm": 100,
        "windshield_height_cm": 200,
        "total_height_cm": 320,
        "description": "Bus average heights"
    }
}

STOP_SIGN_DIMENSIONS = {
    "stop_sign_height_cm": 90,
    "pole_height_cm": 200,
    "total_height_cm": 290,
    "description": "Stop sign physical dimensions"
}

OBJECT_REFERENCE_SPECS = {
    "car": {
        "height": int(VEHICLE_REFERENCE_DIMENSIONS["sedan"]["total_height_cm"]),
        "width": 175,
        "description": "Sedan average total height and width in cm"
    },
    "truck": {
        "height": int(VEHICLE_REFERENCE_DIMENSIONS["truck"]["total_height_cm"]),
        "width": 250,
        "description": "Truck average total height and width in cm"
    },
    "bus": {
        "height": int(VEHICLE_REFERENCE_DIMENSIONS["bus"]["total_height_cm"]),
        "width": 260,
        "description": "Bus average total height and width in cm"
    },
    "motorcycle": {
        "height": 90,
        "width": 80,
        "description": "Motorcycle average height and width in cm"
    },
    "bicycle": {
        "height": 100,
        "width": 70,
        "description": "Bicycle average height and width in cm"
    },
    "person": {
        "height": PERSON_REFERENCE_HEIGHT_CM,
        "description": "Average person height in cm based on Indian reference values"
    },
    "stop sign": {
        "height": int(STOP_SIGN_DIMENSIONS["stop_sign_height_cm"]),
        "width": int(STOP_SIGN_DIMENSIONS["stop_sign_height_cm"]),
        "description": "Standard stop sign height in cm"
    },
    "stop_sign": {
        "height": int(STOP_SIGN_DIMENSIONS["stop_sign_height_cm"]),
        "width": int(STOP_SIGN_DIMENSIONS["stop_sign_height_cm"]),
        "description": "Alternate key for stop sign"
    }
}


def get_object_specs(class_name):
    """
    Return the reference object specifications for a YOLO class name.

    Args:
        class_name: Detected object class from YOLO

    Returns:
        dict: Physical size specifications in centimeters
    """
    if not class_name:
        return {}
    return OBJECT_REFERENCE_SPECS.get(class_name.lower(), {})
