DEPTH_BANDS = {
    0: ("0-5 cm", 5),
    1: ("5-20 cm", 15),
    2: ("20-50 cm", 35),
    3: ("50-80 cm", 65),
    4: ("80+ cm", 100)
}


def estimate_depth(severity):
    band, depth_cm = DEPTH_BANDS.get(severity, ("Unknown", 0))

    return {
        "depth_band": band,
        "depth_cm": depth_cm
    }


if __name__ == "__main__":
    for severity in range(5):
        print(severity, estimate_depth(severity))