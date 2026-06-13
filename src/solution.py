def get_solution(text, severity):

    text = text.lower()

    if "crash" in text:
        return "Check memory usage and debug crash-related modules."

    if "nullpointer" in text:
        return "Initialize variables before usage."

    if severity == 0:
        return "Minor issue. Fix UI/text alignment."

    if severity == 1:
        return "Check module functionality."

    if severity == 2:
        return "Debug backend logic immediately."

    return "Critical issue. Immediate system-level debugging required."