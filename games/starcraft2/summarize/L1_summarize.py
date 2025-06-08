import json
import weave


@weave.op()
def generate_summarize_L1(information):
    def create_summary(category_data):
        summary = ""
        for key, value in category_data.items():
            if isinstance(value, dict):  # Special handling of cases with submodules
                sub_summary = create_summary(value)
                if sub_summary != "":
                    summary += f"\n{key.replace('_', ' ').capitalize()}:\n{sub_summary}"
            elif value != 0:  # Add to summary only if value is not 0
                summary += f"- {key.replace('_', ' ').capitalize()}: {value}\n"
        return summary
    # Convert a string value to a dictionary
    for key in information:
        if isinstance(information[key], str):
            try:
                information[key] = json.loads(information[key].replace("'", "\""))  # Replace single quotes with double quotes to ensure the JSON format is correct
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse value of '{key}' as JSON. Value: {information[key]}")


    if not isinstance(information.get('resource'), dict):
        raise ValueError(f"Expected 'resource' to be a dictionary, but got: {type(information.get('resource'))}, value: {information.get('resource')}")

    game_time = information['resource'].get('game_time', "unknown time")

    summary = f"At {game_time} game time, our current StarCraft II situation is as follows:\n\n"

    categories = [
        ("Resources", information.get("resource", {})),
        ("Buildings", information.get("building", {})),
        ("Units", information.get("unit", {})),
        ("Planning", information.get("planning", {})),
        ("Research", information.get("research", {})),
        ("Enemy", information.get("enemy", {})),
        # ... 
    ]

    for category, category_data in categories:
        category_summary = create_summary(category_data)
        if category_summary != "":
            summary += f"{category}:\n{category_summary}\n"

            print(f"{category}:\n{category_summary}")

    return summary
