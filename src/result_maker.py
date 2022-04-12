"""
Creates an HTML-file containing information about the results of the
experiment that was run.

Added by @dansah
"""

def add_architecture(html_str, arch_name, diagrams):
    """
    Adds a header for an architecture and image tags referring to the
    result diagrams provided.
    html_str (string): The HTML-string to append the result to.
    arch_name (string): Name of the architecture.
    diagrams (dict): { "diagram_name1": "diagram_filepath1", ... }.
    """
    additional_str = """<h3>%s</h3>""" % arch_name
    for diagram_name in diagrams:
        additional_str += """<img src=\"%s\" alt=\"%s\">""" % (diagrams[diagram_name], diagram_name)

    return html_str + additional_str


def add_environment(html_str, env_name, architecture_diagrams):
    """
    Adds a header for an environment. Calls add_architecture for
    every architecture used with the enivronment.
    html_str (string): The HTML-string to append the result to.
    env_name (string): The name of the environment.
    architecture_diagrams (dict): { "arch_name1": { "diagram_name1": "diagram_filepath1", ... }, ... }
    """
    additional_str = """<h2>%s</h2>""" % env_name
    for arch_name in architecture_diagrams:
        additional_str = add_architecture(additional_str, arch_name, architecture_diagrams[arch_name])

    return html_str + additional_str


def make_html(output_file_str, environment_architecture_data):
    """
    output_file_str (string): Full filepath to the resulting output file.
    environment_architecture_data (dict): Should have the following format:
        {
            "environment_name1": {
                "arch_name1": {
                    "diagram_name1": "diagram_filepath1",
                    ...
                },
                ...
            },
            ...
        }
    """
    # Initialize
    file = open(output_file_str, 'w')
    html_str = """<html><head></head><body>"""

    # Add content
    for env_name in environment_architecture_data:
        html_str = add_environment(html_str, env_name, environment_architecture_data[env_name])

    # Finish up
    html_str += "</body></html>"
    file.write(html_str)
    file.close()
