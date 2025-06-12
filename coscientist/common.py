import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

_env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "prompts")),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def load_prompt(name: str, **kwargs) -> str:
    """
    Load a template from the prompts directory and renders
    it with the given kwargs.

    Parameters
    ----------
    name: str
        The name of the template to load, without the .md extension.
    **kwargs: dict
        The kwargs to render the template with.

    Returns
    -------
    str
        The rendered template.
    """
    return _env.get_template(f"{name}.md").render(**kwargs)
