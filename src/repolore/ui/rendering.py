from pathlib import Path
from typing import Any

from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import Response
from starlette.templating import Jinja2Templates

from repolore import __version__

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def form_text(form: FormData, name: str) -> str:
    value = form.get(name)
    return value.strip() if isinstance(value, str) else ""


def render(request: Request, db: str, template: str, status_code: int = 200, **context: Any) -> Response:
    return templates.TemplateResponse(
        request, template, {"version": __version__, "db": db, **context}, status_code=status_code
    )
