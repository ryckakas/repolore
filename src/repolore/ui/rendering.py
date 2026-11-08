from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import Response
from starlette.templating import Jinja2Templates

from repolore import __version__

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def render(request: Request, db: str, template: str, status_code: int = 200, **context: Any) -> Response:
    return templates.TemplateResponse(
        request, template, {"version": __version__, "db": db, **context}, status_code=status_code
    )
