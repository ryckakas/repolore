from dataclasses import dataclass, replace
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response

from repolore.clients import CLIENTS, McpClient, RegistrationError, repolore_entries
from repolore.doctor import CLIENT_ERRORS, Check, registration_check
from repolore.instructions import INSTRUCTIONS
from repolore.ui.rendering import render


@dataclass(frozen=True)
class ClientRow:
    key: str
    check: Check
    can_register: bool


class ClientPages:
    def __init__(self, db: str) -> None:
        self._db = db

    def page(self, request: Request) -> Response:
        return render(request, self._db, "clients.html", instructions=INSTRUCTIONS)

    def listing(self, request: Request) -> Response:
        return self._fragment(request)

    async def register(self, request: Request) -> Response:
        form = await request.form()
        key = form.get("client")
        return await run_in_threadpool(self._register, request, key if isinstance(key, str) else "")

    def _register(self, request: Request, key: str) -> Response:
        try:
            message = _registered(key)
        except CLIENT_ERRORS as error:
            return self._fragment(request, 400, notice=f"Couldn't register repolore: {error}")
        return self._fragment(request, message=message)

    def _fragment(self, request: Request, status_code: int = 200, **context: Any) -> Response:
        rows: list[ClientRow] = []
        missing: list[str] = []
        for client in CLIENTS:
            row = _row(client)
            if row is None:
                missing.append(client.name)
            else:
                rows.append(row)
        return render(request, self._db, "_clients.html", status_code, rows=rows, missing=missing, **context)


def _row(client: McpClient) -> ClientRow | None:
    # Not the connection: asking Claude Code about it starts the shared server, and so the model.
    check = registration_check(client, connection=False)
    if check is None:
        return None
    if check.warning:
        return ClientRow(client.key, replace(check, fix=None), can_register=True)
    return ClientRow(client.key, check, can_register=False)


def _registered(key: str) -> str:
    # The key only picks a row of the table, so a request can't make the page run or write anything else.
    client = next((client for client in CLIENTS if client.key == key and client.present()), None)
    if client is None:
        raise RegistrationError(f"{key!r} isn't an MCP client on this machine.")
    if repolore_entries(client):
        raise RegistrationError(f"{client.name} already has it.")
    return f"{client.name}: registered repolore {client.add()}."
