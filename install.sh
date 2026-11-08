#!/bin/sh
# Installs repolore with uv, then runs `repolore setup` with any arguments given:
#
#   curl -fsSL https://raw.githubusercontent.com/ryckakas/repolore/main/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/ryckakas/repolore/main/install.sh | sh -s -- --include "you/*" --register
#
# REPOLORE_INSTALL_FROM_GIT=1 installs the latest main instead of the last release.
#
# Everything runs from main, called on the last line, so a download cut short runs nothing. Stdin is this script when
# it's piped in, so commands get /dev/null, and setup gets the terminal.

PACKAGE=repolore-mcp
GIT_SOURCE=git+https://github.com/ryckakas/repolore
UV_INSTALLER=https://astral.sh/uv/install.sh
UV_DOCS=https://docs.astral.sh/uv/getting-started/installation/

say() {
    printf 'repolore: %s\n' "$1"
}

fail() {
    printf 'repolore: %s\n' "$1" >&2
    exit 1
}

has_uv() {
    command -v uv >/dev/null 2>&1
}

uv_folder() {
    if [ -n "${XDG_BIN_HOME:-}" ]; then
        printf '%s\n' "$XDG_BIN_HOME"
    elif [ -n "${XDG_DATA_HOME:-}" ]; then
        printf '%s\n' "$XDG_DATA_HOME/../bin"
    else
        printf '%s\n' "$HOME/.local/bin"
    fi
}

ensure_uv() {
    has_uv && return 0
    say "installing uv"
    curl -fsSL "$UV_INSTALLER" | sh
    PATH="$(uv_folder):$PATH"
    export PATH
    has_uv || fail "couldn't install uv; see $UV_DOCS"
}

is_installed() {
    listed=$(uv tool list </dev/null 2>/dev/null) || fail "couldn't list uv's tools"
    printf '%s\n' "$listed" | grep -q "^$PACKAGE "
}

install_repolore() {
    if [ "${REPOLORE_INSTALL_FROM_GIT:-}" = 1 ]; then
        say "installing the latest main; extras such as rerank have to be added again"
        uv tool install --managed-python --python 3.12 --upgrade "$GIT_SOURCE" </dev/null ||
            fail "couldn't install $GIT_SOURCE"
    elif is_installed; then
        uv tool upgrade "$PACKAGE" </dev/null || fail "couldn't upgrade $PACKAGE"
    else
        uv tool install --managed-python --python 3.12 "$PACKAGE" </dev/null || fail "couldn't install $PACKAGE"
    fi
}

put_tools_on_path() {
    tools=$(uv tool dir --bin </dev/null) || fail "couldn't find uv's tool folder"
    case ":$PATH:" in
    *":$tools:"*) ;;
    *)
        # It errors when the shell files have the folder but this process doesn't, which is the case right after a
        # first install.
        uv tool update-shell </dev/null >/dev/null 2>&1 || true
        PATH="$tools:$PATH"
        export PATH
        say "open a new terminal to use the repolore command"
        ;;
    esac
}

run_setup() {
    if (: </dev/tty) 2>/dev/null; then
        repolore setup "$@" </dev/tty
    elif [ "$#" -gt 0 ]; then
        repolore setup "$@" </dev/null ||
            fail "setup stopped; its output says why, and without a terminal every question needs its flag"
    else
        say "installed; there's no terminal to ask questions in, so run \`repolore setup\` in one to finish"
    fi
}

main() {
    ensure_uv
    install_repolore
    put_tools_on_path
    run_setup "$@"
}

main "$@" || exit 1
