#!/bin/sh
# DeepLoop installer.
#
#   curl -fsSL https://raw.githubusercontent.com/<you>/deeploop/main/install.sh | sh
#
# Installs the `deeploop` command. Prefers uv, then pipx, then a private venv
# under ~/.local/share/deeploop with a symlink in ~/.local/bin.
#
# Options:
#   --git [URL]   install from a git repository (default)
#   --pypi        install from PyPI (deeploop-cli)
#   --local       install from the checkout this script lives in
#   --spec SPEC   install an explicit pip spec
#   --dry-run     print what would be installed and exit
#
# Environment:
#   DEEPLOOP_REPO       git URL to install from
#   DEEPLOOP_PYPI       PyPI distribution name
#   DEEPLOOP_HOME       install prefix for the venv fallback
#   DEEPLOOP_BIN_DIR    where to symlink the command for the venv fallback
#   DEEPLOOP_INSTALLER  force uv | pipx | venv
#   PYTHON              interpreter for the venv fallback

set -eu

REPO_URL="${DEEPLOOP_REPO:-https://github.com/coffeequantz/deeploop.git}"
PYPI_PACKAGE="${DEEPLOOP_PYPI:-deeploop-cli}"
MODE="auto"
EXPLICIT_SPEC=""
DRY_RUN=0
MIN_PYTHON="3.9"

say() { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --git)
      MODE="git"
      if [ $# -gt 1 ] && [ "${2#-}" = "$2" ]; then
        REPO_URL="$2"
        shift
      fi
      ;;
    --pypi) MODE="pypi" ;;
    --local) MODE="local" ;;
    --spec) shift; [ $# -gt 0 ] || die "--spec needs a value"; EXPLICIT_SPEC="$1"; MODE="spec" ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

SCRIPT_DIR=""
if [ -f "$0" ]; then
  SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fi

if [ "$MODE" = "auto" ]; then
  if [ -n "$EXPLICIT_SPEC" ]; then
    MODE="spec"
  elif [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ] &&
       grep -q '^name = "deeploop-cli"' "$SCRIPT_DIR/pyproject.toml" 2>/dev/null; then
    MODE="local"
  else
    MODE="git"
  fi
fi

case "$MODE" in
  local)
    [ -n "$SCRIPT_DIR" ] || die "--local requires running install.sh from the checkout"
    SPEC="$SCRIPT_DIR"
    ;;
  spec) SPEC="$EXPLICIT_SPEC" ;;
  pypi) SPEC="$PYPI_PACKAGE" ;;
  git)
    command -v git >/dev/null 2>&1 || die "git is required to install from $REPO_URL"
    SPEC="deeploop-cli @ git+$REPO_URL"
    ;;
  *) die "unknown mode: $MODE" ;;
esac

have() { command -v "$1" >/dev/null 2>&1; }

python_ok() {
  "$1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >/dev/null 2>&1
}

install_with_uv() {
  uv tool install --force "$SPEC"
}

install_with_pipx() {
  pipx install --force "$SPEC"
}

install_with_venv() {
  PYTHON="${PYTHON:-python3}"
  have "$PYTHON" || die "no $PYTHON on PATH; install Python $MIN_PYTHON+ or use --pypi with uv/pipx"
  python_ok "$PYTHON" || die "$PYTHON is older than $MIN_PYTHON"
  PREFIX="${DEEPLOOP_HOME:-$HOME/.local/share/deeploop}"
  BIN_DIR="${DEEPLOOP_BIN_DIR:-$HOME/.local/bin}"
  say "installing into a private venv: $PREFIX/venv"
  "$PYTHON" -m venv "$PREFIX/venv" || die "could not create a venv (on Debian/Ubuntu try: apt install python3-venv)"
  "$PREFIX/venv/bin/python" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || warn "could not upgrade pip; continuing"
  "$PREFIX/venv/bin/python" -m pip install --quiet "$SPEC" || die "pip install failed for: $SPEC"
  mkdir -p "$BIN_DIR"
  ln -sf "$PREFIX/venv/bin/deeploop" "$BIN_DIR/deeploop"
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "$BIN_DIR is not on your PATH; add it with: export PATH=\"$BIN_DIR:\$PATH\"" ;;
  esac
}

if [ "$DRY_RUN" = "1" ]; then
  say "mode: $MODE"
  say "spec: $SPEC"
  exit 0
fi

say "installing deeploop ($MODE)"

INSTALLER="${DEEPLOOP_INSTALLER:-auto}"
case "$INSTALLER" in
  uv) have uv || die "uv not found"; install_with_uv ;;
  pipx) have pipx || die "pipx not found"; install_with_pipx ;;
  venv) install_with_venv ;;
  auto)
    if have uv; then
      say "using uv"
      install_with_uv
    elif have pipx; then
      say "using pipx"
      install_with_pipx
    else
      say "using a private venv (install uv or pipx for a cleaner setup)"
      install_with_venv
    fi
    ;;
  *) die "DEEPLOOP_INSTALLER must be uv, pipx or venv" ;;
esac

if have deeploop; then
  say ""
  say "deeploop installed: $(deeploop --version 2>/dev/null || echo 'installed')"
  say ""
  say "next steps:"
  say "  export DEEPSEEK_API_KEY=sk-...        # or use --provider openrouter|ollama|mock"
  say "  deeploop brief <your-brief-folder>    # derive mission.yaml from BRIEF.md"
  say "  deeploop run <your-brief-folder>      # run it, budget-capped"
else
  warn "installed, but the deeploop command is not on PATH yet; open a new shell or fix PATH"
fi
