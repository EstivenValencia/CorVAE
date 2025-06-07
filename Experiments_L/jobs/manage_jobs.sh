#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob   # para que globs vacíos no devuelvan literales

# Como este script vive dentro de jobs/, JOB_DIR es su propio directorio
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$JOB_DIR/pids"
LOG_DIR="$JOB_DIR/logs"

mkdir -p "$PID_DIR" "$LOG_DIR"

# Nombre de este propio script, para poder saltarlo en el bucle
SELF_SCRIPT="$(basename "${BASH_SOURCE[0]}")"

run_jobs() {
  echo "=== Lanzando jobs desde $JOB_DIR ==="
  for scriptpath in "$JOB_DIR"/*_latent_*.sh; do
    scriptname="$(basename "$scriptpath")"
    # Saltarse a sí mismo
    [[ "$scriptname" == "$SELF_SCRIPT" ]] && continue

    name="${scriptname%.sh}"
    # Lanzar dentro de jobs/ para respetar rutas relativas
    nohup bash -c "cd \"$JOB_DIR\" && bash \"$scriptname\"" \
      > "$LOG_DIR/${name}.out" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_DIR/${name}.pid"
    echo "  → $name (PID $pid)"
  done
  echo "Todos los jobs iniciados."
}

stop_jobs() {
  echo "=== Parando jobs listados en $PID_DIR ==="
  any=false
  for pidfile in "$PID_DIR"/*.pid; do
    any=true
    pid="$(< "$pidfile")"
    name="$(basename "$pidfile" .pid)"
    if kill "$pid" >/dev/null 2>&1; then
      echo "  → $name (PID $pid) detenido"
    else
      echo "  → $name: PID $pid no existe"
    fi
    rm -f "$pidfile"
  done
  if ! $any; then
    echo "  ¡No había archivos .pid en $PID_DIR!"
  else
    echo "Todos los PIDs eliminados."
  fi
}

case "${1:-}" in
  -r) run_jobs ;;
  -e) stop_jobs ;;
  *)
    cat <<EOF
Uso: $(basename "$0") [opción]
  -r    Lanzar todos los jobs (se quedan vivos, logs en logs/, pids en pids/)
  -e    Parar todos los jobs previamente arrancados
EOF
    exit 1
    ;;
esac
