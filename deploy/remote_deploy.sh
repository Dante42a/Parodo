#!/usr/bin/env bash
# Выкатка релиза на сервере (запускает .github/workflows/deploy.yml под пользователем deploy; можно и руками).
#
#   bash remote_deploy.sh <sha> <архив-с-кодом.tar.gz>
#
# Схема /opt/parodo:
#   checkpoints/           — веса, общие для всех релизов (кладутся вручную, root)
#   releases/<sha>/        — код релиза + .env с PARODO_TAG=<sha> (тег образов в GHCR)
#   current -> releases/…  — рабочий релиз
#
# Шаги: распаковать → скачать образы → поднять → дождаться /readyz = 200 → переключить current.
# Если модель не стала готовой — вернуть предыдущий релиз и завершиться с ошибкой.
set -euo pipefail

SHA="${1:?Передай sha коммита}"
ARCHIVE="${2:?Передай путь к архиву с кодом}"
ROOT="${PARODO_ROOT:-/opt/parodo}"  # переопределяется для локальной проверки скрипта
RELEASE="$ROOT/releases/$SHA"
KEEP_RELEASES=5
READY_TIMEOUT_S=180

log() { echo "[deploy] $*"; }
compose() {  # compose <папка релиза> <команда…>; проект всегда parodo — контейнеры заменяются, а не дублируются
    local dir="$1"; shift
    docker compose -p parodo --project-directory "$dir" -f "$dir/docker-compose.yml" "$@"
}

PREVIOUS=""  # первый деплой — откатываться некуда
[[ -L "$ROOT/current" ]] && PREVIOUS="$(readlink -f "$ROOT/current")"

rollback() {  # вернуть предыдущий релиз и завершиться с ошибкой
    log "ошибка: $1"
    compose "$RELEASE" logs --tail 50 api || true
    if [[ -n "$PREVIOUS" && "$PREVIOUS" != "$RELEASE" ]]; then
        log "откат на $PREVIOUS"
        compose "$PREVIOUS" up -d --no-build
    fi
    exit 1
}

# 0. Веса на месте — иначе новый релиз гарантированно не станет готовым
for weights in semantic.ckpt yolo.pt; do
    [[ -f "$ROOT/checkpoints/$weights" ]] || { log "нет $ROOT/checkpoints/$weights — положи веса до деплоя"; exit 1; }
done

# 1. Код релиза
log "релиз $SHA (предыдущий: ${PREVIOUS:-нет})"
rm -rf "$RELEASE"
mkdir -p "$RELEASE"
tar -xzf "$ARCHIVE" -C "$RELEASE"
rm -f "$ARCHIVE"
ln -sfn "$ROOT/checkpoints" "$RELEASE/checkpoints"
echo "PARODO_TAG=$SHA" > "$RELEASE/.env"

# 2. Образы из GHCR и запуск
compose "$RELEASE" pull  # при ошибке — выход до остановки старых контейнеров, сервис не прерывается
compose "$RELEASE" up -d --no-build || rollback "не удалось запустить контейнеры"

# 3. Ждём готовности модели
log "жду /readyz (до ${READY_TIMEOUT_S} с)"
ready=false
body=""
for _ in $(seq 1 $((READY_TIMEOUT_S / 3))); do
    code="$(curl -s -o /tmp/parodo-readyz.json -w '%{http_code}' http://127.0.0.1:8000/readyz || true)"
    body="$(cat /tmp/parodo-readyz.json 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then ready=true; break; fi
    # загрузка упала или модель не настроена — ждать дальше бессмысленно
    if [[ "$body" == *'"load_failed"'* || "$body" == *'"not_configured"'* ]]; then break; fi
    sleep 3
done

[[ "$ready" == true ]] || rollback "модель не готова: ${body:-нет ответа}"
log "готово: $body"

# 4. Переключаем current атомарно (ln во временное имя + rename)
ln -sfn "$RELEASE" "$ROOT/current.new"
mv -T "$ROOT/current.new" "$ROOT/current"

# 5. Уборка: последние KEEP_RELEASES релизов и неиспользуемые образы этого репо (Vikunja и др. не трогаем)
# имена релизов = sha, без пробелов — ls здесь безопасен
# shellcheck disable=SC2012
ls -1dt "$ROOT"/releases/*/ | tail -n +$((KEEP_RELEASES + 1)) | while read -r old; do
    [[ "$(readlink -f "$old")" == "$RELEASE" ]] || rm -rf "$old" 2>/dev/null || log "не удалось удалить $old — пропускаю"
done
docker image prune -af --filter "label=org.opencontainers.image.source=https://github.com/Dante42a/Parodo" >/dev/null || true

log "current -> $RELEASE"
