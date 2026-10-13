#!/usr/bin/env bash
# Создание пользователя для CD из GitHub Actions (однократно, на сервере).
#
# Запуск:
#   sudo bash setup_deploy_user.sh "ssh-ed25519 AAAA... github-actions-parodo"
#
# Что даёт пользователю:
#   - вход только по ключу (restrict: без форвардинга портов/агента и без pty);
#   - группу docker — для `docker compose` (это фактически root-доступ к хосту: держать ключ только в секретах GitHub);
#   - группу parodo с правом записи в /opt/parodo и /opt/parodo/releases (новые релизы, симлинк current).
# Существующие релизы и /opt/parodo/checkpoints не меняются: владелец остаётся root.
set -euo pipefail

DEPLOY_USER="${DEPLOY_USER:-deploy}"
APP_GROUP="parodo"
APP_DIR="/opt/parodo"
PUBKEY="${1:?Передай открытый ключ первым аргументом}"

[[ $EUID -eq 0 ]] || { echo "Запусти через sudo" >&2; exit 1; }
[[ "$PUBKEY" == ssh-* ]] || { echo "Это не похоже на открытый ключ: $PUBKEY" >&2; exit 1; }
getent group docker >/dev/null || { echo "Группа docker не найдена — Docker не установлен?" >&2; exit 1; }

# Пользователь без пароля: '*' (а не '!') — вход по ключу не блокируется
if ! id "$DEPLOY_USER" &>/dev/null; then
    useradd --create-home --shell /bin/bash --password '*' "$DEPLOY_USER"
    echo "Создан пользователь $DEPLOY_USER"
fi

# Ключ (перезаписывается — повторный запуск меняет ключ, а не добавляет второй)
SSH_DIR="/home/$DEPLOY_USER/.ssh"
install -d -m 700 -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$SSH_DIR"
echo "restrict $PUBKEY" > "$SSH_DIR/authorized_keys"
chown "$DEPLOY_USER:$DEPLOY_USER" "$SSH_DIR/authorized_keys"
chmod 600 "$SSH_DIR/authorized_keys"

# Группы: docker — для compose, parodo — для записи релизов
getent group "$APP_GROUP" >/dev/null || groupadd "$APP_GROUP"
usermod -aG docker,"$APP_GROUP" "$DEPLOY_USER"

# Каталоги релизов: владелец root, группа parodo с правом записи; setgid — новые файлы наследуют группу
install -d "$APP_DIR" "$APP_DIR/releases" "$APP_DIR/checkpoints"
chgrp "$APP_GROUP" "$APP_DIR" "$APP_DIR/releases"
chmod 2775 "$APP_DIR" "$APP_DIR/releases"

echo
echo "Готово: $DEPLOY_USER (группы: $(id -nG "$DEPLOY_USER"))"
ls -ld "$APP_DIR" "$APP_DIR/releases" "$APP_DIR/checkpoints"
echo
echo "Отпечатки ключей хоста — сверь с секретом SSH_KNOWN_HOSTS:"
for k in /etc/ssh/ssh_host_*_key.pub; do ssh-keygen -lf "$k"; done
