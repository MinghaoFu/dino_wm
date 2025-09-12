#!/usr/bin/env bash
set -euo pipefail

# ===== 配置部分（可按需改动）=====
BUCKET="mh6-s3"                           # S3 桶
REGION="us-east-2"                         # 桶区域
SRC_DIRS="/home/ubuntu/minghao"            # 要备份的本地目录（可空格分隔多个目录）
BACKUP_ROOT="backups/$(hostname)"          # S3 前缀
LOG_FILE="/var/log/s3-backup.log"
SCRIPT_PATH="/opt/backup/s3-backup.sh"
SERVICE="/etc/systemd/system/s3-backup.service"
TIMER="/etc/systemd/system/s3-backup.timer"

echo "[1/4] 创建备份脚本..."
sudo mkdir -p /opt/backup
sudo tee "${SCRIPT_PATH}" >/dev/null <<'SH'
#!/usr/bin/env bash
set -euo pipefail

BUCKET="mh6-s3"
REGION="us-east-2"
SRC_DIRS="/home/ubuntu/minghao"
BACKUP_ROOT="backups/$(hostname)"
LOG_FILE="/var/log/s3-backup.log"
LOCK_FILE="/tmp/s3-backup.lock"

aws configure set default.region "${REGION}" >/dev/null 2>&1 || true
mkdir -p "$(dirname "${LOG_FILE}")"
exec >>"${LOG_FILE}" 2>&1

# 简单排他锁，避免并发
if ( set -o noclobber; echo $$ > "${LOCK_FILE}" ) 2>/dev/null; then
  trap 'rm -f "${LOCK_FILE}"' EXIT
else
  echo "$(date -Is) [INFO] another backup is running, exit."
  exit 0
fi

echo "$(date -Is) [INFO] backup start -> s3://${BUCKET}/${BACKUP_ROOT}/"
for d in ${SRC_DIRS}; do
  if [[ -d "${d}" ]]; then
    dst="s3://${BUCKET}/${BACKUP_ROOT}/$(echo "${d}" | sed 's#^/##')/"
    echo "$(date -Is) [INFO] sync ${d} -> ${dst}"
    aws s3 sync "${d}" "${dst}" \
      --only-show-errors \
      --no-progress \
      --delete
  else
    echo "$(date -Is) [WARN] skip ${d}, not a directory"
  fi
done
echo "$(date -Is) [INFO] backup done"
SH
sudo chmod +x "${SCRIPT_PATH}"

echo "[2/4] 创建 systemd service..."
sudo tee "${SERVICE}" >/dev/null <<UNIT
[Unit]
Description=S3 periodic backup (EC2 -> S3)
After=network-online.target

[Service]
Type=oneshot
ExecStart=${SCRIPT_PATH}
UNIT

echo "[3/4] 创建 systemd timer (每 5 分钟)..."
sudo tee "${TIMER}" >/dev/null <<'TIMER'
[Unit]
Description=Run S3 backup every 5 minutes

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
TIMER

echo "[4/4] 重新加载并启动定时器..."
sudo systemctl daemon-reload
sudo systemctl enable --now s3-backup.timer
sudo systemctl start s3-backup.service

echo "✅ 已配置完成。备份脚本在 ${SCRIPT_PATH}，日志在 ${LOG_FILE}。"
echo "可以用以下命令检查状态："
echo "  systemctl list-timers --all | grep s3-backup"
echo "  tail -n 50 ${LOG_FILE}"
