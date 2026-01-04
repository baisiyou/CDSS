"""
Gunicorn configuration for Render deployment
"""
import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5003')}"
backlog = 2048

# Worker processes
workers = 1  # Render免费版建议使用1个worker
worker_class = 'sync'
worker_connections = 1000
timeout = 120  # 增加超时时间，因为加载模型可能需要时间
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'cdss_api'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (not needed for Render)
keyfile = None
certfile = None
