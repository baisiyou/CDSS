#!/bin/bash
# 一键启动API服务并打开前端

echo "=========================================="
echo "一键启动CDSS系统"
echo "=========================================="
echo ""

# 1. 停止旧进程
PORT=5003
echo "1. 检查并停止旧进程..."
PID=$(lsof -ti:$PORT 2>/dev/null)
if [ ! -z "$PID" ]; then
    echo "   发现进程 $PID，正在停止..."
    kill -9 $PID 2>/dev/null
    sleep 2
    echo "   ✅ 已停止"
else
    echo "   ✅ 端口未被占用"
fi
echo ""

# 2. 启动API服务（后台）
echo "2. 启动API服务（后台运行）..."
cd "/Users/zrb/Documents/临床决策支持系统（CDSS）"
nohup python3 cdss_api.py > api.log 2>&1 &
API_PID=$!
echo "   API服务PID: $API_PID"
echo "   日志文件: api.log"
echo ""

# 3. 等待API启动
echo "3. 等待API服务启动（最多60秒）..."
SUCCESS=false
for i in {1..30}; do
    sleep 2
    if curl -s --connect-timeout 1 http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "   ✅ API服务已就绪！(等待了 $((i * 2)) 秒)"
        SUCCESS=true
        break
    fi
    if [ $((i % 5)) -eq 0 ]; then
        echo "   等待中... ($i/30) - 已等待 $((i * 2)) 秒"
    fi
done

if [ "$SUCCESS" = false ]; then
    echo "   ⚠️  API服务启动超时"
    echo "   请查看日志: tail -20 api.log"
    echo "   或手动运行: python3 cdss_api.py"
    echo ""
    echo "=========================================="
    exit 1
fi

echo ""

# 4. 打开前端
echo "4. 打开前端界面..."
sleep 1
if [ -f "drug_combination_analyzer.html" ]; then
    open drug_combination_analyzer.html 2>/dev/null || \
    xdg-open drug_combination_analyzer.html 2>/dev/null || \
    echo "   请手动打开: drug_combination_analyzer.html"
    echo "   ✅ 前端已打开"
else
    echo "   ⚠️  未找到 drug_combination_analyzer.html"
fi

echo ""
echo "=========================================="
echo "✅ 系统启动完成！"
echo "=========================================="
echo ""
echo "访问地址:"
echo "  前端界面: drug_combination_analyzer.html (已在浏览器中打开)"
echo "  API健康检查: http://localhost:$PORT/health"
echo "  API文档: http://localhost:$PORT/"
echo ""
echo "查看日志:"
echo "  tail -f api.log"
echo ""
echo "停止服务:"
echo "  kill $API_PID"
echo "  或运行: lsof -ti:$PORT | xargs kill -9"
echo "=========================================="

