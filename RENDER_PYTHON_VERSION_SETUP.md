# Render Dashboard 中设置 Python 版本指南

## 方法 1：通过服务设置页面（推荐）

### 步骤：

1. **登录 Render Dashboard**
   - 访问 https://dashboard.render.com
   - 使用您的账户登录

2. **进入您的服务**
   - 在 Dashboard 中找到您的服务（`cdss-api`）
   - 点击服务名称进入服务详情页面

3. **打开设置页面**
   - 点击左侧菜单中的 **"Settings"**（设置）
   - 或点击页面顶部的 **"Settings"** 标签

4. **找到 Python 版本设置**
   - 在 **"Environment"**（环境）部分
   - 找到 **"Python Version"**（Python 版本）下拉菜单
   - 点击下拉菜单

5. **选择 Python 版本**
   - 选择 **"Python 3"** 或 **"3.9"**（如果显示具体版本）
   - 如果只有 "Python 3"，Render 会自动使用最新的 3.9.x 版本

6. **保存更改**
   - 点击页面底部的 **"Save Changes"**（保存更改）按钮
   - Render 会自动触发重新部署

## 方法 2：通过环境变量设置

### 步骤：

1. **进入服务设置**
   - 按照方法 1 的步骤 1-3 进入 Settings 页面

2. **找到环境变量部分**
   - 在 Settings 页面中找到 **"Environment Variables"**（环境变量）部分
   - 点击 **"Add Environment Variable"**（添加环境变量）

3. **添加 Python 版本变量**
   - **Key（键）**: `PYTHON_VERSION`
   - **Value（值）**: `3.9.18` 或 `3.9`
   - 点击 **"Save Changes"**

4. **重新部署**
   - 保存后，Render 会自动触发重新部署
   - 或者手动点击 **"Manual Deploy"** → **"Deploy latest commit"**

## 方法 3：通过 render.yaml（已配置）

您的 `render.yaml` 文件中已经配置了：

```yaml
envVars:
  - key: PYTHON_VERSION
    value: 3.9.18
```

但是，Render 可能不会自动应用这些设置。建议：

1. **在 Dashboard 中手动确认**
   - 按照方法 1 或方法 2 在 Dashboard 中设置
   - 这样可以确保设置生效

2. **检查当前设置**
   - 在 Settings 页面查看当前 Python 版本
   - 如果显示的是 3.13 或其他版本，需要手动更改

## 验证设置

### 检查构建日志：

1. 进入服务的 **"Logs"**（日志）页面
2. 查看最新的构建日志
3. 查找类似这样的行：
   ```
   Python version: 3.9.18
   ```
   或
   ```
   /opt/render/project/src/.venv/lib/python3.9/
   ```

### 如果仍然显示 Python 3.13：

1. **清除缓存并重新部署**
   - 在 Settings 页面找到 **"Clear build cache"**（清除构建缓存）
   - 点击清除缓存
   - 然后手动触发重新部署

2. **检查服务类型**
   - 确保服务类型是 **"Web Service"**
   - 确保环境是 **"Python"**

## 常见问题

### Q: 找不到 Python Version 设置？
A: 确保您的服务类型是 "Web Service" 且环境是 "Python"。某些服务类型可能不显示此选项。

### Q: 设置后仍然使用 Python 3.13？
A: 
1. 清除构建缓存
2. 手动触发重新部署
3. 检查环境变量是否正确设置

### Q: 只有 "Python 3" 选项，没有具体版本？
A: 选择 "Python 3" 即可，Render 会自动使用最新的 3.9.x 版本。

## 快速检查清单

- [ ] 登录 Render Dashboard
- [ ] 进入服务（cdss-api）的 Settings 页面
- [ ] 找到 Python Version 设置
- [ ] 选择 Python 3.9 或 Python 3
- [ ] 保存更改
- [ ] 等待重新部署完成
- [ ] 检查构建日志确认 Python 版本

## 截图位置参考

在 Render Dashboard 中，Python 版本设置通常位于：
- **Settings** → **Environment** → **Python Version**

如果找不到，也可以尝试：
- **Settings** → **Environment Variables** → 添加 `PYTHON_VERSION=3.9.18`

