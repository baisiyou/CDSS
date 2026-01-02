// API配置
// Render后端部署地址
// 注意：如果你使用URL参数 ?api=xxx，URL参数会优先使用
// 本地开发时，可以通过URL参数覆盖：?api=http://localhost:5003

// 如果 window.API_BASE_URL 还未定义，则设置默认值
if (typeof window.API_BASE_URL === 'undefined') {
    // ⬇️ Render后端地址 ⬇️
    window.API_BASE_URL = 'https://cdss-kd6u.onrender.com';
    // ⬆️ 生产环境API地址 ⬆️
}

