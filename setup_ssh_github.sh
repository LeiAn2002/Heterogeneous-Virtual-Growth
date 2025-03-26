#!/bin/bash

# 设置变量
GITHUB_USER="LeiAn2002"
GITHUB_EMAIL="leian2@illinois.edu"
REPO_URL="git@github.com:$GITHUB_USER/Heterogeneous-Virtual-Growth.git"

echo "🚀 [1/5] 检查 SSH 密钥..."
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "🔑 未找到 SSH 密钥，正在生成..."
    ssh-keygen -t ed25519 -C "$GITHUB_EMAIL" -f ~/.ssh/id_ed25519 -N ""
else
    echo "✅ 已存在 SSH 密钥，跳过生成"
fi

echo ""
echo "📋 [2/5] 请复制以下公钥并粘贴到 GitHub -> Settings -> SSH and GPG Keys 中："
echo "----------------------------------------------------------------"
cat ~/.ssh/id_ed25519.pub
echo "----------------------------------------------------------------"
echo "⏳ 等你添加好后按 Enter 键继续..."
read

echo ""
echo "⚙️ [3/5] 设置 Git 用户信息..."
git config --global user.name "$GITHUB_USER"
git config --global user.email "$GITHUB_EMAIL"

echo ""
echo "🔁 [4/5] 设置远程仓库 URL 为 SSH..."
git remote set-url origin "$REPO_URL"
echo "当前远程仓库地址为："
git remote -v

echo ""
echo "🧪 [5/5] 测试 SSH 连接 GitHub..."
ssh -T git@github.com

echo ""
echo "✅ 设置完成！你现在可以使用 git push/pull 无需输入密码了！"
