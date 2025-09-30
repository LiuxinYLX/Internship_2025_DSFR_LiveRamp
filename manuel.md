# Commandes

## Mac os

复制文件地址 : [OPTION] + [⌘ Command] + C

自由截图 : [⇧ Shift] + [⌘ Command] + 4 （点击截图可复制）

TextEdit 进入纯文本编辑模式 : [⇧ Shift]  + [⌘ Command] + T

光标选择文本 : 单击想选中文本的起始位置，按[⇧ Shift]不动，单击想选中的结束位置

Vscode 选择python编译器：[⇧ Shift]  + [⌘ Command] + P

### Terminal

textutil -convert txt [file/name]

生成文件夹下的树状结构图:c

`find . -not -path '*/\.*' | sed 's/[^\/]*\//|  /g;s/| *\([^|]\)/|--- \1/'`



## GCP

1. 从本地到平台传文件

gsutil cp -r [local/file/path] [remote/file/path]

Local to Dataproc :

 `gs://lranalytics-frds-endpoint-select-vm/660531/LIUXIN/`

`gs://lranalytics-frds-endpoint-select-vm/769631/LIUXIN/`

Dataproc to Jupyter: `gs://lranalytics-eu-660531-aqim-coderepo/660531/LIUXIN/`

`-m`: en parallèle, plus rapide

`-r`: répertoire

2. 查看项目编号：

`gcloud projects describe YOUR_PROJECT_ID`



## Python

显示所有列

pd.set_option('display.max_columns', None)

显示所有行

pd.set_option('display.max_rows', None)

不截断每列的内容

pd.set_option('display.max_colwidth', None)



## Markdown

换行：在行尾加两个空格



## Git

### 初始化

`git init `

`git add . `

`git commit -m "初始化提交：添加所有文件"`

`git remote add origin https://github.com/Liuxin-ylx/Internship_2025_DSFR_Dataquality.git `

`git branch -M main`

`git push -u origin main`

### 更新代码

1. 确保在 main 分支：  `git checkout main`

2. 同步远程 main 到本地 main：

   `git fetch origin`

   `git reset --hard origin/main`

3. 从 main 创建一个新分支保存第二个版本：

​        `git checkout -b v2`

​	`git add .`

​	`git commit -m "第二个版本"`

​	`git push origin v2`

​	输用户名和密码的时候，密码输入创建的personal token

4. 合并main和branch

   1. 如果能保证分支的代码完全正确，check 到分支，强制push覆盖远程：`git push origin v2:main --force`

   2. 先pull，再rebase：

      `git pull origin main --rebase`
      `git push origin main`

      > 如果遇到冲突，查看冲突的文件：`git status`
      >
      > 手动编辑文件，把冲突部分整理好，只保留需要的最终版本：
      >
      > <<<<<< HEAD
      >
      > #来自远程 GitHub 的版本
      >
      > =======
      >
      > #你本地的版本
      >
      > 6c68a19 (初始化提交：添加所有文件)
      >
      > 修改完冲突的所有文件之后：`git add .`， 再继续rebase

5. 更新本地main

   `git checkout main`

   `git reset --hard v2`

### 删除分支

1. 删除远程分支

​	`git push origin --delete 分支名`

2. 删除本地分支

​	`git branch -d 分支名`

### 版本回退 & 日志查看

1. 日志查看

   `git log`

2. 舍弃 分支上 `commit-id` 之后的本地未提交修改

   `git reset --hard commit-id`

3. 