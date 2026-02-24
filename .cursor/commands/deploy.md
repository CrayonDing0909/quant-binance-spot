@devops 幫我部署最新改動到 Oracle Cloud

按照 devops.md 定義的部署同步流程執行：
1. `git add -A` + `git status`（先讓我確認改動內容）
2. `git commit` + `git push`
3. SSH 到 Oracle Cloud `git pull`
4. 根據改動類型判斷是否需要重啟 runner（重啟前先問我）
