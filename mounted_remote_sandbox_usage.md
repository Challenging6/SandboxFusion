# Mounted Remote Sandbox 使用说明

这份说明针对当前的 mounted OJ 沙箱接口：`POST /run_oj_cases`。

它的核心思路是：**不再通过网关传整份 testcase 输入输出，而是把题目数据挂载到沙箱侧磁盘，请求里只传 `problem_id + case_ids + code`。**

## 1. 适用场景

适合以下场景：
- 批量评测 C++ / Python3 submission
- 构建 execution matrix
- RL 训练中的 reward 生成
- 需要避免超长 testcase 输入输出经过 HTTP 网关

当前接口支持：
- `language = "cpp"`
- `language = "python3"`
- `language = "java"`

兼容别名：
- `language = "py3"` 等价于 `python3`
- `language = "python"` 等价于 `python3`

## 2. 数据挂载方式

服务端不会动态执行 mount。
所谓 mounted OJ，是指：
- 先在部署层把个人盘或共享盘挂到容器/机器某个目录
- 服务运行时把这个目录当普通本地目录读取

数据根目录解析优先级：
1. 请求中的 `data_dir`
2. 环境变量 `OJ_DATA_ROOT`
3. 配置项 `config.dataset.oj_data_root`

也就是说：
- 想固定数据目录，就在服务部署时设置 `OJ_DATA_ROOT`
- 想临时切目录，就在请求里传 `data_dir`

## 3. 数据目录格式

目录结构：

```text
<data_root>/
  <problem_id>/
    problem.json
    checker.cpp
    cases/
      0.in
      0.out
      1.in
      1.out
```

其中：
- `<problem_id>` 必须和请求里的 `problem_id` 对应
- `problem.json` 描述 time limit、memory limit、checker、case 列表
- `checker.cpp` 是 testlib checker 或普通 checker
- `cases/*.in` / `cases/*.out` 是输入和标准答案

一个典型 `problem.json` 示例：

```json
{
  "problem_id": "1000_A",
  "time_limit_ms": 1000,
  "memory_limit_mb": 256,
  "shared_files": ["testlib.h"],
  "checker": {
    "source": "checker.cpp",
    "files": [],
    "argv": ["input.txt", "output.txt", "answer.txt"]
  },
  "test_cases": [
    { "id": "0", "input": "cases/0.in", "answer": "cases/0.out", "score": 1.0 },
    { "id": "1", "input": "cases/1.in", "answer": "cases/1.out", "score": 1.0 }
  ]
}
```

## 4. 请求接口

接口：

```text
POST /run_oj_cases
Content-Type: application/json
```

请求字段：

- `problem_id: str`
  - 题目目录名
- `case_ids: List[str|int] | str | int`
  - 可以传单个 case id
  - 可以传一组 case id
  - 也可以传 `"all"`，表示执行该题全部 case
- `code: str`
  - 待评测代码
- `language: "cpp" | "python3" | "java"`
  - `cpp`：按 C++17 编译后执行
  - `python3`：保存为 `solution.py`，先做一次 `py_compile` 语法检查，再逐 case 执行
  - `java`：按 `javac` 编译后执行，入口使用 public class 名或 `Main`
  - 兼容传 `py3` / `python`
- `data_dir: Optional[str]`
  - 可选，临时覆盖 mounted 数据目录
- `compile_timeout: float`
  - 编译超时，默认 `30`
- `run_timeout: Optional[float]`
  - 单个 case 的 wall timeout；不传时按题目时限自动计算
- `time_limit_multiplier: float`
  - 当未显式传 `run_timeout` 时，用它乘 `problem.json.time_limit_ms`
  - 默认 `1.0`
- `memory_limit_MB: Optional[int]`
  - 可选，覆盖 `problem.json.memory_limit_mb`
- `enable_msvc_i64_compat: bool`
  - 是否开启 `%I64d -> %lld` 兼容改写
  - 默认 `false`
  - 评测历史 Codeforces 风格 submission 时，建议显式传 `true`
- `include_details: bool`
  - 是否返回每个 case 的 `run_result/check_result`
  - 默认 `false`
  - 默认关闭时，不会返回长 stdout/stderr 细节

## 5. 最小请求示例

### 5.1 C++

```json
{
  "problem_id": "1000_A",
  "case_ids": ["0", "1", "2"],
  "code": "#include <bits/stdc++.h>\nusing namespace std;\nint main(){return 0;}",
  "language": "cpp",
  "data_dir": "/volume/lzchai/mounted_oj_ccplus_1x_filtered"
}
```

### 5.2 Python3

```json
{
  "problem_id": "1000_A",
  "case_ids": ["0", "1", "2"],
  "code": "import sys\n\ndef main():\n    data = sys.stdin.read()\n    # TODO: implement solution\n    sys.stdout.write(data)\n\nif __name__ == '__main__':\n    main()\n",
  "language": "python3",
  "data_dir": "/volume/lzchai/mounted_oj_ccplus_1x_filtered"
}
```

## 6. 执行全部 case 的示例

```json
{
  "problem_id": "1000_A",
  "case_ids": "all",
  "code": "...",
  "language": "python3",
  "data_dir": "/volume/lzchai/mounted_oj_ccplus_1x_filtered",
  "time_limit_multiplier": 2.0,
  "enable_msvc_i64_compat": false
}
```

说明：
- `case_ids = "all"` 会按 `problem.json` 中 `test_cases` 的顺序展开
- `"all"` 必须单独使用，不能和其它 case id 混传

## 7. 返回结果说明

响应主体包含：
- `status`
- `message`
- `problem_id`
- `data_dir`
- `compile_result`
- `checker_compile_result`
- `cases`
- `total_score`
- `max_score`

其中每个 `case` 默认包含：
- `case_id`
- `passed`
- `verdict`
- `score`
- `max_score`
- `input_path`
- `answer_path`
- `run_result`
- `check_result`

注意：
- 默认 `include_details=false` 时，`run_result` 和 `check_result` 会是 `null`
- 这样可以避免把超长 stdout/stderr 再经 HTTP 返回
- 只有显式传 `include_details=true` 时，才会返回详细执行结果

常见 verdict：
- `AC`
- `WA`
- `TLE`
- `RE`
- `CE`
- `CHECKER_CE`
- `ERROR`

## 8. 当前评测默认行为

当前 mounted OJ 路径的关键行为：
- C++ 默认编译标准：`c++17`
- C++ 默认编译参数：`-O2 -DONLINE_JUDGE`
- Java 使用 `javac` 编译一次，再用 JVM 执行每个 case
- Python3 使用沙箱的 `sandbox-runtime` Python 环境执行
- Python3 会先执行 `python -m py_compile solution.py`
- solution 按请求准备一次，再复用到本次请求的所有 case
- checker 按题目编译并执行
- 执行链路已改为 file-backed stdin / checker file path，不再在 Python 层搬运超长 testcase 文本
- 大输出不会再因为 pipe 堵塞制造假 `TLE`

## 9. timeout 说明

如果不显式传 `run_timeout`，单 case 的默认限制按题目时限自动计算：

- `tl_s = time_limit_ms * time_limit_multiplier / 1000`
- `cpu_limit_s = ceil(tl_s)`
- `wall_timeout = max(tl_s * 2 + 2, 10)`

其中：
- `time_limit_multiplier` 默认是 `1.0`
- 如果你在做 execution matrix，通常建议显式传更大的 `time_limit_multiplier`，例如 `2.0`

## 10. 推荐调用习惯

### 10.1 构建 execution matrix
建议：
- C++ 历史 Codeforces 风格 submission 显式传 `enable_msvc_i64_compat=true`
- Python3 不需要设置 `enable_msvc_i64_compat`
- 显式传 `time_limit_multiplier=2.0`
- 默认 `include_details=false`
- 尽量按 case batch 调，而不是永远用 `"all"`

### 10.2 联调异常题
建议：
- 打开 `include_details=true`
- 只跑少量 `case_ids`
- 必要时显式加大 `run_timeout`

## 11. curl 示例

```bash
curl -sS http://<sandbox-host>:8080/run_oj_cases \
  -H 'Content-Type: application/json' \
  -d '{
    "problem_id": "1000_A",
    "case_ids": "all",
    "data_dir": "/volume/lzchai/mounted_oj_ccplus_1x_filtered",
    "code": "#include <bits/stdc++.h>\\nusing namespace std;\\nint main(){return 0;}",
    "language": "cpp",
    "time_limit_multiplier": 2.0,
    "enable_msvc_i64_compat": true,
    "include_details": false
  }'
```

Python3 示例：

```bash
curl -sS http://<sandbox-host>:8080/run_oj_cases \
  -H 'Content-Type: application/json' \
  -d '{
    "problem_id": "1000_A",
    "case_ids": "all",
    "data_dir": "/volume/lzchai/mounted_oj_ccplus_1x_filtered",
    "code": "import sys\\n\\ndef main():\\n    data = sys.stdin.read()\\n    # TODO: implement solution\\n    sys.stdout.write(data)\\n\\nif __name__ == '\''__main__'\'':\\n    main()\\n",
    "language": "python3",
    "time_limit_multiplier": 2.0,
    "include_details": false
  }'
```

## 12. 目前最重要的注意点

- 这条接口的价值在于：**testcase 不走网关传输，只传 `problem_id + case_ids + code`**
- 默认不会返回长执行输出，避免响应膨胀
- 如果做大规模评测，优先用 mounted 数据目录，不要再走旧的 `run_check_code` 大 payload 路径
- 如果需要和当前 execution matrix 评测口径对齐，建议显式打开：
  - C++：`enable_msvc_i64_compat=true`
  - `time_limit_multiplier=2.0`
