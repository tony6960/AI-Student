# 编写自定义异常类MyException练习

> B站公开课 Python 122 期  姓名：AI学生  学号：2026001  任课教师：刘金玉
> 科目：Python 编程　任课教师：刘金玉　上课时间：2026-10-08 15:10
> 任务类型：练习
> 要求：在 ljytest122.py 中完成并运行；异常信息为“异常：输入的不是数值！当前输入的是：”

代码文件：`ljytest122.py`

```python
# ljytest122.py

# 自定义异常类 MyException，继承 Exception
class MyException(Exception):
    def __init__(self, a):
        self.a = a
        super().__init__(a)  # 初始化父类，保证异常信息可以正常显示


# 通过 input() 输入一个值
a = input("请输入一个值？")

# 判断输入内容是否为数值形式
if not a.isdigit():
    # 不是数值，抛出自定义异常，异常信息按题目要求
    raise MyException("异常：输入的不是数值！当前输入的是：")
else:
    print("输入的是数值。")
```

运行方法：

1. 将代码保存为 `ljytest122.py`。
2. 在终端或 PyCharm 中运行：

```bash
python ljytest122.py
```

预期结果：

- 输入 `123`：程序输出 `输入的是数值。`，正常结束。
- 输入 `abc`：程序抛出自定义异常 `MyException`，Traceback 最后显示：

```text
__main__.MyException: 异常：输入的不是数值！当前输入的是：
```

如果老师要求在异常信息后显示当前输入值，只需把抛出异常那一行改成：

```python
raise MyException(f"异常：输入的不是数值！当前输入的是：{a}")
```

B站公开课 Python 122 期  姓名：AI学生  学号：2026001  任课教师：刘金玉


---

## 解题过程

题目要求：在 `ljytest122.py` 中编写程序，自定义异常类 `MyException`，让它继承 `Exception`；通过 `input()` 输入一个值；如果输入内容不是数值，就抛出自定义异常，异常信息为：

```text
异常：输入的不是数值！当前输入的是：
```

已知条件与课堂资料要点：
- 自定义异常类写作 `class MyException(Exception):`
- 课堂板书使用 `input()` 接收字符串，并使用 `a.isdigit()` 判断是否为数值形式。
- `isdigit()` 只能判断“非负整数”形式的字符串，例如 `123` 是数值，`-1`、`3.14`、`abc` 都不是。
- 题面给出的异常信息末尾是“当前输入的是：”，但未明确要求在冒号后拼接当前输入值。课堂板书中 `raise` 使用的是固定异常信息，因此本解答按固定文本处理；若老师要求显示具体输入值，可在代码中将异常信息改为 f-string 拼接。

假设：按课堂要求，用 `isdigit()` 判断数值；异常信息严格使用题目给出的文本。

> 完成人：AI学生　生成时间：2026-10-08 15:17
