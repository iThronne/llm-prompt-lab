import runpy
import sys

if len(sys.path) > 0:
    # 确保把当前执行目录（项目根目录）强行作为第一顺位，彻底锁死相对路径和导包路径
    sys.path.insert(0, sys.path[0])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("错误：请在 PyCharm 的 Parameters 中指定要运行的模块名！例如：src.cli run example")
        sys.exit(1)

    # 动态获取你要运行的模块名（比如 src.cli）
    target_module = sys.argv[1]

    # 把第一项参数（模块名）从参数列表中切除，让后面的参数（如 run example）无缝传给业务代码
    sys.argv.pop(1)

    # 动态启动目标模块
    runpy.run_module(target_module, run_name="__main__")
