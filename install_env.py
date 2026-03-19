from __future__ import annotations

import argparse
import subprocess
import sys


def main(index_url: str | None = None):
    cmds = [
        [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
        [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
    ]
    if index_url:
        cmds[1].extend(['-i', index_url])
    for cmd in cmds:
        print('[执行]', ' '.join(cmd))
        subprocess.run(cmd, check=True)
    print('[OK] 环境安装完成')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='安装银币缺陷检测项目依赖')
    parser.add_argument('--index-url', default=None, help='pip 镜像地址，可选')
    args = parser.parse_args()
    main(args.index_url)
