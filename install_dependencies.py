# install_dependencies.py
import subprocess
import sys

def run_command(command):
    """
    주어진 커맨드를 실행하고, 실패 시 에러를 발생시킵니다.
    """
    print(f"Running: {command}")
    subprocess.check_call(command, shell=True)

def main():
    # unsloth 설치
    run_command("pip install unsloth")
    
    # 최신 nightly 버전 Unsloth 설치 (기존 unsloth 제거 후 설치)
    run_command("pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git")
    
    # pandas 설치
    run_command("pip install pandas")
    
    # datasets 설치
    run_command("pip install datasets")
    
    print("모든 의존성 설치가 완료되었습니다.")

if __name__ == '__main__':
    main()
