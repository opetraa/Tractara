import os
import sys
import glob


def clean_data():
    # 프로젝트 루트 기준 경로 설정
    base_dir = os.getcwd()
    landing_dir = os.path.join(base_dir, "src", "data", "landing")
    ssot_dir = os.path.join(base_dir, "src", "data", "ssot")

    # 1. 옵션 선택
    print("==========================================")
    print("      CLARA-SSoT 데이터 정리 도구")
    print("==========================================")
    print("삭제할 범위를 선택하세요:")
    print("1. Landing Zone 데이터만 삭제 (임시 데이터 초기화)")
    print("2. Landing Zone + SSoT 데이터 모두 삭제 (전체 초기화)")

    choice = input("\n선택 (1/2): ").strip()

    targets = []
    target_desc = ""

    if choice == '1':
        targets = [landing_dir]
        target_desc = "Landing Zone"
    elif choice == '2':
        targets = [landing_dir, ssot_dir]
        target_desc = "Landing Zone + SSoT"
    else:
        print("잘못된 입력입니다. 작업을 취소합니다.")
        sys.exit(1)

    # 2. 최종 확인
    print(f"\n[경고] '{target_desc}' 경로 하위의 모든 .json 파일이 영구적으로 삭제됩니다.")
    confirm = input("정말 진행하시겠습니까? (y/N): ").strip().lower()

    if confirm != 'y':
        print("작업이 취소되었습니다.")
        sys.exit(0)

    # 3. 삭제 실행
    deleted_count = 0
    for target_dir in targets:
        # 재귀적으로 .json 파일 탐색
        pattern = os.path.join(target_dir, "**", "*.json")
        files = glob.glob(pattern, recursive=True)

        for file_path in files:
            try:
                os.remove(file_path)
                # 상대 경로로 출력하여 보기 좋게 표시
                print(f"삭제됨: {os.path.relpath(file_path, base_dir)}")
                deleted_count += 1
            except Exception as e:
                print(f"오류 발생 ({file_path}): {e}")

    print(f"\n총 {deleted_count}개의 파일이 삭제되었습니다.")


if __name__ == "__main__":
    clean_data()
